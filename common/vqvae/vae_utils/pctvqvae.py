import torch.nn as nn
import torch.nn.modules.transformer
import torch.nn.functional as F
import torch.distributed as dist
from .modules import MixerLayer

'''
encoder=dict(
    drop_rate=0.2,
    num_blocks=4,
    hidden_dim=512,
    token_inter_dim=64,
    hidden_inter_dim=512,
    dropout=0.0,
),
decoder=dict(
    num_blocks=1,
    hidden_dim=32,
    token_inter_dim=64,
    hidden_inter_dim=64,
    dropout=0.0,
),
codebook=dict(
    token_num=34,       24 or 48?
    token_dim=512,
    token_class_num=2048,
    ema_decay=0.9,
),
'''

class PCTVQVAE(nn.Module):
    def __init__(self,
                 cfg,
                 num_joints=24,
                 input_dim = 9,
                 ) -> None:
        super().__init__()

        self.cfg = cfg
        self.num_joints = num_joints
        self.drop_rate = cfg["STAGE_I"]["ENCODER"]["drop_rate"]
        self.enc_num_blocks = cfg["STAGE_I"]["ENCODER"]["num_blocks"]
        self.enc_hidden_dim = cfg["STAGE_I"]["ENCODER"]["hidden_dim"]
        self.enc_token_inter_dim = cfg["STAGE_I"]["ENCODER"]["token_inter_dim"]
        self.enc_hidden_inter_dim = cfg["STAGE_I"]["ENCODER"]["hidden_inter_dim"]
        self.enc_dropout = cfg["STAGE_I"]["ENCODER"]["dropout"]

        self.dec_num_blocks = cfg["STAGE_I"]["DECODER"]["num_blocks"]
        self.dec_hidden_dim = cfg["STAGE_I"]["DECODER"]["hidden_dim"]
        self.dec_token_inter_dim = cfg["STAGE_I"]["DECODER"]["token_inter_dim"]
        self.dec_hidden_inter_dim = cfg["STAGE_I"]["DECODER"]["hidden_inter_dim"]
        self.dec_dropout = cfg["STAGE_I"]["DECODER"]["dropout"]

        self.token_num = cfg["STAGE_I"]["CODEBOOK"]["token_num"]
        self.token_class_num = cfg["STAGE_I"]["CODEBOOK"]["token_class_num"]
        self.token_dim = cfg["STAGE_I"]["CODEBOOK"]["token_dim"]
        self.decay = cfg["STAGE_I"]["CODEBOOK"]["ema_decay"]

        self.start_embed = nn.Linear(
            input_dim, self.enc_hidden_dim)
        
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim, 
                self.num_joints, self.enc_token_inter_dim,
                self.enc_dropout) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        
        self.token_mlp = nn.Linear(
            self.num_joints, self.token_num)
        self.feature_embed = nn.Linear(
            self.enc_hidden_dim, self.token_dim)

        self.register_buffer('codebook', 
            torch.empty(self.token_class_num, self.token_dim))
        self.codebook.data.normal_()
        self.register_buffer('ema_cluster_size', 
            torch.zeros(self.token_class_num))
        self.register_buffer('ema_w', 
            torch.empty(self.token_class_num, self.token_dim))
        self.ema_w.data.normal_()        
        
        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, input_dim)

    def forward(self, pose):

        bs, P, _ = pose.shape   #B, 24, 9
        assert P == self.num_joints
        #------------Encode--------------
        encode_feat = self.start_embed(pose)    # P, H
        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        encode_feat = encode_feat.transpose(2, 1)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
        encode_feat = self.feature_embed(encode_feat).flatten(0,1)

        distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
            + torch.sum(self.codebook**2, dim=1) \
            - 2 * torch.matmul(encode_feat, self.codebook.t())
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.token_class_num, device=pose.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        #--------------Update Codebook------------------
        part_token_feat = torch.matmul(encodings, self.codebook)
        dw = torch.matmul(encodings.t(), encode_feat.detach())
        # sync
        n_encodings, n_dw = encodings.numel(), dw.numel()
        encodings_shape, dw_shape = encodings.shape, dw.shape
        combined = torch.cat((encodings.flatten(), dw.flatten()))
        dist.all_reduce(combined) # math sum
        sync_encodings, sync_dw = torch.split(combined, [n_encodings, n_dw])
        sync_encodings, sync_dw = \
            sync_encodings.view(encodings_shape), sync_dw.view(dw_shape)

        self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                (1 - self.decay) * torch.sum(sync_encodings, 0)
        
        n = torch.sum(self.ema_cluster_size.data)
        self.ema_cluster_size = (
            (self.ema_cluster_size + 1e-5)
            / (n + self.token_class_num * 1e-5) * n)
        
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * sync_dw
        self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(1)
        e_latent_loss = F.mse_loss(part_token_feat.detach(), encode_feat)
        part_token_feat = encode_feat + (part_token_feat - encode_feat).detach()


        #------------------Decode--------------------
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)        
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_pose = self.recover_embed(decode_feat)

        return recoverd_pose, encoding_indices, e_latent_loss

    def get_latent_code(self, pose):
        encode_feat = self.start_embed(pose)    # P, H
        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        encode_feat = encode_feat.transpose(2, 1)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
        encode_feat = self.feature_embed(encode_feat).flatten(0,1)
        distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
            + torch.sum(self.codebook**2, dim=1) \
            - 2 * torch.matmul(encode_feat, self.codebook.t())
            
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
    
    def get_latent_feat(self, pose):
        encode_feat = self.start_embed(pose)    # P, H
        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)
        encode_feat = encode_feat.transpose(2, 1)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
        encode_feat = self.feature_embed(encode_feat).flatten(0,1)
        distances = torch.sum(encode_feat**2, dim=1, keepdim=True) \
            + torch.sum(self.codebook**2, dim=1) \
            - 2 * torch.matmul(encode_feat, self.codebook.t())
            
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self.token_class_num, device=pose.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        part_token_feat = torch.matmul(encodings, self.codebook)
        return part_token_feat

    def get_decode_pose(self, cls_logits):
        bs = cls_logits.shape[0] // self.token_num      # //  self.token_num??
        part_token_feat = torch.matmul(cls_logits, self.codebook)
        part_token_feat = part_token_feat.view(bs, -1, self.token_dim)
        
        part_token_feat = part_token_feat.transpose(2,1)
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_pose = self.recover_embed(decode_feat)
        return recoverd_pose

    def get_pred_latent_feat(self, cls_logits):
        # bs = cls_logits.shape[0] // self.token_num      # //  self.token_num??
        part_token_feat = torch.matmul(cls_logits, self.codebook)

        return part_token_feat