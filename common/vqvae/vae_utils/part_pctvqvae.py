import torch.nn as nn
import torch.nn.modules.transformer
import torch.nn.functional as F
import torch.distributed as dist
from .modules import MixerLayer
# from .sub_vqvae import SubVQVAE
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

class PART_PCTVQVAE(nn.Module):
    def __init__(self,
                 cfg,
                 num_joints=23,
                 ) -> None:
        super().__init__()

        self.cfg = cfg
        self.num_joints = num_joints

        self.dec_num_blocks = cfg.STAGE_I.DECODER.num_blocks
        self.dec_hidden_dim = cfg.STAGE_I.DECODER.hidden_dim
        self.dec_token_inter_dim = cfg.STAGE_I.DECODER.token_inter_dim
        self.dec_hidden_inter_dim = cfg.STAGE_I.DECODER.hidden_inter_dim
        self.dec_dropout = cfg.STAGE_I.DECODER.dropout

        self.token_num = cfg.STAGE_I.CODEBOOK.token_num*5
        self.token_class_num = cfg.STAGE_I.CODEBOOK.token_class_num
        self.token_dim = cfg.STAGE_I.CODEBOOK.token_dim
        self.decay = cfg.STAGE_I.CODEBOOK.ema_decay

        self.decoder_token_mlp = nn.Linear(
            self.token_num, self.num_joints)
        self.decoder_start = nn.Linear(
            self.token_dim, self.dec_hidden_dim)

        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                self.num_joints, self.dec_token_inter_dim, 
                self.dec_dropout) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)

        self.recover_embed = nn.Linear(self.dec_hidden_dim, 9)

        self.parts = ["left_leg_indices","right_leg_indices","left_arm_indices","right_arm_indices","torso_indices"]
        self.partdicts = {
            "left_leg_indices": [0,3,6,9],
            "right_leg_indices": [1,4,7,10],
            "left_arm_indices": [12,15,17,19,21],
            "right_arm_indices": [13,16,18,20,22],
            "torso_indices": [2,5,8,11,14]
        }
        
        # self.encoders = nn.ModuleList()
        # self.partlens = [4,4,5,5,5]
        # for partlen in self.partlens:
        #     self.encoders.append(SubVQVAE(cfg,partlen))


    def forward(self, pose):
        
        bs, P, _ = pose.shape   #B, 23, 9
        assert P == self.num_joints, ValueError(bs,P)
        #------------Encode--------------
        part_token_feat = torch.zeros((bs,self.token_num,self.token_dim)).to(pose)
        e_latent_loss = 0
        for part_idx, body_parts in enumerate(self.parts):
            part_indices = self.partdicts[body_parts]
            sub_pose = pose[:,part_indices,:]
            sub_token_feat,part_latent_loss = self.encoders[part_idx](sub_pose)
            part_token_feat[:,part_idx*8:part_idx*8+8,:]=sub_token_feat
            e_latent_loss+=part_latent_loss
        part_token_feat = part_token_feat.transpose(2,1)    #B, 256, 40
        part_token_feat = self.decoder_token_mlp(part_token_feat).transpose(2,1)
        decode_feat = self.decoder_start(part_token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_pose = self.recover_embed(decode_feat)
        return recoverd_pose, e_latent_loss

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