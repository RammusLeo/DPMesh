import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from omegaconf import OmegaConf
# from ldm.util import instantiate_from_config
import torch.nn.functional as F
import pdb
from cldm.model import create_model, load_state_dict
from lora_diffusion import inject_trainable_lora
from cldm.module_utils import UNetWrapper
from common.utils.lora_utils import set_lora

class VPDEncoder(nn.Module):
    def __init__(self, out_dim=2048, ldm_prior=[320, 670, 1280+1310]):
        # 320, 640, 1280+1280
        super().__init__()

        self.outmodules = nn.ModuleList([
                nn.Sequential(
                nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
                nn.GroupNorm(16, ldm_prior[0]),
                nn.ReLU(),
                nn.Conv2d(ldm_prior[0], ldm_prior[0], 3, stride=2, padding=1),
            ),
                nn.Sequential(
                nn.Conv2d(ldm_prior[1], ldm_prior[1], 3, stride=2, padding=1),
            ),
                nn.Sequential(
                nn.Conv2d(sum(ldm_prior), out_dim, 1),
                nn.GroupNorm(16, out_dim),
                nn.ReLU(),
            )
        ])
        self.apply(self._init_weights)
        self.sd_model = create_model('./common/vpd/cldm_v15.yaml').cpu()
        ckpt = load_state_dict('./common/vpd/control_sd15_openpose.pth', location='cpu')
        ckpt.pop("control_model.input_blocks.0.0.weight",None)
        # for k in list(ckpt.keys()):
        #     if "control_model.input_hint_block" in k:
        #         ckpt.pop(k, None)

        a,b = self.sd_model.load_state_dict(
            ckpt, strict=False)
        print("ldm missing keys:{}".format(a))
        print("ldm unexpected keys:{}".format(b))
        self.unet_lora_params = self.sd_model.unet_lora_params
        self.encoder_vq = self.sd_model.first_stage_model
        self.unet = UNetWrapper(self.sd_model, use_attn=True)
        # self.unet = self.sd_model
        # self.unet_lora_params, self.train_names = inject_trainable_lora(self.sd_model,r=8)
        # self.unet.requires_grad_(False)
        # self.unet_lora_params, self.train_names = inject_trainable_lora(self.unet)
        # self.sd_model.model = None
        # self.sd_model.first_stage_model = None
        # del self.sd_model.cond_stage_model
        del self.encoder_vq.decoder
        # del self.unet.model.diffusion_model.out
        for param in self.encoder_vq.parameters():
            param.requires_grad = False
        for param in self.unet.unet.cond_stage_model.parameters():
            param.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, feats):
        x =  self.ldm_to_net[0](feats[0])
        for i in range(3):
            if i > 0:
                x = x + self.ldm_to_net[i](feats[i])
            x = self.layers[i](x)
            x = self.upsample_layers[i](x)
        return self.out_conv(x)

    def forward(self, x, control,context):
        B,C,H,W = x.shape
        with torch.no_grad():
            latents = self.encoder_vq.encode(x).mode().detach()
        # c_crossattn = self.sd_model.get_unconditional_conditioning(B)
        t = torch.ones((x.shape[0],), device=x.device).long()
        cond = {"c_concat":[control],
                "c_crossattn": [context]}
        outs = self.unet(latents, t,cond = cond)
        feats = [outs[0], outs[1], torch.cat([outs[2],F.interpolate(outs[3], scale_factor=2)], dim=1)]
        # feats = [torch.cat([F.interpolate(outs[0], scale_factor=2), outs[1]], dim=1),outs[2],outs[3]]
        # 1280,4,4 1280,8,8 640,16,16 320,32,32
        y = torch.cat([self.outmodules[0](feats[0]), self.outmodules[1](feats[1]), feats[2]], dim=1)      #B, 3520,8,8
        # y = torch.cat([feats[0], self.outmodules[1](feats[1]), self.outmodules[0](feats[2])], dim=1)
        y = self.outmodules[2](y)           #B, 1024, 8, 8 
        return y

if __name__ == "__main__":
    inputs = torch.rand((4,3,256,256)).cuda()
    control = torch.rand((4,30,64,64)).cuda()
    vpdencoder = VPDEncoder().cuda()
    outputs = vpdencoder(inputs,control)
    print(outputs.shape)