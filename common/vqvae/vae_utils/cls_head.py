import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import MixerLayer, FCBlock, BasicBlock

class ClsHead(nn.Module):
    def __init__(self,cfg,in_channels=2048 ,image_size=(224,224)) -> None:
        super().__init__()
        self.conv_channels = cfg["STAGE_I"]["CLS_HEAD"]["conv_channels"]
        self.hidden_dim = cfg["STAGE_I"]["CLS_HEAD"]["hidden_dim"]
        self.conv_num_blocks = cfg["STAGE_I"]["CLS_HEAD"]["conv_num_blocks"]
        self.dilation = cfg["STAGE_I"]["CLS_HEAD"]["dilation"]

        self.num_blocks = cfg["STAGE_I"]["CLS_HEAD"]["num_blocks"]
        self.hidden_inter_dim = cfg["STAGE_I"]["CLS_HEAD"]["hidden_inter_dim"]
        self.token_inter_dim = cfg["STAGE_I"]["CLS_HEAD"]["token_inter_dim"]
        self.dropout = cfg["STAGE_I"]["CLS_HEAD"]["dropout"]

        self.token_num = cfg["STAGE_I"]["CODEBOOK"]["token_num"]
        self.token_class_num = cfg["STAGE_I"]["CODEBOOK"]["token_class_num"]
        
        self.conv_trans = self._make_transition_for_head(
            in_channels, self.conv_channels)
        self.conv_head = self._make_cls_head()
        input_size = (image_size[0]//32)*(image_size[1]//32)
        self.mixer_trans = FCBlock(
            self.conv_channels * input_size,        #256*7
            self.token_num * self.hidden_dim)       #64*64

        self.mixer_head = nn.ModuleList(
            [MixerLayer(self.hidden_dim, self.hidden_inter_dim,
                self.token_num, self.token_inter_dim,  
                self.dropout) for _ in range(self.num_blocks)])
        self.mixer_norm_layer = FCBlock(
            self.hidden_dim, self.hidden_dim)
        
        self.cls_pred_layer = nn.Linear(
                self.hidden_dim, self.token_class_num)

    def forward(self, x):
        B = x.shape[0]      #x:B,2048,7,7
        cls_feat = self.conv_head[0](self.conv_trans(x))

        cls_feat = cls_feat.flatten(2).transpose(2,1).flatten(1)
        cls_feat = self.mixer_trans(cls_feat)
        cls_feat = cls_feat.reshape(B, self.token_num, -1)
        for mixer_layer in self.mixer_head:
            cls_feat = mixer_layer(cls_feat)
        cls_feat = self.mixer_norm_layer(cls_feat)
        cls_logits = self.cls_pred_layer(cls_feat)
        cls_logits = cls_logits.flatten(0,1)
        cls_logits_softmax = cls_logits.clone().softmax(1)

        return cls_logits, cls_logits_softmax

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_cls_head(self):
        feature_convs = []
        feature_conv = self._make_layer(
            BasicBlock,
            self.conv_channels,
            self.conv_channels,
            self.conv_num_blocks,
            self.dilation
        )
        feature_convs.append(feature_conv)
        
        return nn.ModuleList(feature_convs)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(inplanes, planes, 
                stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)