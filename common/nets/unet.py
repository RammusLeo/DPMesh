""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self,in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        self.norm = nn.GroupNorm(2,out_channels)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class UpBlock(nn.Module):
    def __init__(self,in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        self.norm = nn.GroupNorm(2,out_channels)
        self.relu = nn.ReLU()
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self,in_channels, out_channels) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels,out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        self.downblocks = nn.ModuleList([
            DownBlock(in_channels=in_channels,out_channels=8),
            DownBlock(in_channels=8,out_channels=8),
            DownBlock(in_channels=8,out_channels=16),
            DownBlock(in_channels=16,out_channels=16),
            DownBlock(in_channels=16,out_channels=16),
        ])
        self.mlps = nn.ModuleList([
            MLPBlock(16,128),
            MLPBlock(128,128),
            MLPBlock(128,256),
        ])
        self.upblocks = nn.ModuleList([
            UpBlock(in_channels=256, out_channels=16),
            UpBlock(in_channels=16, out_channels=16),
            UpBlock(in_channels=16, out_channels=8),
            UpBlock(in_channels=8, out_channels=8),
            UpBlock(in_channels=8, out_channels=8),
        ])
        self.head = nn.Conv2d(in_channels=8,
                              out_channels=n_classes,
                              kernel_size=1,
                              stride=1,
                              padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        for downblock in self.downblocks:
            x = downblock(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        for mlp in self.mlps:
            x = mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        for upblock in self.upblocks:
            x = upblock(x)
        x = self.head(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = UNet(in_channels=3,n_classes=5)
    x = torch.rand((2,3,256,256))
    y = model(x)
    print(y.shape)
    