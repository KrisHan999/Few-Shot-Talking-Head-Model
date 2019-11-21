import torch
import torch.nn as nn
from network.buildingBlock import *

class Embedder(nn.Module):
    '''
    input -> [B, 6, 256, 256]
    output -> [B, 512, 1, 1]
    '''
    def __init__(self):

        super(Embedder, self).__init__()

        self.resDown1 = ResidualBlockDown(in_dim=6, out_dim=64)         # [B, 64, 128, 128]
        self.resDown2 = ResidualBlockDown(in_dim=64, out_dim=128)       # [B, 128, 64, 64]
        self.resDown3 = ResidualBlockDown(in_dim=128, out_dim=256)      # [B, 256, 32, 32]

        self.selfAttendion = SelfAttention(in_dim=256)                  # [B, 256, 32, 32]

        self.resDown4 = ResidualBlockDown(in_dim=256, out_dim=512)      # [B, 512, 16, 16]
        self.resDown5 = ResidualBlockDown(in_dim=512, out_dim=512)      # [B, 512, 8, 8]
        self.resDown6 = ResidualBlockDown(in_dim=512, out_dim=512)      # [B, 512, 4, 4]

        # average pooling is propositional to the sum pooling
        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))         # [B, 512, 1, 1]

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, y):

        B, C, H, W = x.shape

        out = torch.cat((x, y), dim=1)

        out = self.resDown1(out)
        out = self.resDown2(out)
        out = self.resDown3(out)

        out = self.selfAttendion(out)

        out = self.resDown4(out)
        out = self.resDown5(out)
        out = self.resDown6(out)

        out = self.avgPool(out)

        out = self.relu(out)

        out = out.view(B, 512, 1)

        return out


class Discriminator(nn.Module):

    def __init__(self, num_person, finetuning=False, wPrime = None):

        super(Discriminator, self).__init__()

        self.resDown1 = ResidualBlockDown(in_dim=6, out_dim=64)         # [B, 64, 128, 128]
        self.resDown2 = ResidualBlockDown(in_dim=64, out_dim=128)       # [B, 128, 64, 64]
        self.resDown3 = ResidualBlockDown(in_dim=128, out_dim=256)      # [B, 256, 32, 32]

        self.selfAttendion = SelfAttention(in_dim=256)                  # [B, 256, 32, 32]

        self.resDown4 = ResidualBlockDown(in_dim=256, out_dim=512)      # [B, 512, 16, 16]
        self.resDown5 = ResidualBlockDown(in_dim=512, out_dim=512)      # [B, 512, 8, 8]
        self.resDown6 = ResidualBlockDown(in_dim=512, out_dim=512)      # [B, 512, 4, 4]

        self.resBlock = ResidualBlock(in_dim=512)                       # [B, 512, 4, 4]

        # average pooling is propositional to the sum pooling
        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))         # [B, 512, 1, 1]

        self.relu = nn.ReLU(inplace=False)

        self.W = nn.Parameter(torch.randn(512, num_person))
        self.w_0 = nn.Parameter(torch.randn(512, 1))
        self.b = nn.Parameter(torch.randn(1))

        self.finetuning = finetuning
        self.wPrime = wPrime

    def initFinetuning(self, wPrime):
        self.finetuning = True
        self.wPrime = wPrime                                        # [B, 512, 1]

    def forward(self, x, y, i=None):

        B, C, H, W = x.shape

        out = torch.cat((x, y), dim=1)

        out = self.resDown1(out)
        out = self.resDown2(out)
        out = self.resDown3(out)

        out = self.selfAttendion(out)

        out = self.resDown4(out)
        out = self.resDown5(out)
        out = self.resDown6(out)

        out = self.resBlock(out)

        out = self.avgPool(out)

        out = self.relu(out)                                            # [B, 512, 1, 1]

        out = out.view(B, 512, 1).transpose(1, 2)                       # [B, 1, 512]

        if self.finetuning:
            print('finetuning')
            out = torch.bmm(out, self.wPrime) + self.b
        else:
            assert i != None, "input person id"
            out = torch.bmm(out, self.W[:, i].transpose(0, 1).unsqueeze(-1) + self.w_0) + self.b

        out = torch.squeeze(out)

        return out


