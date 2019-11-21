import torch
import torch.nn as nn

class AdaIn(nn.Module):
    '''
    input -> [B, C, H, W]
    output -> [B, C, H, W]
    '''
    def __init__(self):

        super(AdaIn, self).__init__(self)

        self.eps = 1e-5

    def forward(self, feature, mean_style, std_style):
        B, C, H, W = feature.shape

        # get the channel-wise feature
        feature = feature.view(B, C, H*W)
        # get the mean and std of feature channel-wisely
        mean_feature = torch.mean(feature, dim=2).view(B, C, 1)
        std_feature = (torch.std(feature, dim=2) + self.eps).view(B, C, 1)

        adaIn_feature = ((feature - mean_feature) / std_feature) * std_style + mean_style
        # change back to the input shape
        adaIn_feature = adaIn_feature.view(B, C, H, W)

        return adaIn_feature


class SelfAttention(nn.Module):
    '''
    input -> [B, C, H, W]
    output -> [B, C, H. W]
    '''
    def __init__(self, in_dim):

        super(SelfAttention, self).__init__()

        self.conv_f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.conv_g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.conv_h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv_o = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.randn(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):

        B, C, H, W = x.shape

        f = self.conv_f(x).view(B, -1, H*W)                                                    # [B, C//8, N]
        g = self.conv_g(x).view(B, -1, H*W).permute(0, 2, 1)                                   # [B, N, C//8]
        h = self.conv_h(x).view(B, -1, H*W)                                                    # [B, C, N]
        attention = self.softmax(torch.bmm(g, f))                                              # [B, N, N]
        # permute here is because of the paper
        o = self.conv_o(torch.bmm(h, attention.permute(0, 2, 1)).view(B, C, H, W))             # [B, C, H, W]
        y = self.gamma * o + x

        return y


# Based on [Large Scale GAN Training for High Fidelity Natural Image Synthesis].
class ResidualBlockDown(nn.Module):
    '''
    input -> [B, C_in, H, W]
    output -> [B, C_out, H//2, W//2]
    '''
    def __init__(self, in_dim, out_dim):

        super(ResidualBlockDown, self).__init__()

        self.convl = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1))

        self.convr1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=1))
        self.convr2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=1))

        self.avgPool = nn.AvgPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        out_res = self.convl(x)
        out_res = self.avgPool(out_res)

        out = self.relu(x)
        out = self.convr1(out)
        out = self.relu(out)
        out = self.convr2(out)
        out = self.avgPool(out)

        out = out + out_res

        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_dim):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1))
        self.instanceNorm1 = nn.InstanceNorm2d(num_features=in_dim, affine=True)

        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=1))
        self.instanceNorm2 = nn.InstanceNorm2d(num_features=in_dim, affine=True)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        out_res = x

        out = self.conv1(x)
        out = self.instanceNorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.instanceNorm2(out)
        out = out + out_res
        out = self.relu(out)

        return out


