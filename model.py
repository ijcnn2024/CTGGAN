import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W = x.size()
        q = self.query(x).view(B, -1, W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, W)
        v = self.value(x).view(B, -1, W)
        attn = torch.bmm(q, k)
        attn = torch.nn.functional.softmax(attn, dim=2)
        o = torch.bmm(v, attn.permute(0, 2, 1))
        o = self.gamma * o.view(B, C, W) + x
        return o


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=stride-1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
            nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.residual = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, output_padding=stride-1)

    def forward(self, x):
        return nn.ReLU()(self.block(x) + self.residual(x))

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose1d(input_dim + 2, 256, kernel_size=25, stride=1, padding=0),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            ResidualBlock(256, 128, stride=2),
            ResidualBlock(128, 128),

            SelfAttention(128),

            ResidualBlock(128, 64, stride=2),
            ResidualBlock(64, 32),
            nn.ConvTranspose1d(32, output_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Upsample(1000, mode='linear', align_corners=False) # Upsample to 4000
        )

    def forward(self, z, labels):
        labels = torch.nn.functional.one_hot(labels, 2).float().to(device)
        z = z.view(z.size(0), z.size(1), 1)
        x = torch.cat([z, labels.unsqueeze(2)], 1)
        return self.gen(x)

class ResidualBlockDisc(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlockDisc, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        return nn.LeakyReLU(0.2, True)(self.block(x) + self.residual(x))

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv1d(input_dim + 2, 32, kernel_size=25, stride=1, padding=12, dilation=2),  # dilated convolution
            nn.LeakyReLU(0.2, True),
            ResidualBlockDisc(32, 64, stride=2),
            SelfAttention(64),
            ResidualBlockDisc(64, 128),
            ResidualBlockDisc(128, 256, stride=2),
            nn.Conv1d(256, 1, kernel_size=20, stride=1, padding=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels = torch.nn.functional.one_hot(labels, 2).float().view(-1, 2, 1).expand(-1, 2, x.size(2)).to(device)
        x = torch.cat([x, labels], 1)
        return self.disc(x)