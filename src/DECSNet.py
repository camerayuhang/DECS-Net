import torch
import torch.nn as nn
from encoder import Encoder
from decoder import decoder


class DECSNet(nn.Module):
    def __init__(self, in_channel, out_channel, resolution, patchsz, mode, pretrained, cnn_dim, transform_dim, num_class):
        super().__init__()
        self.encoder = Encoder(in_channel, out_channel, resolution, patchsz, mode, pretrained, cnn_dim, transform_dim)
        self.decoder = decoder(num_class, transform_dim)

    def forward(self, x):
        o1, o2, o3, o4 = self.encoder(x)
        out = self.decoder(o1, o2, o3, o4)

        return out


if __name__ == '__main__':
    device = 'cuda'

    Net = DECSNet(3, 64, 512, 4, 'resnet50', True, [256, 512, 1024, 2048], [64, 128, 256, 512], 1).to(device)

    x = torch.ones(4, 3, 256, 256).to(device)

    output = Net(x)

    print(output.shape)
