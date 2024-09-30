import torch

from src.DECSNet import DECSNet


device = 'cuda'

Net = DECSNet(3, 64, 512, 4, 'resnet50', True, [256, 512, 1024, 2048], [64, 128, 256, 512], 1).to(device)

x = torch.ones(4, 3, 256, 256).to(device)

output = Net(x)

print(output.shape)
