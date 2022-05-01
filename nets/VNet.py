# -*- coding: utf-8 -*-
import monai
import torch
from torchsummary import summary
from torchviz import make_dot


def VNet(num_classes):
    net = monai.networks.nets.VNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        dropout_dim = 1,
        dropout_prob=0.1
    )
    return net

if __name__ == '__main__':
    device = 'cpu'
    model = VNet(2)
    summary(model, (1, 192, 192, 32), device=device)
    x = torch.rand((1, 1, 192, 192, 32), device=device)
    out = model(x)
    graph = make_dot(out)
    graph.render('VNet', view=True)
    