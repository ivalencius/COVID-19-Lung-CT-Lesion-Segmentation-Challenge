# -*- coding: utf-8 -*-
import monai
import torch
from torchsummary import summary
from torchviz import make_dot

#edited from Monai networks HRNet site at
#https://docs.monai.io/en/stable/_modules/monai/networks/nets/highresnet.html

DEFAULT_LAYER_PARAMS_3D = (
    # initial conv layer
    {"name": "conv_0", "n_features": 16, "kernel_size": 3},
    # residual blocks
    {"name": "res_1", "n_features": 16, "kernels": (3, 3), "repeat": 1},
    {"name": "res_2", "n_features": 32, "kernels": (3, 3), "repeat": 1},
    {"name": "res_3", "n_features": 64, "kernels": (3, 3), "repeat": 1},
    # final conv layers
    {"name": "conv_1", "n_features": 80, "kernel_size": 1},
    {"name": "conv_2", "kernel_size": 1},
)

def HighResNet(num_classes):
    net = monai.networks.nets.HighResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        dropout_prob=0.1,
        acti_type=('leakyrelu', {'inplace': True}),
        layer_params= DEFAULT_LAYER_PARAMS_3D,
        
    )
    return net

if __name__ == '__main__':
    device = 'cpu'
    model = HighResNet(2)
    summary(model, (1, 192, 192, 32), device=device)
    x = torch.rand((1, 1, 192, 192, 32), device=device)
    out = model(x)
    graph = make_dot(out)
    graph.render('HighResNet', view=True)
    
