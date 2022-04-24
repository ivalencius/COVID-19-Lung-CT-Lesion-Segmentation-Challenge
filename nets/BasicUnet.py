import monai
import torch
from torchsummary import summary
from torchviz import make_dot

def BasicUnet(num_classes):
    net = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )
    return net

if __name__ == '__main__':
    device = 'cpu'
    model = BasicUnet(2)
    summary(model, (1, 192, 192, 32), device=device)
    x = torch.rand((1, 1, 192, 192, 32), device=device)
    out = model(x)
    graph = make_dot(out)
    graph.render('BasicUnet', view=True)
    