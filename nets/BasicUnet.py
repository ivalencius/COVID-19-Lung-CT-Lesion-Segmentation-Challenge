import monai
import torch

def net(num_classes):
    net = monai.networks.nets.BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
    )
    return net

if __name__ == '__main__':
    # Add later
    '''
    device = 'cpu'
    m = VGG16J(cut_layers=1, down_conv_z=False, up_conv_z2=True).to(device)
    summary(m, (1, 192, 192, 32), device=device)
    x = torch.rand((1, 1, 192, 192, 32), device=device)
    out = m(x)
    make_dot(out).render('vgg16j', view=True)
    '''