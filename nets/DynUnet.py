import monai
import torch
from torchsummary import summary
from torchviz import make_dot


def DynUnet(num_classes):
    """
    # Adapted from: https://github.com/Project-MONAI/tutorials/blob/master/modules/dynunet_pipeline/create_network.py 
    
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
    """
    sizes = [192, 192, 32]
    spacings = [0.79, 0.79, 4.8] # pixel resolution (?)
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [
            2 if ratio <= 2 and size >= 8 else 1
            for (ratio, size) in zip(spacing_ratio, sizes)
        ]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    
    #print(strides)
    #print(kernels)
    #[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    #[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    
    # From https://arxiv.org/pdf/2110.03352.pdf
    #The first and last kernel and stride values of the input sequences 
    #are used for input block and bottleneck respectively, 
    #and the rest value(s) are used for downsample and upsample blocks.
    # Also see: https://arxiv.org/pdf/1904.08128.pdf
    # below is from pg 35/55 of that paper
    #kernels = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
    #strides = [[2,2,2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]
    #filters = [64, 96, 128, 192, 256, 384, 512][: len(strides)] # output channels for each block
    print('Kernels')
    print(kernels)
    print('Strides')
    print(strides)
    net = monai.networks.nets.DynUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        dropout=0.1,
        res_block=True,
        #deep_supervision=True, # alters output shape
        trans_bias=False
    )
    return net

if __name__ == '__main__':
    device = 'cpu'
    model = DynUnet(2)
    summary(model, (1, 192, 192, 32), device=device)
    x = torch.rand((1, 1, 192, 192, 32), device=device)
    out = model(x)
    graph = make_dot(out)
    graph.render('DynUnet', view=True)
    