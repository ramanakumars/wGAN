from torch import nn
from einops.layers.torch import Rearrange


class UpConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation, norm=True, **kwargs):
        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)

        layers = [nn.ConvTranspose2d(in_channels, out_channels, **kwargs)]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [activation]

        super().__init__(*layers)


class DownConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation, norm=False, **kwargs):
        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)

        layers = [nn.Conv2d(in_channels, out_channels, **kwargs)]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [activation]

        super().__init__(*layers)


class Generator(nn.Sequential):
    def __init__(self, n_z, input_filt=512, norm=False, n_layers=5, out_channels=3, final_size=256):
        self.n_z = n_z

        layers = []

        prev_filt = input_filt
        for _ in range(n_layers):
            layers.append(UpConvLayer(prev_filt, int(prev_filt / 2), activation='leakyrelu', norm=norm,
                                      kernel_size=(6, 6), stride=(2, 2), padding=2))
            prev_filt = int(prev_filt / 2)

        initial_size = final_size / 2 ** n_layers
        if initial_size % 1 != 0:
            raise ValueError(f"Cannot create a model to produce a {final_size} x {final_size} image with {n_layers} layers")

        initial_size = int(initial_size)

        super().__init__(
            nn.Linear(n_z, initial_size * initial_size * input_filt),
            nn.LeakyReLU(0.2, True),
            Rearrange('b (h w z) -> b z h w', h=initial_size, w=initial_size, z=input_filt),
            *layers,
            nn.Conv2d(prev_filt, out_channels, (5, 5), stride=(1, 1), padding=2),
            nn.Sigmoid()
        )


class Discriminator(nn.Sequential):
    def __init__(self, in_channels, n_layers=5, input_size=256):
        prev_filt = 8
        layers = []
        for i in range(n_layers):
            layers.append(DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False))
            prev_filt = prev_filt * 2
            input_size = input_size / 2

        super().__init__(
            *layers,
            Rearrange('b z h w -> b (z h w)'),
            nn.Linear(int(input_size) * int(input_size) * prev_filt, 1)
        )
