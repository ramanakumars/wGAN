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

        self.initial_size = int(initial_size)
        self.input_filt = input_filt
        self.n_layers = n_layers
        self.final_size = final_size
        self.out_channels = out_channels
        self.norm = norm

        super().__init__(
            nn.Linear(self.n_z, self.initial_size * self.initial_size * self.input_filt),
            nn.LeakyReLU(0.2, True),
            Rearrange('b (h w z) -> b z h w', h=self.initial_size, w=self.initial_size, z=input_filt),
            *layers,
            nn.Conv2d(prev_filt, out_channels, (5, 5), stride=(1, 1), padding=2),
            nn.Sigmoid()
        )


class Discriminator(nn.Module):
    def __init__(self, in_channels, n_layers=5, input_size=256):
        super().__init__()

        prev_filt = 8
        layers = []
        for i in range(n_layers):
            layers.append(DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False))
            prev_filt = prev_filt * 2
            input_size = input_size / 2

        self.conv_layers = nn.Sequential(*layers)
        self.rearrange = Rearrange('b z h w -> b (z h w)')
        self.predict = nn.Linear(int(input_size) * int(input_size) * prev_filt, 1)

    def forward(self, x):
        return self.predict(self.rearrange(self.conv_layers(x)))

    def get_features(self, x):
        return self.rearrange(self.conv_layers(x))


class Encoder(nn.Sequential):
    def __init__(self, n_z, input_channels=3, input_filt=512, input_size=256, n_layers=5, norm=False):
        layers = []

        initial_filt = int(input_filt / 2**n_layers)
        prev_filt = initial_filt
        print(prev_filt)
        for _ in range(n_layers):
            layers.append(DownConvLayer(prev_filt, int(prev_filt * 2), activation='leakyrelu', norm=norm,
                                        kernel_size=(6, 6), stride=(2, 2), padding=2))
            prev_filt = prev_filt * 2

        initial_size = input_size / 2 ** n_layers
        if initial_size % 1 != 0:
            raise ValueError(f"Cannot create a model to produce a {input_size} x {input_size} image with {n_layers} layers")

        self.initial_size = int(initial_size)
        super().__init__(
            nn.Conv2d(input_channels, initial_filt, (5, 5), stride=(1, 1), padding=2),
            *layers,
            Rearrange('b z h w -> b (z h w)'),
            nn.Linear(self.initial_size * self.initial_size * input_filt, n_z)
        )

    @classmethod
    def from_generator(cls, generator: Generator):
        encoder = cls(generator.n_z, generator.out_channels, generator.input_filt, generator.final_size, generator.n_layers, generator.norm)

        return encoder
