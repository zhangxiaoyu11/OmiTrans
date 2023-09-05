import torch
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


# GENERATOR

# For one dimensional U-Net generator
class UNetDe(nn.Module):
    """
    Create a 1D U-Net network (decreasing skip connection)
    """
    def __init__(self, input_chan_num, output_chan_num, output_dim, filter_num=64, kernel_size=9,
                 norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0):
        """
        Construct a one dimensional U-Net
        Parameters:
            input_chan_num (int)  -- the number of channels in input omics data
            output_chan_num (int) -- the number of channels in output omics data
            output_dim (int)      -- the dimension of the output omics data
            filter_num (int)      -- the number of filters in the first convolution layer
            kernel_size (int)     -- the kernel size of convolution layers
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
        """
        super(UNetDe, self).__init__()
        self.input_chan_num = input_chan_num
        self.output_chan_num = output_chan_num
        self.output_dim = output_dim

        # Encoder
        # 1 -> 64 deal with the input data the first double convolution layer
        self.input_conv = DoubleConv1D(input_chan_num, filter_num, kernel_size=kernel_size, norm_layer=norm_layer,
                                       leaky_slope=leaky_slope)
        # 64 -> 128
        self.down_sample_1 = DownSample(filter_num, filter_num * 2, down_ratio=4, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 128 -> 256
        self.down_sample_2 = DownSample(filter_num * 2, filter_num * 4, down_ratio=4, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 256 -> 512
        self.down_sample_3 = DownSample(filter_num * 4, filter_num * 8, down_ratio=4, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 512 -> 1024
        self.down_sample_4 = DownSample(filter_num * 8, filter_num * 16, down_ratio=4, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)

        # Decoder
        # 1024 -> 512
        self.up_sample_1 = UpSampleDe(filter_num * 16, filter_num * 8, up_ratio=2, skip_ratio=2, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 512 -> 256
        self.up_sample_2 = UpSampleDe(filter_num * 8, filter_num * 4, up_ratio=2, skip_ratio=4, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 256 -> 128
        self.up_sample_3 = UpSampleDe(filter_num * 4, filter_num * 2, up_ratio=2, skip_ratio=8, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 128 -> 64
        self.up_sample_4 = UpSampleDe(filter_num * 2, filter_num, up_ratio=5, skip_ratio=6, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 64 -> 1
        self.output_conv = OutputConv(filter_num, output_chan_num)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down_sample_1(x1)
        x3 = self.down_sample_2(x2)
        x4 = self.down_sample_3(x3)
        x5 = self.down_sample_4(x4)
        y1 = self.up_sample_1(x5, x4)
        y2 = self.up_sample_2(y1, x3)
        y3 = self.up_sample_3(y2, x2)
        y4 = self.up_sample_4(y3, x1)
        y5 = self.output_conv(y4)
        # Let the output dim the same as targeting dim
        output = y5[:, :, 0:self.output_dim]
        return output


class UNetIn(nn.Module):
    """
    Create a 1D U-Net network
    """
    def __init__(self, input_chan_num, output_chan_num, output_dim, filter_num=64, kernel_size=9,
                 norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0):
        """
        Construct a one dimensional U-Net
        Parameters:
            input_chan_num (int)  -- the number of channels in input omics data
            output_chan_num (int) -- the number of channels in output omics data
            output_dim (int)      -- the dimension of the output omics data
            filter_num (int)      -- the number of filters in the first convolution layer
            kernel_size (int)     -- the kernel size of convolution layers
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
        """
        super(UNetIn, self).__init__()
        self.input_chan_num = input_chan_num
        self.output_chan_num = output_chan_num
        self.output_dim = output_dim

        # Encoder
        # 1 -> 64 deal with the input data the first double convolution layer
        self.input_conv = DoubleConv1D(input_chan_num, filter_num, kernel_size=kernel_size, norm_layer=norm_layer,
                                       leaky_slope=leaky_slope)
        # 64 -> 128
        self.down_sample_1 = DownSample(filter_num, filter_num * 2, down_ratio=2, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 128 -> 256
        self.down_sample_2 = DownSample(filter_num * 2, filter_num * 4, down_ratio=2, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 256 -> 512
        self.down_sample_3 = DownSample(filter_num * 4, filter_num * 8, down_ratio=2, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 512 -> 1024
        self.down_sample_4 = DownSample(filter_num * 8, filter_num * 16, down_ratio=2, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)

        # Decoder
        # 1024 -> 512
        self.up_sample_1 = UpSampleIn(filter_num * 16, filter_num * 8, up_ratio=4, skip_ratio=2, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 512 -> 256
        self.up_sample_2 = UpSampleIn(filter_num * 8, filter_num * 4, up_ratio=2, skip_ratio=2, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 256 -> 128
        self.up_sample_3 = UpSampleIn(filter_num * 4, filter_num * 2, up_ratio=2, skip_ratio=2, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=0)
        # 128 -> 64
        self.up_sample_4 = UpSampleIn(filter_num * 2, filter_num, up_ratio=2, skip_ratio=2, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 64 -> 1
        self.output_conv = OutputConv(filter_num, output_chan_num)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down_sample_1(x1)
        x3 = self.down_sample_2(x2)
        x4 = self.down_sample_3(x3)
        x5 = self.down_sample_4(x4)
        y1 = self.up_sample_1(x5, x4)
        y2 = self.up_sample_2(y1, x3)
        y3 = self.up_sample_3(y2, x2)
        y4 = self.up_sample_4(y3, x1)
        y5 = self.output_conv(y4)
        # Let the output dim the same as targeting dim
        output = y5[:, :, 0:self.output_dim]
        return output


class UNetSDe(nn.Module):
    """
    Create a simplified 1D U-Net network (decreasing skip connection)
    """
    def __init__(self, input_chan_num, output_chan_num, output_dim, filter_num=64, kernel_size=9,
                 norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0):
        """
        Construct a one dimensional U-Net
        Parameters:
            input_chan_num (int)  -- the number of channels in input omics data
            output_chan_num (int) -- the number of channels in output omics data
            output_dim (int)      -- the dimension of the output omics data
            filter_num (int)      -- the number of filters in the first convolution layer
            kernel_size (int)     -- the kernel size of convolution layers
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
        """
        super(UNetSDe, self).__init__()
        self.input_chan_num = input_chan_num
        self.output_chan_num = output_chan_num
        self.output_dim = output_dim

        # Encoder
        # 1 -> 64 deal with the input data the first double convolution layer
        self.input_conv = DoubleConv1D(input_chan_num, filter_num, kernel_size=kernel_size, norm_layer=norm_layer,
                                       leaky_slope=leaky_slope)
        # 64 -> 256
        self.down_sample_1 = DownSample(filter_num, filter_num * 4, down_ratio=16, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 256 -> 1024
        self.down_sample_2 = DownSample(filter_num * 4, filter_num * 16, down_ratio=16, kernel_size=kernel_size,
                                        norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)

        # Decoder
        # 1024 -> 256
        self.up_sample_1 = UpSampleDe(filter_num * 16, filter_num * 4, up_ratio=4, skip_ratio=4, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 256 -> 64
        self.up_sample_2 = UpSampleDe(filter_num * 4, filter_num, up_ratio=10, skip_ratio=6, kernel_size=kernel_size,
                                    norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
        # 64 -> 1
        self.output_conv = OutputConv(filter_num, output_chan_num)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down_sample_1(x1)
        x3 = self.down_sample_2(x2)
        y1 = self.up_sample_1(x3, x2)
        y2 = self.up_sample_2(y1, x1)
        y3 = self.output_conv(y2)
        # Let the output dim the same as targeting dim
        output = y3[:, :, 0:self.output_dim]
        return output


class DoubleConv1D(nn.Module):
    """
    (Convolution1D => Norm1D => LeakyReLU) * 2
    The omics data dimension keep the same during this process
    """
    def __init__(self, input_chan_num, output_chan_num, kernel_size=9, norm_layer=nn.BatchNorm1d, leaky_slope=0.2):
        """
        Construct a double convolution block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output tensor
            kernel_size (int)     -- the kernel size of the convolution layer
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
        """
        super(DoubleConv1D, self).__init__()

        # Only if the norm method is instance norm we use bias for the corresponding conv layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        self.double_conv_1d = nn.Sequential(
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2, bias=use_bias),
            norm_layer(output_chan_num),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
            nn.Conv1d(output_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2, bias=use_bias),
            norm_layer(output_chan_num),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )

    def forward(self, x):
        return self.double_conv_1d(x)


class DownSample(nn.Module):
    """
    Downsampling with MaxPool then call the DoubleConv1D module
    The output dimension = input dimension // ratio
    """
    def __init__(self, input_chan_num, output_chan_num, down_ratio, kernel_size=9, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2, dropout_p=0):
        """
        Construct a downsampling block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output tensor
            down_ratio (int)      -- the kernel size and stride of the MaxPool1d layer
            kernel_size (int)     -- the kernel size of the DoubleConv1D block
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
        """
        super(DownSample, self).__init__()
        self.maxpool_double_conv = nn.Sequential(
            nn.MaxPool1d(down_ratio),
            nn.Dropout(p=dropout_p),
            DoubleConv1D(input_chan_num, output_chan_num, kernel_size, norm_layer, leaky_slope)
        )

    def forward(self, x):
        return self.maxpool_double_conv(x)


class UpSampleDe(nn.Module):
    """
    Upsampling with ConvTranspose1d then call the DoubleConv1D module (decreasing skip connection)
    The output dimension = input dimension * ratio
    """
    def __init__(self, input_chan_num, output_chan_num, up_ratio, skip_ratio, kernel_size=9, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2, dropout_p=0):
        """
        Construct a upsampling block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor (the tensor from get from the last layer, not the tensor from the skip-connection mechanism)
            output_chan_num (int) -- the number of channels of the output tensor
            up_ratio (int)        -- the kernel size and stride of the ConvTranspose1d layer
            skip_ratio (int)      -- the kernel size the MaxPool1d layer for the skip connection
            kernel_size (int)     -- the kernel size of the DoubleConv1D block
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
        """
        super(UpSampleDe, self).__init__()
        self.up_sample = nn.ConvTranspose1d(input_chan_num, output_chan_num, kernel_size=up_ratio, stride=up_ratio)
        self.skip_connect_pool = nn.MaxPool1d(skip_ratio)
        self.dropout = nn.Dropout(p=dropout_p)
        self.double_conv = DoubleConv1D(output_chan_num * 2, output_chan_num, kernel_size, norm_layer, leaky_slope)

    def forward(self, input1, input2):
        x1 = self.up_sample(input1)
        x2 = self.skip_connect_pool(input2)
        x2_crop = x2[:, :, 0:x1.shape[2]]
        # The skip connection mechanism
        x = torch.cat([x2_crop, x1], dim=1)
        x_dropout = self.dropout(x)
        return self.double_conv(x_dropout)


class UpSampleIn(nn.Module):
    """
    Upsampling with ConvTranspose1d then call the DoubleConv1D module (increasing skip connection)
    The output dimension = input dimension * ratio
    """
    def __init__(self, input_chan_num, output_chan_num, up_ratio, skip_ratio, kernel_size=9, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2, dropout_p=0):
        """
        Construct a upsampling block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor (the tensor from get from the last layer, not the tensor from the skip-connection mechanism)
            output_chan_num (int) -- the number of channels of the output tensor
            up_ratio (int)        -- the kernel size and stride of the ConvTranspose1d layer
            skip_ratio (int)      -- the kernel size and stride of the ConvTranspose1d  layer for the skip connection
            kernel_size (int)     -- the kernel size of the DoubleConv1D block
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
        """
        super(UpSampleIn, self).__init__()
        self.up_sample = nn.ConvTranspose1d(input_chan_num, output_chan_num, kernel_size=up_ratio, stride=up_ratio)
        self.skip_connect_pool = nn.ConvTranspose1d(output_chan_num, output_chan_num, kernel_size=skip_ratio, stride=skip_ratio)
        self.dropout = nn.Dropout(p=dropout_p)
        self.double_conv = DoubleConv1D(output_chan_num * 2, output_chan_num, kernel_size, norm_layer, leaky_slope)

    def forward(self, input1, input2):
        x1 = self.up_sample(input1)
        x2 = self.skip_connect_pool(input2)
        x2_crop = x2[:, :, 0:x1.shape[2]]
        # The skip connection mechanism
        x = torch.cat([x2_crop, x1], dim=1)
        x_dropout = self.dropout(x)
        return self.double_conv(x_dropout)


class OutputConv(nn.Module):
    """
    Output convolution layer
    """
    def __init__(self, input_chan_num, output_chan_num):
        """
        Construct the output convolution layer
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output omics data
        """
        super(OutputConv, self).__init__()
        self.output_conv = nn.Conv1d(input_chan_num, output_chan_num, kernel_size=1)

    def forward(self, x):
        return self.output_conv(x)


class SingleConv1D(nn.Module):
    """
    Convolution1D => Norm1D => LeakyReLU
    The omics data dimension keep the same during this process
    """
    def __init__(self, input_chan_num, output_chan_num, kernel_size=9, norm_layer=nn.BatchNorm1d, leaky_slope=0.2):
        """
        Construct a double convolution block
        Parameters:
            input_chan_num (int)  -- the number of channels of the input tensor
            output_chan_num (int) -- the number of channels of the output tensor
            kernel_size (int)     -- the kernel size of the convolution layer
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
        """
        super(SingleConv1D, self).__init__()

        # Only if the norm method is instance norm we use bias for the corresponding conv layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        self.double_conv_1d = nn.Sequential(
            nn.Conv1d(input_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2, bias=use_bias),
            norm_layer(output_chan_num),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
        )

    def forward(self, x):
        return self.double_conv_1d(x)

# For fully-connected generator
class FcVed(nn.Module):
    """
        Defines a fully-connected variational encoder-decoder model for DNA methylation to gene expression translation
        DNA methylation input not separated by chromosome
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=4096, dim_2B=1024,
                 dim_3_B=512, dim_1A=4096, dim_2A=1024, dim_3_A=512, latent_dim=256):
        """
            Parameters:
                input_dim (int)         -- dimension of B
                output_dim (int)        -- dimension of A
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """
        super(FcVed, self).__init__()

        # ENCODER
        # Layer 1
        self.encode_fc_1 = FCBlock(input_dim, dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.encode_fc_2 = FCBlock(dim_1B, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3_B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3_B, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, activation=False,
                                      normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3_B, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, activation=False,
                                         normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3_A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3_A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3 = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.decode_fc_4 = FCBlock(dim_1A, output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, activation=False,
                                   normalization=False)

    def encode(self, x):
        level_2 = self.encode_fc_1(x)
        level_3 = self.encode_fc_2(level_2)
        level_4 = self.encode_fc_3(level_3)
        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)
        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)
        level_2 = self.decode_fc_2(level_1)
        level_3 = self.decode_fc_3(level_2)
        recon_A = self.decode_fc_4(level_3)
        return recon_A

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class FcVedSep(nn.Module):
    """
        Defines a fully-connected variational encoder-decoder model for DNA methylation to gene expression translation
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=128, dim_2B=1024,
                 dim_3B=512, dim_1A=2048, dim_2A=1024, dim_3A=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                input_dim (list)        -- dimension list of B
                output_dim (int)        -- dimension of A
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """
        super(FcVedSep, self).__init__()

        # ENCODER
        # Layer 1
        self.encode_fc_1_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1_list.append(
                FCBlock(input_dim[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        # Layer 2
        self.encode_fc_2 = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_mean = FCBlock(dim_3B, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, activation=False,
                                      normalization=False)
        self.encode_fc_log_var = FCBlock(dim_3B, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, activation=False,
                                         normalization=False)

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3A, norm_layer=norm_layer, leaky_slope=leaky_slope,
                                   dropout_p=dropout_p, activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3 = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.decode_fc_4 = FCBlock(dim_1A, output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, activation=False,
                                   normalization=False)

    def encode(self, x):
        level_2_list = []
        for i in range(0, 23):
            level_2_list.append(self.encode_fc_1_list[i](x[i]))
        level_2 = torch.cat(level_2_list, 2)
        level_3 = self.encode_fc_2(level_2)
        level_4 = self.encode_fc_3(level_3)
        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)
        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)
        level_2 = self.decode_fc_2(level_1)
        level_3 = self.decode_fc_3(level_2)
        recon_A = self.decode_fc_4(level_3)
        return recon_A

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_A = self.decode(z)
        return z, recon_A, mean, log_var


class FCG(nn.Module):
    """
    Create a fully-connected generator network
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, latent_dim=256):
        """
        Construct a one dimensional U-Net
        Parameters:
            input_dim (int)       -- the dimension of the input omics data
            output_dim (int)      -- the dimension of the output omics data
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            latent_dim (int)        -- the dimensionality of the latent space
        """
        super(FCG, self).__init__()

        dim_1 = 4096
        dim_2 = 1024
        dim_3 = 512

        # dim_1 = 256
        # dim_2 = 128
        # dim_3 = 128

        mul_fc_block = [FCBlock(input_dim, dim_1, norm_layer, leaky_slope, dropout_p),
                        FCBlock(dim_1, dim_2, norm_layer, leaky_slope, dropout_p),
                        FCBlock(dim_2, dim_3, norm_layer, leaky_slope, dropout_p),
                        FCBlock(dim_3, latent_dim, norm_layer, leaky_slope, dropout_p),
                        FCBlock(latent_dim, dim_3, norm_layer, leaky_slope, dropout_p),
                        FCBlock(dim_3, dim_2, norm_layer, leaky_slope, dropout_p),
                        FCBlock(dim_2, dim_1, norm_layer, leaky_slope, dropout_p),
                        nn.Linear(dim_1, output_dim)
                        ]
        self.mul_fc = nn.Sequential(*mul_fc_block)

    def forward(self, x):
        return self.mul_fc(x)


class FCGSep(nn.Module):
    """
        Defines a fully-connected encoder-decoder model for DNA methylation to gene expression translation
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=256, dim_2B=1024,
                 dim_3B=512, dim_1A=2048, dim_2A=1024, dim_3A=512, latent_dim=256):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                input_dim (list)        -- dimension list of B
                output_dim (int)        -- dimension of A
                norm_layer              -- normalization layer
                leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
                dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
                latent_dim (int)        -- the dimensionality of the latent space
        """
        super(FCGSep, self).__init__()

        # ENCODER
        # Layer 1
        self.encode_fc_1_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1_list.append(
                FCBlock(input_dim[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        # Layer 2
        self.encode_fc_2 = FCBlock(dim_1B*23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B, dim_3B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_4 = FCBlock(dim_3B, latent_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(latent_dim, dim_3A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 2
        self.decode_fc_2 = FCBlock(dim_3A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 3
        self.decode_fc_3 = FCBlock(dim_2A, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.decode_fc_4 = FCBlock(dim_1A, output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=False, normalization=False)

    def encode(self, x):
        level_2_list = []
        for i in range(0, 23):
            level_2_list.append(self.encode_fc_1_list[i](x[i]))
        level_2 = torch.cat(level_2_list, 2)
        level_3 = self.encode_fc_2(level_2)
        level_4 = self.encode_fc_3(level_3)
        latent = self.encode_fc_4(level_4)
        return latent

    def decode(self, z):
        level_1 = self.decode_fc_z(z)
        level_2 = self.decode_fc_2(level_1)
        level_3 = self.decode_fc_3(level_2)
        recon_A = self.decode_fc_4(level_3)
        return recon_A

    def forward(self, x):
        latent = self.encode(x)
        recon_A = self.decode(latent)
        return recon_A


class FCGSingle(nn.Module):
    """
        Defines a single hidden layer fully-connected model for DNA methylation to gene expression translation
    """
    def __init__(self, input_dim, output_dim, latent_dim=4000):
        """
            Construct a fully-connected variational autoencoder
            Parameters:
                input_dim (list)        -- dimension list of B
                output_dim (int)        -- dimension of A
                norm_layer              -- normalization layer
                latent_dim (int)        -- the dimensionality of the latent space
        """
        super(FCGSingle, self).__init__()
        layers = [nn.Linear(input_dim, latent_dim),
                  nn.Sigmoid(),
                  nn.Linear(latent_dim, output_dim)
                  ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        recon_A = self.layers(x)
        return recon_A


class LinearRegression(nn.Module):
    """
        Defines a linear regression model for DNA methylation to gene expression translation
    """
    def __init__(self, input_dim, output_dim):
        """
            Construct a linear regression
            Parameters:
                input_dim (list)        -- dimension list of B
                output_dim (int)        -- dimension of A
        """
        super(LinearRegression, self).__init__()
        self.lr = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        recon_A = self.lr(x)
        return recon_A


class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, activation=True, normalization=True, activation_name='leakyrelu'):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        if normalization:
            # FC block doesn't support BatchNorm1d
            if isinstance(norm_layer, functools.partial) and norm_layer.func == nn.BatchNorm1d:
                norm_layer = nn.InstanceNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU
        if activation:
            if activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            elif activation_name.lower() == 'sigmoid':
                self.fc_block.append(nn.Sigmoid())
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc_block(x)
        return y


class TransformerG(nn.Module):
    """
    Create a transformer generator network
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, latent_dim=256,
                 nhead=8):
        """
        Construct a transformer generator
        Parameters:
            input_dim (int)       -- the dimension of the input omics data
            output_dim (int)      -- the dimension of the output omics data
            norm_layer            -- normalization layer
            leaky_slope (float)   -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)     -- probability of an element to be zeroed in a dropout layer
            latent_dim (int)      -- the dimensionality of the latent space
            nhead (int)           -- the number of heads in the transformer encoder layer
        """
        super(TransformerG, self).__init__()

        mul_fc_block = [FCBlock(input_dim, latent_dim, norm_layer, leaky_slope, dropout_p),
                        nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead),
                        nn.Linear(latent_dim, output_dim)
                        ]
        self.mul_fc = nn.Sequential(*mul_fc_block)

    def forward(self, x):
        return self.mul_fc(x)


# DISCRIMINATOR

# For one dimensional multi-layer convolution discriminator
class MulLayerDiscriminator(nn.Module):
    """
    Defines a one dimensional multi-layer convolution discriminator
    """
    def __init__(self, input_1_chan_num, input_2_chan_num, filter_num=64, layer_num=3, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2):
        """
        Construct a one dimensional multi-layer discriminator
        Parameters:
            input_1_chan_num (int)  -- the number of channels in the first input omics data (A)
            input_2_chan_num (int)  -- the number of channels in the second input omics data (B)
            filter_num (int)        -- the number of filters in the first convolution layer
            layer_num (int)         -- the number of convolution layers in the discriminator
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        """
        super(MulLayerDiscriminator, self).__init__()

        self.input_conv = DisInputConv(input_1_chan_num, input_2_chan_num, filter_num, leaky_slope)

        # create a list to store conv blocks
        mul_conv_block = []
        conv_block_filter_num = filter_num * 2
        # the block number of the multi-layer convolution block should not exceed 6
        block_layer_num = min(layer_num, 6)
        for num in range(0, block_layer_num):
            # the filter number should not exceed 1024
            next_filter_num = min(conv_block_filter_num, 1024)
            mul_conv_block += [SingleConv1D(conv_block_filter_num, next_filter_num, 8, norm_layer, leaky_slope)]
            conv_block_filter_num = next_filter_num
        self.mul_conv = nn.Sequential(*mul_conv_block)

        # the output convolution layer of the discriminator
        self.output_conv = nn.Conv1d(conv_block_filter_num, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input1, input2):
        combined_input = self.input_conv(input1, input2)
        x = self.mul_conv(combined_input)
        return self.output_conv(x)


class MulLayerDiscriminatorNew(nn.Module):
    """
    Defines a one dimensional multi-layer convolution discriminator
    """
    def __init__(self, input_1_chan_num, input_2_chan_num, a_dim, output_dim=64, filter_num=64, kernel_size=9, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2, dropout_p=0):
        """
        Construct a one dimensional multi-layer discriminator
        Parameters:
            input_1_chan_num (int)  -- the number of channels in the first input omics data (A)
            input_2_chan_num (int)  -- the number of channels in the second input omics data (B)
            filter_num (int)        -- the number of filters in the first convolution layer
            layer_num (int)         -- the number of convolution layers in the discriminator
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        """
        super(MulLayerDiscriminatorNew, self).__init__()

        self.input_conv = DisInputConvUp(input_1_chan_num, input_2_chan_num, filter_num, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope)

        # create a list to store conv blocks
        self.mul_conv = nn.Sequential(
            # 128 -> 256
            DownSample(filter_num * 2, filter_num * 4, down_ratio=4, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p),
            # 256 -> 512
            DownSample(filter_num * 4, filter_num * 8, down_ratio=4, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p),
            # 512 -> 1024
            DownSample(filter_num * 8, filter_num * 16, down_ratio=4, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p),
            # 1024 -> 512
            SingleConv1D(filter_num * 16, filter_num * 8, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope),
            # 512 -> 256
            SingleConv1D(filter_num * 8, filter_num * 4, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope),
            # 256 -> 128
            SingleConv1D(filter_num * 4, filter_num * 2, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope),
            # 128 -> 64
            SingleConv1D(filter_num * 2, filter_num, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope),
            # 64 -> 1
            OutputConv(filter_num, 1),
            # FC
            FCBlock(a_dim // 4 // 4 // 4, output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, input1, input2):
        combined_input = self.input_conv(input1, input2)
        x = self.mul_conv(combined_input)
        return x


class MulLayerDiscriminatorNewS(nn.Module):
    """
    Defines a one dimensional multi-layer convolution discriminator
    """
    def __init__(self, input_1_chan_num, input_2_chan_num, filter_num=64, kernel_size=9, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2, dropout_p=0):
        """
        Construct a one dimensional multi-layer discriminator
        Parameters:
            input_1_chan_num (int)  -- the number of channels in the first input omics data (A)
            input_2_chan_num (int)  -- the number of channels in the second input omics data (B)
            filter_num (int)        -- the number of filters in the first convolution layer
            layer_num (int)         -- the number of convolution layers in the discriminator
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        """
        super(MulLayerDiscriminatorNewS, self).__init__()

        self.input_conv = DisInputConvUp(input_1_chan_num, input_2_chan_num, filter_num, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope)

        # create a list to store conv blocks
        self.mul_conv = nn.Sequential(
            # 128 -> 256
            DownSample(filter_num * 2, filter_num * 4, down_ratio=4, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p),
            # 256 -> 512
            DownSample(filter_num * 4, filter_num * 8, down_ratio=4, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p),
            # 512 -> 1024
            DownSample(filter_num * 8, filter_num * 16, down_ratio=4, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p),
            # 1024 -> 1
            nn.Conv1d(filter_num * 16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input1, input2):
        combined_input = self.input_conv(input1, input2)
        x = self.mul_conv(combined_input)
        return x


class DisInputConv(nn.Module):
    """
    The input convolution block for the conditional GAN multi-layer discriminator
    The input of this block are the two different omics data
    """
    def __init__(self, input_1_chan_num, input_2_chan_num, output_chan_num, leaky_slope=0.2):
        """
        Construct the input convolution layer
        Parameters:
            input_1_chan_num (int)  -- the number of channels of the first input omics data (A)
            input_2_chan_num (int)  -- the number of channels of the second input omics data (B)
            output_chan_num (int)   -- the number of channels of the output tensor
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        """
        super(DisInputConv, self).__init__()

        self.input_1_conv_layer = nn.Sequential(
            nn.Conv1d(input_1_chan_num, output_chan_num, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )
        self.input_2_conv_layer = nn.Sequential(
            nn.Conv1d(input_2_chan_num, output_chan_num, kernel_size=9, stride=7, padding=1),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )

    def forward(self, input1, input2):
        x1 = self.input_1_conv_layer(input1)
        x2 = self.input_2_conv_layer(input2)
        x2_crop = x2[:, :, 0:x1.shape[2]]
        # concat the two input omics data together
        x = torch.cat([x1, x2_crop], dim=1)
        return x


class DisInputConvUp(nn.Module):
    """
    The input convolution block for the conditional GAN multi-layer discriminator
    The input of this block are the two different omics data
    """
    def __init__(self, input_1_chan_num, input_2_chan_num, output_chan_num, up_ratio=2, kernel_size=9, norm_layer=nn.BatchNorm1d, leaky_slope=0.2):
        """
        Construct the input convolution layer
        Parameters:
            input_1_chan_num (int)  -- the number of channels of the first input omics data (A)
            input_2_chan_num (int)  -- the number of channels of the second input omics data (B)
            output_chan_num (int)   -- the number of channels of the output tensor
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        """
        super(DisInputConvUp, self).__init__()

        # Only if the norm method is instance norm we use bias for the corresponding conv layer
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        self.input_1_conv = DoubleConv1D(input_1_chan_num, output_chan_num, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope)
        self.input_2_conv = DoubleConv1D(input_2_chan_num, output_chan_num, kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope)

        self.input_1_out = nn.Conv1d(output_chan_num, output_chan_num, kernel_size=kernel_size, padding=kernel_size // 2, bias=use_bias)

        self.input_2_out = nn.ConvTranspose1d(output_chan_num, output_chan_num, kernel_size=up_ratio, stride=up_ratio)

    def forward(self, input1, input2):
        x1 = self.input_1_conv(input1)
        x1_out = self.input_1_out(x1)
        x2 = self.input_2_conv(input2)
        x2_out = self.input_2_out(x2)
        x2_crop = x2_out[:, :, 0:x1.shape[2]]
        # concat the two input omics data together
        x = torch.cat([x1_out, x2_crop], dim=1)
        return x


class MulLayerDiscriminatorSep(nn.Module):
    """
    Defines a one dimensional multi-layer convolution discriminator
    """
    def __init__(self, input_1_chan_num, input_2_chan_num, filter_num=64, layer_num=3, norm_layer=nn.BatchNorm1d,
                 leaky_slope=0.2):
        """
        Construct a one dimensional multi-layer discriminator
        Parameters:
            input_1_chan_num (int)  -- the number of channels in the first input omics data (A)
            input_2_chan_num (int)  -- the number of channels in the second input omics data (B)
            filter_num (int)        -- the number of filters in the first convolution layer
            layer_num (int)         -- the number of convolution layers in the discriminator
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        """
        super(MulLayerDiscriminatorSep, self).__init__()

        self.input_conv = DisInputConvSep(input_1_chan_num, input_2_chan_num, filter_num, leaky_slope)

        # create a list to store conv blocks
        mul_conv_block = []
        conv_block_filter_num = filter_num * 2
        # the block number of the multi-layer convolution block should not exceed 6
        block_layer_num = min(layer_num, 6)
        for num in range(0, block_layer_num):
            # the filter number should not exceed 1024
            next_filter_num = min(conv_block_filter_num, 1024)
            mul_conv_block += [SingleConv1D(conv_block_filter_num, next_filter_num, 8, norm_layer, leaky_slope)]
            conv_block_filter_num = next_filter_num
        self.mul_conv = nn.Sequential(*mul_conv_block)

        # the output convolution layer of the discriminator
        self.output_conv = nn.Conv1d(conv_block_filter_num, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, input1, input2):
        combined_input = self.input_conv(input1, input2)
        x = self.mul_conv(combined_input)
        return self.output_conv(x)


class DisInputConvSep(nn.Module):
    """
    The input convolution block for the conditional GAN multi-layer discriminator
    The input of this block are the two different omics data
    """
    def __init__(self, input_1_chan_num, input_2_chan_num, output_chan_num, leaky_slope=0.2):
        """
        Construct the input convolution layer
        Parameters:
            input_1_chan_num (int)  -- the number of channels of the first input omics data (A)
            input_2_chan_num (int)  -- the number of channels of the second input omics data (B)
            output_chan_num (int)   -- the number of channels of the output tensor
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        """
        super(DisInputConvSep, self).__init__()

        self.input_1_conv_layer = nn.Sequential(
            nn.Conv1d(input_1_chan_num, output_chan_num, kernel_size=9, stride=1, padding=4),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )
        self.input_2_conv_layer = nn.Sequential(
            nn.Conv1d(input_2_chan_num, output_chan_num, kernel_size=9, stride=7, padding=1),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
        )

    def forward(self, input1, input2):
        input2_list = []
        for i in range(0, 23):
            input2_list.append(input2[i])
        input2_cat = torch.cat(input2_list, 2)

        x1 = self.input_1_conv_layer(input1)
        x2 = self.input_2_conv_layer(input2_cat)
        x2_crop = x2[:, :, 0:x1.shape[2]]
        # concat the two input omics data together
        x = torch.cat([x1, x2_crop], dim=1)
        return x


# For fully-connected discriminator
class FCD(nn.Module):
    """
    Defines a fully-connected discriminator
    """
    def __init__(self, input_1_dim, input_2_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, output_dim=16):
        """
        Construct a fully-connected discriminator
        Parameters:
            input_1_dim (int)       -- the dimension of the first input omics data (A)
            input_2_dim (int)       -- the dimension of the second input omics data (B)
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            output_dim (int)        -- the output dimension of the discriminator
        """
        super(FCD, self).__init__()

        dim_1 = 128
        dim_2 = 64
        dim_3 = 32

        mul_fc_block = [FCBlock(input_1_dim+input_2_dim, dim_1, norm_layer, leaky_slope),
                        FCBlock(dim_1, dim_2, norm_layer, leaky_slope),
                        FCBlock(dim_2, dim_3, norm_layer, leaky_slope),
                        nn.Linear(dim_3, output_dim)]
        self.mul_fc = nn.Sequential(*mul_fc_block)

    def forward(self, input1, input2):
        combined_input = torch.cat([input1, input2], dim=2)
        output = self.mul_fc(combined_input)
        return output


class FCDSingle(nn.Module):
    """
    Defines a single hidden layer fully-connected discriminator
    """
    def __init__(self, input_1_dim, input_2_dim, hidden_dim=64, output_dim=16):
        """
        Construct a fully-connected discriminator
        Parameters:
            input_1_dim (int)       -- the dimension of the first input omics data (A)
            input_2_dim (int)       -- the dimension of the second input omics data (B)
            output_dim (int)        -- the output dimension of the discriminator
        """
        super(FCDSingle, self).__init__()

        layers = [nn.Linear(input_1_dim+input_2_dim, hidden_dim),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                  nn.Linear(hidden_dim, output_dim)
                  ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input1, input2):
        combined_input = torch.cat([input1, input2], dim=2)
        output = self.layers(combined_input)
        return output


class FCDSep(nn.Module):
    """
        Defines a fully-connected discriminator
        DNA methylation input separated by chromosome
    """
    def __init__(self, input_1_dim, input_2_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout_p=0, dim_1B=16,
                 dim_2B=64, dim_1A=128, dim_2A=64, dim_3=32, output_dim=16):
        """
        Construct a fully-connected discriminator
        Parameters:
            input_1_dim (int)       -- the dimension of the first input omics data (A)
            input_2_dim (int)       -- the dimension of the second input omics data (B)
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            output_dim (int)        -- the output dimension of the discriminator
        """
        super(FCDSep, self).__init__()

        A_dim = input_1_dim
        B_dim_list = input_2_dim

        # ENCODER
        # Layer 1
        self.encode_fc_1B_list = nn.ModuleList()
        for i in range(0, 23):
            self.encode_fc_1B_list.append(
                FCBlock(B_dim_list[i], dim_1B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                        activation=True))
        self.encode_fc_1A = FCBlock(A_dim, dim_1A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 2
        self.encode_fc_2B = FCBlock(dim_1B * 23, dim_2B, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        self.encode_fc_2A = FCBlock(dim_1A, dim_2A, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                    activation=True)
        # Layer 3
        self.encode_fc_3 = FCBlock(dim_2B + dim_2A, dim_3, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=True)
        # Layer 4
        self.encode_fc_4 = FCBlock(dim_3, output_dim, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p,
                                   activation=False)

    def forward(self, input1, input2):
        level_2_B_list = []
        for i in range(0, 23):
            level_2_B_list.append(self.encode_fc_1B_list[i](input2[i]))
        level_2_B = torch.cat(level_2_B_list, 2)
        level_2_A = self.encode_fc_1A(input1)
        level_3_B = self.encode_fc_2B(level_2_B)
        level_3_A = self.encode_fc_2A(level_2_A)
        level_3 = torch.cat((level_3_B, level_3_A), 2)
        level_4 = self.encode_fc_3(level_3)
        output = self.encode_fc_4(level_4)
        return output


class TransformerD(nn.Module):
    """
    Create a transformer discriminator network
    """
    def __init__(self, input_1_dim, input_2_dim, hidden_dim=64, output_dim=16, nhead=8):
        """
        Construct a transformer discriminator
        Parameters:
            input_1_dim (int)       -- the dimension of the first input omics data (A)
            input_2_dim (int)       -- the dimension of the second input omics data (B)
            output_dim (int)        -- the output dimension of the discriminator
            nhead (int)             -- the number of heads in the transformer encoder layer
        """
        super(TransformerD, self).__init__()

        layers = [nn.Linear(input_1_dim+input_2_dim, hidden_dim),
                  nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead),
                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                  nn.Linear(hidden_dim, output_dim)
                  ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input1, input2):
        combined_input = torch.cat([input1, input2], dim=2)
        output = self.layers(combined_input)
        return output


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
    """
    # define the initialization function
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm1d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network:
    1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights
    Parameters:
        net (nn.Module)    -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # multi-GPUs
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_norm_layer(norm_type='batch'):
    """
    Return a normalization layer
    Parameters:
        norm_type (str) -- the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization method [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_chan_num, output_chan_num, netG, A_dim, B_dim, gen_filter_num=64, kernel_size=9, norm_type='batch',
             init_type='normal', init_gain=0.02, gpu_ids=[], leaky_slope=0.2, dropout_p=0, latent_dim=256):
    """
    Create a generator

    Parameters:
        input_chan_num (int)    -- the number of channels in input omics data
        output_chan_num (int)   -- the number of channels in output omics data
        netG (str)              -- the name of the generator architecture, default: unet_1d
        A_dim (int)             -- the dimension of omics type A
        B_dim (int)             -- the dimension of omics type B
        gen_filter_num (int)    -- the number of filters in the first convolution layer in the generator
        kernel_size (int)       -- the kernel size of convolution layers
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        init_type (str)         -- the name of our initialization method
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)      -- which GPUs the network runs on: e.g., 0,1
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- latent dimension for the encoder-decoder model

    Returns a generator

    The default implementation of the generator is the one dimensional U-Net architecture.

    This architecture is modified from the original architecture mentioned in the U-Net paper: https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28

    The generator has been initialized by <init_net>.
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    if netG == 'unet_de':
        net = UNetDe(input_chan_num, output_chan_num, output_dim=A_dim, filter_num=gen_filter_num,
                     kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
    elif netG == 'unet_in':
        net = UNetIn(input_chan_num, output_chan_num, output_dim=A_dim, filter_num=gen_filter_num,
                     kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
    elif netG == 'unet_s_de':
        net = UNetSDe(input_chan_num, output_chan_num, output_dim=A_dim, filter_num=gen_filter_num,
                      kernel_size=kernel_size, norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
    elif netG == 'fcg':
        net = FCG(B_dim, A_dim, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    elif netG == 'fcg_sep':
        net = FCGSep(B_dim, A_dim, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    elif netG == 'fcg_single':
        net = FCGSingle(B_dim, A_dim)
    elif netG == 'fc_ved':
        net = FcVed(B_dim, A_dim, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    elif netG == 'fc_ved_sep':
        net = FcVedSep(B_dim, A_dim, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    elif netG == 'linear_regression':
        net = LinearRegression(B_dim, A_dim)
    elif netG == 'transformer_g':
        net = TransformerG(B_dim, A_dim, norm_layer, leaky_slope, dropout_p, latent_dim=latent_dim)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_1_chan_num, input_2_chan_num, dis_filter_num, netD, A_dim, B_dim, layer_num_D=3, norm_type='batch',
             init_type='normal', init_gain=0.02, gpu_ids=[], leaky_slope=0.2, dropout_p=0):
    """
    Create a discriminator

    Parameters:
        input_1_chan_num (int)    -- the number of channels in the first input omics data
        input_2_chan_num (int)    -- the number of channels in the second input omics data
        dis_filter_num (int)      -- the number of filters in the first convolution layer in the discriminator
        netD (str)                -- the name of the discriminator architecture, default: patch_gan_1d
        A_dim (int)               -- the dimension of omics type A
        B_dim (int)               -- the dimension of omics type B
        layer_num_D (int)         -- the number of convolution layers in the discriminator
        norm_type (str)           -- the type of normalization layers used in the network
        init_type (str)           -- the name of the initialization method
        init_gain (float)         -- scaling factor for normal, xavier and orthogonal initialization methods
        gpu_ids (int list)        -- which GPUs the network runs on: e.g., 0,1
        leaky_slope (float)       -- the negative slope of the Leaky ReLU activation function
        output_dim (int)          -- the output dimension of the discriminator

    Returns a discriminator

    The discriminator has been initialized by <init_net>.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm_type)

    if netD == 'multi_conv':  # default one dimensional multi-layer convolution discriminator
        net = MulLayerDiscriminator(input_1_chan_num, input_2_chan_num, filter_num=dis_filter_num,
                                      layer_num=layer_num_D, norm_layer=norm_layer, leaky_slope=leaky_slope)
    elif netD == 'multi_conv_sep':  # default one dimensional multi-layer convolution discriminator
        net = MulLayerDiscriminatorSep(input_1_chan_num, input_2_chan_num, filter_num=dis_filter_num,
                                         layer_num=layer_num_D, norm_layer=norm_layer, leaky_slope=leaky_slope)
    elif netD == 'multi_conv_new':  # default one dimensional multi-layer convolution discriminator
        net = MulLayerDiscriminatorNew(input_1_chan_num, input_2_chan_num, a_dim=A_dim, filter_num=dis_filter_num,
                                         norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
    elif netD == 'multi_conv_new_s':  # default one dimensional multi-layer convolution discriminator
        net = MulLayerDiscriminatorNewS(input_1_chan_num, input_2_chan_num, filter_num=dis_filter_num,
                                       norm_layer=norm_layer, leaky_slope=leaky_slope, dropout_p=dropout_p)
    elif netD == 'fcd':
        net = FCD(A_dim, B_dim, norm_layer, leaky_slope)
    elif netD == 'fcd_sep':
        net = FCDSep(A_dim, B_dim, norm_layer, leaky_slope)
    elif netD == 'fcd_single':
        net = FCDSingle(A_dim, B_dim)
    elif netD == 'transformer_d':
        net = TransformerD(A_dim, B_dim)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


def get_scheduler(optimizer, param):
    """
    Return a learning rate scheduler

    Parameters:
        optimizer (opt class)     -- the optimizer of the network
        param (params class)      -- param.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <param.niter> epochs and linearly decay the rate to zero
    over the next <param.niter_decay> epochs.

    """
    if param.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_lambda = 1.0 - max(0, epoch + param.epoch_count - param.epoch_num) / float(param.epoch_num_decay + 1)
            return lr_lambda
        # lr_scheduler is imported from torch.optim
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif param.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=param.decay_step_size, gamma=0.1)
    elif param.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif param.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=param.epoch_num, eta_min=0)
    else:
        return NotImplementedError('Learning rate policy [%s] is not found', param.lr_policy)
    return scheduler
