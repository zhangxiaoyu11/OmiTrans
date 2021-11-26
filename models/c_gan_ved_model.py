import torch
from .basic_model import BasicModel
from . import networks
from . import losses
import torch.nn as nn


class CGanVedModel(BasicModel):
    """
    This class implements the conditional GAN VED model, for learning a mapping from input omics data type to output omics
    data type given paired data.
    """
    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        # changing the default values of parameters to match the conditional GAN model
        parser.set_defaults(netG='fc_ved_sep', netD='fcd_sep')
        parser.add_argument('--latent_dim', type=int, default=256,
                            help='the dimensionality of the latent space')
        if is_train:
            parser.add_argument('--lambda_dist', type=float, default=100.0,
                                help='weight for the dist loss')
            parser.add_argument('--lambda_kl', type=float, default=1.0,
                                help='weight for the kl loss')
        return parser

    def __init__(self, param):
        """
        Initialize the conditional GAN VED class.
        """
        BasicModel.__init__(self, param)
        # Declare mean and var for the kl
        self.mean = []
        self.log_var = []
        # specify the training losses you want to print out.
        self.loss_names = ['D_GAN', 'G_GAN', 'G_dist', 'G_kl']
        # specify the models you want to save to the disk.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test phase, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(param.input_chan_num, param.output_chan_num, param.netG, param.A_dim, param.B_dim,
                                      param.gen_filter_num, param.conv_k_size, param.norm_type, param.init_type,
                                      param.init_gain, self.gpu_ids, param.leaky_slope, param.dropout_p, param.latent_dim)

        # define a discriminator if it is the training phase, if it's the testing phase the discriminator is not necessary
        if self.isTrain:
            self.netD = networks.define_D(param.input_chan_num, param.output_chan_num, param.dis_filter_num, param.netD,
                                          param.A_dim, param.B_dim, param.layer_num_D, param.norm_type, param.init_type,
                                          param.init_gain, self.gpu_ids, param.leaky_slope, param.dropout_p)

            # define loss functions: G = L_GAN + λ1 L_dist + λ2 KL
            # The GAN part of the loss function, this return a loss function nn.Module not a value
            # self.device was defined in BaseModel
            self.lossFuncGAN = losses.GANLossObj(param.GAN_mode).to(self.device)
            self.lossFuncDist = losses.get_dist_loss(param.dist_loss)

            # Set optimizer for both generator and discriminator
            # generator and discriminator actually can set to different initial learning rate
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=param.lr_G, betas=(param.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=param.lr_D, betas=(param.beta1, 0.999))
            # optimizer list was already defined in BaseModel
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.loss_D_fake = []
            self.loss_D_real = []
            self.loss_D_GAN = []
            self.loss_G_GAN = []
            self.loss_G_dist = []
            self.loss_G_kl = []
            self.loss_G = []

        if self.param.zo_norm:
            self.sigmoid = nn.Sigmoid()

    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader

        Parameters:
            input_dict (dict): include the data tensor and its index.
        """
        self.real_A_tensor = input_dict['A_tensor'].to(self.device)
        if self.param.ch_separate:
            self.real_B_tensor = []
            for ch in range(0, 23):
                self.real_B_tensor.append(input_dict['B_tensor'][ch].to(self.device))
        else:
            self.real_B_tensor = input_dict['B_tensor'].to(self.device)
        self.data_index = input_dict['index']

    def forward(self):
        # Default B -> A
        _, self.fake_A_tensor, self.mean, self.log_var = self.netG(self.real_B_tensor)  # A' = G(B)
        if self.param.zo_norm:
            self.fake_A_tensor = self.sigmoid(self.fake_A_tensor)
            self.fake_A_tensor = (self.param.target_max - self.param.target_min) * self.fake_A_tensor + self.param.target_min
        # Calculate metrics of the fake tensor
        self.calculate_current_metrics()

    def backward_G(self):
        """Calculate GAN and dist loss for the generator"""
        # G(B) should fake the discriminator to treat it as real omics data
        # The different part compared with the backward_D is that we don't need the detach fake tensor here
        pred_fake = self.netD(self.fake_A_tensor, self.real_B_tensor)   # The prediction vector get from the discriminator for the fake omics data
        self.loss_G_GAN = self.lossFuncGAN(pred_fake, True)     # The boolean variable will be extend to a vector as the same size of pred_fake

        # G(B) should be as close as A, we use the distance loss
        if self.param.zo_norm and self.param.dist_loss == 'BCE':
            self.loss_G_dist = self.lossFuncDist(
                (self.fake_A_tensor - self.param.target_min) / (self.param.target_max - self.param.target_min),
                (self.real_A_tensor - self.param.target_min) / (self.param.target_max - self.param.target_min))
        else:
            self.loss_G_dist = self.lossFuncDist(self.fake_A_tensor, self.real_A_tensor)

        # Add the kl constrain
        self.loss_G_kl = losses.kl_loss(self.mean, self.log_var)

        # Combine the loss and calculate gradients
        # G = L_GAN + λ1 L_dist + λ2 KL
        # The parameter lambda_dist was introduced in this class
        self.loss_G = self.loss_G_GAN + self.param.lambda_dist * self.loss_G_dist + self.param.lambda_kl * self.loss_G_kl
        self.loss_G.backward()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake
        # Stop backprop to the generator by detaching fake_A
        # Conditional GAN was applied so both the input and output of generator were fed to discriminator
        pred_fake = self.netD(self.fake_A_tensor.detach(), self.real_B_tensor)  # the prediction vector get from the discriminator for the fake omics data
        self.loss_D_fake = self.lossFuncGAN(pred_fake, False)   # The boolean variable will be extend to a vector as the same size of pred_fake

        # Real
        pred_real = self.netD(self.real_A_tensor, self.real_B_tensor)
        self.loss_D_real = self.lossFuncGAN(pred_real, True)

        # Combine the loss and calculate gradients
        self.loss_D_GAN = (self.loss_D_fake + self.loss_D_real) / 2
        self.loss_D_GAN.backward()

    def update(self):
        self.forward()                              # Get the fake omics data: G(B)

        # Update parameters of the discriminator
        # the method <set_requires_grad> is defined in BaseModel
        self.set_requires_grad(self.netD, True)     # Enable backprop for D
        self.optimizer_D.zero_grad()                # Set D's gradients to zero
        self.backward_D()                           # Calculate gradients for D
        self.optimizer_D.step()                     # Update D's weights

        # Update parameters of the generator
        self.set_requires_grad(self.netD, False)    # Stop backprop for D when optimizing G
        self.optimizer_G.zero_grad()                # Set G's gradients to zero
        self.backward_G()                           # Calculate gradients for G
        self.optimizer_G.step()                     # Update G's weights

