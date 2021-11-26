from .basic_params import BasicParams


class TrainParams(BasicParams):
    """
    This class is a son class of BasicParams.
    This class includes parameters for training and parameters inherited from the father class.
    """
    def initialize(self, parser):
        parser = BasicParams.initialize(self, parser)

        # Training parameters
        parser.add_argument('--GAN_mode', type=str, default='vanilla',
                            help='The type of GAN objective. Vanilla GAN loss is the cross-entropy objective used in the original GAN paper. [vanilla | lsgan]')
        parser.add_argument('--lr_G', type=float, default=0.0002,
                            help='initial learning rate for the generator')
        parser.add_argument('--lr_D', type=float, default=0.0002,
                            help='initial learning rate for the discriminator')
        parser.add_argument('--beta1', type=float, default=0.5,
                            help='momentum term of adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='The learning rate policy for the scheduler. [linear | step | plateau | cosine]')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, default start from 1')
        parser.add_argument('--epoch_num', type=int, default=100,
                            help='Number of epoch using starting learning rate')
        parser.add_argument('--epoch_num_decay', type=int, default=0,
                            help='Number of epoch to linearly decay learning rate to zero')
        parser.add_argument('--decay_step_size', type=int, default=50,
                            help='The original learning rate multiply by a gamma every decay_step_size epoch')

        # Network saving and loading parameters
        parser.add_argument('--continue_train', action='store_true',
                            help='load the latest model and continue training ')
        parser.add_argument('--save_model', action='store_true',
                            help='save the model during training')
        parser.add_argument('--save_epoch_freq', type=int, default=-1,
                            help='frequency of saving checkpoints at the end of epochs, -1 means only save the last epoch')

        # Logging and visualization
        parser.add_argument('--print_freq', type=int, default=1,
                            help='frequency of showing training results on console')

        self.isTrain = True
        self.isTest = False
        return parser
