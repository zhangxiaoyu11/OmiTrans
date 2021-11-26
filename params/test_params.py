from .basic_params import BasicParams


class TestParams(BasicParams):
    """
    This class is a son class of BasicParams.
    This class includes parameters for testing and parameters inherited from the father class.
    """
    def initialize(self, parser):
        parser = BasicParams.initialize(self, parser)

        # Testing parameters
        parser.add_argument('--start_epoch', type=int, default=1,
                            help='start epoch number for testing')
        parser.add_argument('--end_epoch', type=int, default=1,
                            help='end epoch number for testing')

        # Logging and visualization
        parser.add_argument('--print_freq', type=int, default=10,
                            help='frequency of showing testing results on console')
        parser.add_argument('--save_fake', action='store_true',
                            help='save the fake omics data to disc')

        self.isTrain = False
        self.isTest = True
        return parser
