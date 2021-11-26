import os
import torch
from abc import ABC, abstractmethod
from collections import OrderedDict
from . import networks
import numpy as np
import sklearn.metrics as metrics


class BasicModel(ABC):
    """
    This class is an abstract base class for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                          Initialize the class, first call BaseModel.__init__(self, param)
        -- <modify_commandline_parameters>:     Add model-specific parameters, and rewrite default values for existing parameters
        -- <set_input>:                         Unpack input data from the output dictionary of the dataloader
        -- <forward>:                           Get the fake omics data
        -- <update>:                            Calculate losses, gradients and update network parameters
    """

    def __init__(self, param):
        """
        Initialize the BaseModel class
        """
        self.param = param
        self.gpu_ids = param.gpu_ids
        self.isTrain = param.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(param.checkpoints_dir, param.experiment_name)  # save all the checkpoints to save_dir, and this is where to load the models
        self.load_net_dir = os.path.join(param.checkpoints_dir, param.experiment_to_load)  # load pretrained networks from certain experiment folder

        # Improve the performance if the dimensionality and shape of the input data keep the same
        torch.backends.cudnn.benchmark = True

        self.loss_names = []
        self.model_names = []
        self.optimizers = []
        self.schedulers = []

        self.real_A_tensor = []
        self.real_B_tensor = []
        self.fake_A_tensor = []

        self.data_index = []    # The indexes of input data

        # specify the metrics you want to print out.
        self.metric_names = ['MSE', 'RMSE', 'MAE', 'MEDAE', 'R2']
        self.metric_MSE = []
        self.metric_RMSE = []
        self.metric_MAE = []
        self.metric_MEDAE = []
        self.metric_R2 = []

        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_parameters(parser, is_train):
        """
        Add model-specific parameters, and rewrite default values for existing parameters.

        Parameters:
            parser              -- original parameter parser
            is_train (bool)     -- whether it is currently training phase or test phase. Use this flag to add or change training-specific or test-specific parameters.

        Returns:
            The modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input_dict):
        """
        Unpack input data from the output dictionary of the dataloader

        Parameters:
            input_dict (dict): include the data tensor and its path. This dict looks like this {'A_tensor': A_tensor, 'B_tensor': B_tensor, 'A_path': A_path, 'B_path': B_path}
        """
        pass

    @abstractmethod
    def forward(self):
        """
        Run forward pass
        """
        pass

    @abstractmethod
    def update(self):
        """
        Calculate losses, gradients and update network weights; called in every training iteration
        """
        pass

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requies_grad=Fasle to avoid back propagation
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for net_param in net.parameters():
                    net_param.requires_grad = requires_grad

    def setup(self, param):
        """
        Load and print networks, create schedulers
        """
        if self.isTrain:
            self.print_networks(param)
            # For every optimizer we have a scheduler
            self.schedulers = [networks.get_scheduler(optimizer, param) for optimizer in self.optimizers]

        # Loading the networks
        if not self.isTrain or param.continue_train:
            self.load_networks(param.epoch_to_load)

    def save_networks(self, epoch):
        """
        Save all the networks to the disk.

        Parameters:
            epoch (str) -- current epoch
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '{:s}_net_{:s}.pth'.format(epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                # Use the str to get the attribute aka the network (self.netG / self.netD)
                net = getattr(self, 'net' + name)
                # If we use multi GPUs and apply the data parallel
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """
        Load networks at specified epoch from the disk.

        Parameters:
            epoch (str) -- Which epoch to load
        """
        for model_name in self.model_names:
            if isinstance(model_name, str):
                load_filename = '{:s}_net_{:s}.pth'.format(epoch, model_name)
                load_path = os.path.join(self.load_net_dir, load_filename)
                # Use the str to get the attribute aka the network (self.netG / self.netD)
                net = getattr(self, 'net' + model_name)
                # If we use multi GPUs and apply the data parallel
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('Loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                net.load_state_dict(state_dict)

    def print_networks(self, param):
        """
        Print the total number of parameters in the network and network architecture if detail is true
        Save the networks information to the disk
        """
        message = '\n----------------------Networks Information----------------------'
        for model_name in self.model_names:
            if isinstance(model_name, str):
                net = getattr(self, 'net' + model_name)
                num_params = 0
                for parameter in net.parameters():
                    num_params += parameter.numel()
                if param.detail:
                    message += '\n' + str(net)
                message += '\n[Network {:s}] Total number of parameters : {:.3f} M'.format(model_name, num_params / 1e6)
        message += '\n----------------------------------------------------------------\n'

        # Save the networks information to the disk
        net_info_filename = os.path.join(param.checkpoints_dir, param.experiment_name, 'net_info.txt')
        with open(net_info_filename, 'w') as log_file:
            log_file.write(message)

        print(message)

    def update_learning_rate(self):
        """
        Update learning rates for all the networks
        Called at the end of each epoch
        """

        lr = self.optimizers[0].param_groups[0]['lr']
        # print('Learning rate for this epoch: %.7f' % lr)

        for scheduler in self.schedulers:
            if self.param.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def set_train(self):
        """
        Set train mode for networks
        """
        for model_name in self.model_names:
            if isinstance(model_name, str):
                # Use the str to get the attribute aka the network (self.netG / self.netD)
                net = getattr(self, 'net' + model_name)
                net.train()

    def set_eval(self):
        """
        Set eval mode for networks
        """
        for model_name in self.model_names:
            if isinstance(model_name, str):
                # Use the str to get the attribute aka the network (self.netG / self.netD)
                net = getattr(self, 'net' + model_name)
                net.eval()

    def test(self):
        """
        Forward in testing to get the fake omics data
        """
        with torch.no_grad():
            self.forward()

    def init_losses_dict(self):
        """
        initialize a losses dictionary
        """
        losses_acc = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                losses_acc[name] = []
        return losses_acc

    def update_losses_dict(self, losses_dict):
        """
        losses_dict (OrderedDict)  -- the losses dictionary to be updated
        """
        for name in self.loss_names:
            if isinstance(name, str):
                losses_dict[name].append(float(getattr(self, 'loss_' + name)))

    def init_metrics_dict(self):
        """
        initialize a metrics dictionary
        """
        metrics_acc = OrderedDict()
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_acc[name] = []
        return metrics_acc

    def update_metrics_dict(self, metrics_dict):
        """
        Return performance metrics during testing
        metrics_dict (OrderedDict)  -- the metrics dictionary to be updated
        """
        for name in self.metric_names:
            if isinstance(name, str):
                metrics_dict[name].extend(getattr(self, 'metric_' + name))

    def calculate_current_metrics(self):
        """
        Calculate current metrics
        """
        with torch.no_grad():
            if self.gpu_ids:
                fake_A_array = self.fake_A_tensor.cpu().numpy()
                real_A_array = self.real_A_tensor.cpu().numpy()
            else:
                fake_A_array = self.fake_A_tensor.detach().numpy()
                real_A_array = self.real_A_tensor.detach().numpy()

            self.metric_MSE = []
            self.metric_RMSE = []
            self.metric_MAE = []
            self.metric_MEDAE = []
            self.metric_R2 = []

            for i in range(fake_A_array.shape[0]):
                fake_A_single = fake_A_array[i].ravel()
                real_A_single = real_A_array[i].ravel()

                self.metric_MSE.append(metrics.mean_squared_error(real_A_single, fake_A_single))
                self.metric_RMSE.append(metrics.mean_squared_error(real_A_single, fake_A_single, squared=False))
                self.metric_MAE.append(metrics.mean_absolute_error(real_A_single, fake_A_single))
                self.metric_MEDAE.append(metrics.median_absolute_error(real_A_single, fake_A_single))
                self.metric_R2.append(metrics.r2_score(real_A_single, fake_A_single))

    def init_fake_dict(self):
        """
        initialize and return an empty fake array and an empty index array
        """
        fake_dict = OrderedDict()
        fake_dict['index'] = np.zeros(shape=[0])
        fake_dict['fake'] = np.zeros(shape=[0, self.param.A_dim])
        return fake_dict

    def update_fake_dict(self, fake_dict):
        """
        update the fake array that stores the predicted omics data
        fake_dict (OrderedDict)  -- the fake array that stores the predicted omics data and the index array
        """
        with torch.no_grad():
            if self.param.add_channel:
                current_fake_array = np.squeeze(self.fake_A_tensor.cpu().numpy(), axis=1)
            else:
                current_fake_array = self.fake_A_tensor.cpu().numpy()
            fake_dict['fake'] = np.concatenate((fake_dict['fake'], current_fake_array))
            current_index_array = self.data_index.cpu().numpy()
            fake_dict['index'] = np.concatenate((fake_dict['index'], current_index_array))
            return fake_dict
