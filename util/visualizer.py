import os
import time
import numpy as np
import pandas as pd
from util import util
from datasets import save_file
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """
    This class print/save logging information
    """

    def __init__(self, param):
        """
        Initialize the Visualizer class
        """
        self.param = param
        tb_dir = os.path.join(param.checkpoints_dir, param.experiment_name, 'tb_log')
        util.mkdir(tb_dir)

        if param.isTrain:
            # Create a logging file to store training losses
            self.train_log_filename = os.path.join(param.checkpoints_dir, param.experiment_name, 'train_log.txt')
            with open(self.train_log_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Training Log ({:s}) -----------------------\n'.format(now))

            self.train_summary_filename = os.path.join(param.checkpoints_dir, param.experiment_name, 'train_summary.txt')
            with open(self.train_summary_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Training Summary ({:s}) -----------------------\n'.format(now))

            # Create log folder for TensorBoard
            tb_train_dir = os.path.join(param.checkpoints_dir, param.experiment_name, 'tb_log', 'train')
            util.mkdir(tb_train_dir)
            util.clear_dir(tb_train_dir)

            # Create TensorBoard writer
            self.train_writer = SummaryWriter(log_dir=tb_train_dir)

        if param.isTest:
            # Create a logging file to store testing metrics
            self.test_log_filename = os.path.join(param.checkpoints_dir, param.experiment_name, 'test_log.txt')
            with open(self.test_log_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Testing Log ({:s}) -----------------------\n'.format(now))

            self.test_summary_filename = os.path.join(param.checkpoints_dir, param.experiment_name, 'test_summary.txt')
            with open(self.test_summary_filename, 'a') as log_file:
                now = time.strftime('%c')
                log_file.write('----------------------- Testing Summary ({:s}) -----------------------\n'.format(now))

            # Create log folder for TensorBoard
            tb_test_dir = os.path.join(param.checkpoints_dir, param.experiment_name, 'tb_log', 'test')
            util.mkdir(tb_test_dir)
            util.clear_dir(tb_test_dir)

            # Create TensorBoard writer
            self.test_writer = SummaryWriter(log_dir=tb_test_dir)

    def print_train_log(self, epoch, iteration, losses, metrics, load_time, comp_time, batch_size, dataset_size):
        """
        print train log on console and save the message to the disk

        Parameters:
            epoch (int)             -- current epoch
            iteration (int)         -- current training iteration during this epoch
            losses (OrderedDict)    -- training losses stored in the ordered dict
            metrics (OrderedDict)   -- metrics stored in the ordered dict
            load_time (float)       -- data loading time per data point (normalized by batch_size)
            comp_time (float)       -- computational time per data point (normalized by batch_size)
            batch_size (int)        -- batch size of training
            dataset_size (int)      -- size of the training dataset
        """
        data_point_covered = min((iteration + 1) * batch_size, dataset_size)
        message = '[TRAIN] [Epoch: {:3d}   Iter: {:4d}   Load_t: {:.3f}   Comp_t: {:.3f}]   '.format(epoch, data_point_covered, load_time, comp_time)
        for name, loss in losses.items():
            message += '{:s}: {:.4f}   '.format(name, loss[-1])
        for name, metric in metrics.items():
            message += '{:s}: {:.4f}   '.format(name, metric[-1])

        print(message)  # print the message

        with open(self.train_log_filename, 'a') as log_file:
            log_file.write(message + '\n')  # save the message

    def print_train_summary(self, epoch, losses, metrics, train_time):
        """
        print the summary of this training epoch

        Parameters:
            epoch (int)                             -- epoch number of this training model
            losses (OrderedDict)                    -- the losses dictionary
            metrics (OrderedDict)                   -- the metrics dictionary
            train_time (float)                      -- time used for training this epoch
        """
        write_message = '{:s}\t'.format(str(epoch))
        print_message = '[TRAIN] [Epoch: {:4d}]      '.format(int(epoch))

        for name, loss in losses.items():
            write_message += '{:.6f}\t'.format(np.mean(loss))
            print_message += name + ': {:.6f}      '.format(np.mean(loss))
            self.train_writer.add_scalar(name, np.mean(loss), epoch)
        for name, metric in metrics.items():
            write_message += '{:.6f}\t'.format(np.mean(metric))
            print_message += name + ': {:.6f}      '.format(np.mean(metric))
            self.train_writer.add_scalar(name, np.mean(metric), epoch)

        with open(self.train_summary_filename, 'a') as log_file:
            log_file.write(write_message + '\n')

        train_time_msg = 'Training time used: {:.3f}s'.format(train_time)
        print_message += '\n' + train_time_msg
        print(print_message)
        with open(self.train_log_filename, 'a') as log_file:
            log_file.write(train_time_msg + '\n')

    def print_test_log(self, epoch, iteration, metrics, batch_size, dataset_size):
        """
        print performance metrics of this iteration on console and save the message to the disk

        Parameters:
            epoch (int)             -- epoch number of this testing model
            iteration (int)         -- current testing iteration during this epoch
            metrics (OrderedDict)   -- testing metrics stored in the dictionary
            batch_size (int)        -- batch size of testing
            dataset_size (int)      -- size of the testing dataset
        """
        data_point_covered = min((iteration + 1) * batch_size, dataset_size)
        message = '[TEST] [Epoch: {:3d}   Iter: {:4d}]   '.format(int(epoch), data_point_covered)
        for name, metric in metrics.items():
            message += '{:s}: {:.4f}   '.format(name, metric[-1])

        print(message)

        with open(self.test_log_filename, 'a') as log_file:
            log_file.write(message + '\n')

    def print_test_summary(self, epoch, metrics, test_time):
        """
        print the summary of this testing epoch

        Parameters:
            epoch (int)                             -- epoch number of this testing model
            metrics (OrderedDict)                   -- the metrics dictionary
            test_time (float)                       -- time used for testing this epoch
        """
        write_message = '{:s}\t'.format(str(epoch))
        print_message = '[TEST] [Epoch: {:4d}]      '.format(int(epoch))

        for name, metric in metrics.items():
            write_message += '{:.6f}\t'.format(np.mean(metric))
            print_message += name + ': {:.6f}      '.format(np.mean(metric))
            self.test_writer.add_scalar(name, np.mean(metric), epoch)

        with open(self.test_summary_filename, 'a') as log_file:
            log_file.write(write_message + '\n')

        test_time_msg = 'Testing time used: {:.3f}s'.format(test_time)
        print_message += '\n' + test_time_msg
        print(print_message)
        with open(self.test_log_filename, 'a') as log_file:
            log_file.write(test_time_msg + '\n')

    def save_fake_omics(self, fake_dict, sample_list, feature_list):
        """
            save the fake omics data to disc

            Parameters:
                fake_dict (OrderedDict))                -- the fake omics data and the corresponding index
                sample_list (ndarray)                   -- the sample list for the input data
                feature_list (ndarray)                  -- the feature list of the generated omics data
        """
        output_sample_list = sample_list[fake_dict['index'].astype(int)]
        fake_df = pd.DataFrame(data=fake_dict['fake'].T, index=feature_list, columns=output_sample_list)
        print('Saving generated omics file...')
        save_file(self.param, fake_df, 'fake_A')
