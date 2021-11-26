"""
The training part for OmiTrans
"""
import time
from util import util
from params.train_params import TrainParams
from datasets import create_single_dataloader
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    full_start_time = time.time()

    # Get parameters
    param = TrainParams().parse()
    if param.deterministic:
        util.setup_seed(param.seed)

    # Dataset related
    dataloader, sample_list = create_single_dataloader(param, shuffle=True, enable_drop_last=True)
    print('The size of training set is {}'.format(len(dataloader)))
    # Get the dimension of two omics types
    param.A_dim = dataloader.get_A_dim()
    param.B_dim = dataloader.get_B_dim()
    print('The dimension of omics type A is %d' % param.A_dim)
    print('The dimension of omics type B is %d' % param.B_dim)
    if param.zo_norm:
        param.target_min = dataloader.get_values_min()
        param.target_max = dataloader.get_values_max()

    # Model related
    model = create_model(param)                     # Create a model given param.model and other parameters
    model.setup(param)                              # Regular setup for the model: load and print networks, create schedulers
    visualizer = Visualizer(param)                  # Create a visualizer to print results

    # Start the epoch loop
    for epoch in range(param.epoch_count, param.epoch_num + param.epoch_num_decay + 1):    # outer loop for different epochs
        epoch_start_time = time.time()              # Start time of this epoch

        # Training
        model.set_train()                           # Set train mode for training
        iter_load_start_time = time.time()          # Start time of data loading for this iteration
        losses_acc = model.init_losses_dict()       # Initialize the losses dictionary
        metrics_acc_train = model.init_metrics_dict()     # Initialize the metrics dictionary

        # Start training loop
        for i, data in enumerate(dataloader):  # Inner loop for different iteration within one epoch
            dataset_size = len(dataloader)
            actual_batch_size = len(data['index'])
            iter_start_time = time.time()           # Timer for computation per iteration
            if i % param.print_freq == 0:
                load_time = iter_start_time - iter_load_start_time       # Data loading time for this iteration
            model.set_input(data)                   # Unpack input data from the output dictionary of the dataloader
            model.update()                          # Calculate losses, gradients and update network parameters
            model.update_losses_dict(losses_acc)    # Update the losses dictionary
            model.update_metrics_dict(metrics_acc_train)  # Update the metrics dictionary
            if i % param.print_freq == 0:           # Print training losses and save logging information to the disk
                comp_time = time.time() - iter_start_time   # Computational time for this iteration
                visualizer.print_train_log(epoch, i, losses_acc, metrics_acc_train, load_time / actual_batch_size, comp_time / actual_batch_size, param.batch_size, dataset_size)
            iter_load_start_time = time.time()

        if param.save_model:
            if param.save_epoch_freq == -1:  # Only save networks during last epoch
                if epoch == param.epoch_num + param.epoch_num_decay:
                    print('Saving the model at the end of epoch {:d}'.format(epoch))
                    model.save_networks(str(epoch))
            elif epoch % param.save_epoch_freq == 0:  # Save both the generator and the discriminator every <save_epoch_freq> epochs
                print('Saving the model at the end of epoch {:d}'.format(epoch))
                # model.save_networks('latest')
                model.save_networks(str(epoch))

        # total num of epoch for the training phase
        total_epoch = param.epoch_num + param.epoch_num_decay
        # Running time for this epoch
        train_time = time.time() - epoch_start_time
        visualizer.print_train_summary(epoch, losses_acc, metrics_acc_train, train_time)
        model.update_learning_rate()                # update learning rates at the end of each epoch

    full_time = time.time() - full_start_time
    print('Full running time: {:.3f}s'.format(full_time))
