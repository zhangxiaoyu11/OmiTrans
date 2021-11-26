"""
The testing part for OmiTrans
"""
import time
from util import util
from params.test_params import TestParams
from datasets import create_single_dataloader
from models import create_model
from util.visualizer import Visualizer


if __name__ == '__main__':
    # Get testing parameter
    param = TestParams().parse()
    if param.deterministic:
        util.setup_seed(param.seed)

    # Dataset related
    dataloader, param.sample_list = create_single_dataloader(param, shuffle=False)  # No shuffle for testing
    print('The size of testing set is {}'.format(len(dataloader)))
    # Get sample list of the dataset
    sample_list = dataloader.get_sample_list()
    # Get feature list of the dataset
    feature_list_A = dataloader.get_feature_list_A()
    # Get the dimension of two omics types
    param.A_dim = dataloader.get_A_dim()
    param.B_dim = dataloader.get_B_dim()
    print('The dimension of omics type A is %d' % param.A_dim)
    print('The dimension of omics type B is %d' % param.B_dim)
    if param.zo_norm:
        param.target_min = dataloader.get_values_min()
        param.target_max = dataloader.get_values_max()

    # Model related
    model = create_model(param)                 # Create a model given param.model and other parameters
    model.set_eval()
    visualizer = Visualizer(param)              # Create a visualizer to print results

    # TESTING
    model.setup(param)  # load saved networks for testing
    metrics_acc = model.init_metrics_dict()  # Initialize the metrics dictionary
    if param.save_fake:  # Initialize the fake array during the last epoch
        fake_dict = model.init_fake_dict()
    test_start_time = time.time()  # Start time of testing

    for i, data in enumerate(dataloader):
        dataset_size = len(dataloader)
        model.set_input(data)  # Unpack input data from the output dictionary of the dataloader
        model.test()  # Run forward to get the fake omics data
        model.update_metrics_dict(metrics_acc)  # Update the metrics dictionary
        if param.save_fake:  # Update the fake array during the last epoch
            fake_dict = model.update_fake_dict(fake_dict)
        visualizer.print_test_log(param.epoch_to_load, i, metrics_acc, param.batch_size, dataset_size)
    test_time = time.time() - test_start_time
    # Save average metrics of this epoch to the disk
    visualizer.print_test_summary(param.epoch_to_load, metrics_acc, test_time)

    if param.save_fake:  # Save the fake omics data
        visualizer.save_fake_omics(fake_dict, sample_list, feature_list_A)
