from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .ftops import FTOPS_Dataset

def load_dataset(dataset_name, data_path, max_obj, normal_class, net_name):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', '4tops')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == '4tops':
        dataset = FTOPS_Dataset(root=data_path, max_obj=max_obj, normal_class=normal_class, net_name=net_name)


    return dataset
