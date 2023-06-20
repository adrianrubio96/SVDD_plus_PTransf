from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
from .ftops_Mlp import FTOPS_Mlp, FTOPS_Mlp_Autoencoder 

def build_network(net_name, **kwargs):
    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ftops_Mlp')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()

    if net_name == 'ftops_Mlp':
        net = FTOPS_Mlp(**kwargs)    

    return net


def build_autoencoder(net_name, **kwargs):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ftops_Mlp')
    assert net_name in implemented_networks

    ae_net = None

    if net_name == 'mnist_LeNet':
        ae_net = MNIST_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet':
        ae_net = CIFAR10_LeNet_Autoencoder()

    if net_name == 'cifar10_LeNet_ELU':
        ae_net = CIFAR10_LeNet_ELU_Autoencoder()

    if net_name == 'ftops_Mlp':
        ae_net = FTOPS_Mlp_Autoencoder(**kwargs) 

    return ae_net
