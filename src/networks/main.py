from .mnist_LeNet import MNIST_LeNet, MNIST_LeNet_Autoencoder
from .cifar10_LeNet import CIFAR10_LeNet, CIFAR10_LeNet_Autoencoder
from .cifar10_LeNet_elu import CIFAR10_LeNet_ELU, CIFAR10_LeNet_ELU_Autoencoder
#from .ftops_ptransformer import ParticleTransformer
#from .ftops_ptransformer_SMids import ParticleTransformer
#from .ftops_ptransformer_SMcoupling import ParticleTransformer
from .ftops_Mlp import FTOPS_Mlp
from .ftops_ParticleNET import ParticleNet

def build_network(
                 #net_name, 
                 #input_dim,
                 #rep_dim=None,
                 #num_features=None,
                 #aux_dim=None,
                 ## network configurations
                 #embed_dims=[128, 512, 128],
                 #pair_embed_dims=[64, 64, 64],
                 ##Divide by four, divisble by 8, leave num_layers to 2 
                 #num_heads=8,
                 #num_layers=8,
                 #num_cls_layers=2,
                 #block_params=None,
                 #cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 #fc_params=[],
                 #aux_fc_params=[],
                 #activation='gelu',
                 #add_bias_attn=False,
                 #seq_len=-1,    # Required for add_bias_attn
                 ## misc
                 #trim=True,
                 #for_inference=False,
                 #use_amp=False,
                 **kwargs):

    # Copy kwargs to not alter the original dict
    kwargs = dict(kwargs)
    
#    # For ParticleTransformer
    net_name = kwargs['net_name']
    net_version = kwargs['net_version']
#    input_dim = kwargs['input_dim']
#    rep_dim = kwargs['rep_dim']
#    aux_dim = kwargs['aux_dim']
#    # network configurations
#    embed_dims = [128, 512, 128]
#    pair_embed_dims = [64, 64, 64]
#    num_heads = kwargs['num_heads']
#    num_layers = kwargs['num_layers']
#    num_cls_layers = kwargs['num_cls_layers']
#    block_params = kwargs['block_params']
#    cls_block_params = kwargs['cls_block_params']
#    fc_params = kwargs['fc_params']
#    aux_fc_params = kwargs['aux_fc_params']
#    activation = kwargs['activation']
#    add_bias_attn = kwargs['add_bias_attn']
#    seq_len = kwargs['seq_len']
#    trim = kwargs['trim']
#    for_inference = kwargs['for_inference']
#    use_amp = kwargs['use_amp']
#
#    # For Mlp
#    num_features = kwargs['num_features']


    """Builds the neural network."""

    implemented_networks = ('mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU', 'ftops_Transformer', 'ftops_Mlp', 'ftops_ParticleNET')
    assert net_name in implemented_networks

    net = None

    if net_name == 'mnist_LeNet':
        net = MNIST_LeNet()

    if net_name == 'cifar10_LeNet':
        net = CIFAR10_LeNet()

    if net_name == 'cifar10_LeNet_ELU':
        net = CIFAR10_LeNet_ELU()
    
    if net_name == 'ftops_Transformer':

        if net_version == 'SMids':
            from .ftops_ptransformer_SMids import ParticleTransformer
        elif net_version == 'SMcoupling':
            from .ftops_ptransformer_SMcoupling import ParticleTransformer
        elif net_version == 'standard':
            from .ftops_ptransformer import ParticleTransformer

        net = ParticleTransformer(
                 #input_dim,
                 #rep_dim,
                 #aux_dim,
                 ## network configurations
                 #embed_dims,
                 #pair_embed_dims,
                 #num_heads,
                 #num_layers,
                 #num_cls_layers,
                 #block_params,
                 #cls_block_params,
                 #fc_params,
                 #aux_fc_params,
                 #activation,
                 #add_bias_attn,
                 #seq_len,
                 #trim,
                 #for_inference,
                 #use_amp,
                 **kwargs)  

    if net_name == 'ftops_Mlp':
        net = FTOPS_Mlp(**kwargs)

    if net_name == 'ftops_ParticleNET':
        net = ParticleNet(**kwargs)

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
