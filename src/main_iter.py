import os
import glob
import click
import torch
import logging
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from deepSVDD import DeepSVDD
from datasets.main import load_dataset

import wandb


def plot_loghist(x, bins, alpha, normalised=True, logX=False):
    # Make histograms
    hist, bins = np.histogram(x, bins=bins)

    # Normalise if specified
    if normalised:
        hist = hist / float(np.sum(hist))
    
    # Set log scale for X axis
    if logX:
        bins_ = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.hist(x, bins=bins_, alpha=alpha)
        plt.xscale('log')
    else:
        bins_ = bins
        plt.hist(x, bins=bins_, alpha=alpha)


################################################################################
# Settings
################################################################################
@click.command()
@click.option('--network_name', type=str, default='fcn_2l_32_16', help='Name of the network')
@click.argument('dataset_name', type=click.Choice(['mnist', 'cifar10', '4tops']))
@click.argument('net_name', type=click.Choice(['mnist_LeNet', 'cifar10_LeNet', 'cifar10_LeNet_ELU','ftops_Transformer','ftops_Mlp']))
@click.argument('xp_path', type=click.Path(exists=True))
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--load_config', type=click.Path(exists=True), default=None,
              help='Config JSON-file path (default: None).')
@click.option('--load_model', type=click.Path(exists=True), default=None,
              help='Model file path (default: None).')
@click.option('--objective', type=click.Choice(['one-class', 'soft-boundary']), default='one-class',
              help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
@click.option('--nu', type=float, default=0.1, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
@click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
@click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
@click.option('--optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for Deep SVDD network training.')
@click.option('--lr', type=float, default=0.001,
              help='Initial learning rate for Deep SVDD network training. Default=0.001')
@click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
#@click.option('--lr_milestone', type=int, default=[0], multiple=True,
#              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--lr_milestone', default=[0], multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
@click.option('--weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
@click.option('--pretrain', type=bool, default=True,
              help='Pretrain neural network parameters via autoencoder.')
@click.option('--ae_optimizer_name', type=click.Choice(['adam', 'amsgrad']), default='adam',
              help='Name of the optimizer to use for autoencoder pretraining.')
@click.option('--ae_lr', type=float, default=0.001,
              help='Initial learning rate for autoencoder pretraining. Default=0.001')
@click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
@click.option('--ae_lr_milestone', type=int, default=[0], multiple=True,
              help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
@click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
@click.option('--ae_weight_decay', type=float, default=1e-6,
              help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
@click.option('--n_jobs_dataloader', type=int, default=0,
              help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
@click.option('--normal_class', type=int, default=0,
              help='Specify the normal class of the dataset (all other classes are considered anomalous).')
@click.option('--rep_dim', type=int, default=[2, 5, 10, 20, 25], multiple=True, 
              help='Specify the latent space dimensions.')


def main(network_name, dataset_name, net_name, xp_path, data_path, load_config, load_model, objective, nu, device, seed, optimizer_name, lr, n_epochs, lr_milestone, batch_size, weight_decay, pretrain, ae_optimizer_name, ae_lr, ae_n_epochs, ae_lr_milestone, ae_batch_size, ae_weight_decay, n_jobs_dataloader, normal_class, rep_dim):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """
    isExist = os.path.exists(xp_path)
    if not isExist:

     # Create a new directory because it does not exist
     os.makedirs(xp_path)
     print("A new " + xp_path  + " directory is created!")

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
 
    dataset = load_dataset(dataset_name, data_path, normal_class, net_name)

    # Extract number of feautures
    train_loader, _, _= dataset.loaders(batch_size=10)
    for data in train_loader:
        inputs, _, _ = data
        break
    
    if net_name == 'ftops_Mlp':
        num_features = int(inputs.shape[1])
    if net_name == 'ftops_Transformer':
        num_features = 18

    # Pass through keyboard num of dimensions and feautures via dictionary
    # to set_network

    #Here the loop
    #for _z in tqdm(cfg.settings['rep_dim']): # Not looping anymore on latent space dimensions

    #print('Latent space dimensions', _z)

    # Start a W&B run
    wandb.init(project='test', name=cfg.settings['network_name']) # + '_dim_' + str(_z)) 

    input_dim = 9
    embed_dims = 128
    pair_embed_dims = 64  
    fc_nodes = 64 
    aux_dim = 2  

    set_network_dic = {'net_name': net_name,
                       'num_features': num_features, 
                       'rep_dim': cfg.settings['rep_dim'][0], 
                       'input_dim': input_dim, 
                       'embed_dims': embed_dims, 
                       'pair_embed_dims': pair_embed_dims, 
                       'fc_nodes': fc_nodes,
                       'aux_dim': aux_dim,
                       #Divide by four, divisble by 8, leave num_layers to 2 
                       'num_heads': 8,
                       'num_layers': 8,
                       'num_cls_layers': 2,
                       'block_params': None,
                       'cls_block_params': {'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                       'fc_params': [],
                       'aux_fc_params': [],
                       'activation': 'gelu',
                       'add_bias_attn': False,
                       'seq_len': -1,    # Required for add_bias_attn
                       # misc
                       'trim': True,
                       'for_inference': False,
                       'use_amp': False,}

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(net_name, cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network(**set_network_dic)
 
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(dataset,
                           net_name=net_name,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader, 
                           **set_network_dic)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

     # Train model on dataset
    deep_SVDD.train(dataset,
                    net_name=net_name,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader)

    # Test model
    deep_SVDD.test(dataset, net_name=net_name, device=device, n_jobs_dataloader=n_jobs_dataloader)

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    #idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=xp_path + '/results_' + cfg.settings['network_name']+ '.json') # + '_zdim_' + str(_z) + '.json')
    deep_SVDD.save_model(export_model=xp_path + '/model_' + cfg.settings['network_name']+ '.tar', save_ae=False) #  + '_zdim_' + str(_z) + '.tar', save_ae=False)
    cfg.save_config(export_json=xp_path + '/config_' + cfg.settings['network_name']+ '.json') # + '_zdim_' + str(_z) + '.json')

    #_, labels, scores = zip(*deep_SVDD.results)
    #labels = np.array(labels)
    #scores = np.array(scores)

    mask_bkg = (labels == 0)
    mask_sgn = (labels == 1)
    scores_bkg = scores[mask_bkg]
    scores_sgn = scores[mask_sgn]

    number_bins = 100
    plot_loghist(scores_bkg, bins=number_bins, alpha=0.3)
    plot_loghist(scores_sgn, bins=number_bins, alpha=0.7)  
    plt.legend(['bkg', 'sgn'])
    plt.xlabel('scores')
    
    # Save scores plot adding trial number

    ## Deine plot name
    plotname = xp_path + '/scores_' + cfg.settings['network_name']+ '_trial_*.pdf' # + '_zdim_' + str(_z) + '.pdf'

    ## Get list of all files with such name
    plot_list = glob.glob(plotname)

    ## Get list of trials
    trial_list = [int(plot.split('_')[-1].split('.')[0]) for plot in plot_list if 'trial' in plot]

    ## Get trial number
    trial_number = 0
    if len(trial_list) > 0:
        trial_number = max(trial_list) + 1 

    ## Locate in plot name
    plotname = plotname.replace('*', str(trial_number))
    
    plt.savefig(plotname)

    wandb.finish()

    del deep_SVDD

if __name__ == '__main__':


    main()
