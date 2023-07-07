from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

#import EarlyStopping
#from pytorchtools import EarlyStopping

import logging
import time
import torch
import torch.optim as optim
import numpy as np

import wandb

class DeepSVDDTrainer(BaseTrainer):

    # Start a W&B run
    #wandb.init(project='test') 

    def __init__(self, objective, R, c, nu: float, net_name, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

        # Network
        self.net_name = net_name

    def train_org(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # To automatically log gradients
        wandb.watch(net, log_freq=100)

        # Get train data loader
        train_loader, val_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')


        # Set learning rate scheduler
        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr = 1e-7, factor = 0.5, verbose = True)

        # initialize the early_stopping object
        #early_stopping = EarlyStopping(patience=50, verbose=True, path='checkpoints/checkpoint.pt')


        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')
 
        #self.c = torch.tensor([10, 10, 10, 10, 10]).to(self.device)
        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        for epoch in range(self.n_epochs):
            #net.train()
            #scheduler.step()
            #if epoch in self.lr_milestones:
            #    logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches_train = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                
                if net_name == 'ftops_Mlp':
                    inputs = inputs.to(self.device)
                
                if net_name == 'ftops_Transformer':
                    aux, tokens, momenta, id_int, mask = inputs
    
                    # Fetch data and move to device
                    tokens = tokens.to(self.device) 
                    momenta = momenta.to(self.device) 
                    id_int = id_int.to(self.device) 
                    mask = mask.to(self.device) 
                    aux = aux.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                ## Compute forward pass through model
                if net_name == 'ftops_Mlp':
                    outputs = net(inputs)
                if net_name == 'ftops_Transformer':
                    outputs = net(tokens, v=momenta, ids=id_int, mask=mask, aux=aux)
                
                ## Compute loss
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)

                ## Backpropagation
                loss.backward()

                ## Update weights
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches_train += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches_train))
            #wandb.log({"loss": np.log10(loss_epoch / n_batches_train)}) 

            #scheduler.step()
            #if epoch in self.lr_milestones:
                #logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            #    logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))

            net.eval()            
            with torch.no_grad():
             validation_loss = 0
             val_data_size = 0
             n_batches_val = 0
             for data in val_loader:
                inputs, _, _ = data

                if net_name == 'ftops_Mlp':
                    inputs = inputs.to(self.device)
                    outputs = net(inputs)

                    val_data_size += inputs.shape[0]
                
                if net_name == 'ftops_Transformer':
                    aux, tokens, momenta, id_int, mask = inputs
    
                    # Fetch data and move to device
                    tokens = tokens.to(self.device) 
                    momenta = momenta.to(self.device) 
                    id_int = id_int.to(self.device) 
                    mask = mask.to(self.device) 
                    aux = aux.to(self.device)
    
                    # Compute forward pass through model
                    outputs = net(tokens, v=momenta, ids=id_int, mask=mask, aux=aux)

                    val_data_size += tokens.shape[0]

                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                
                validation_loss += loss.item()
                n_batches_val += 1

             logger.info('  Validation Loss: {:.8f}'
                         .format(validation_loss /len(val_loader))) #val_data_size)) 
 
             wandb.log({"loss": np.log10(loss_epoch / n_batches_train), "val_loss": np.log10(validation_loss / len(val_loader))})

            #If LR scehdule is on
            #early_stopping(validation_loss/len(val_loader), net)
        
            #if early_stopping.early_stop:
            #  print("Early stopping")
            #  break

            scheduler.step(validation_loss / len(val_loader))

            wandb.log({"lr": optimizer.param_groups[0]['lr']})

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data

                aux, tokens, momenta, id_int, mask = inputs

                # Fetch data and move to device
                tokens = tokens.to(self.device) 
                momenta = momenta.to(self.device) 
                id_int = id_int.to(self.device) 
                mask = mask.to(self.device) 
                aux = aux.to(self.device)

                # Compute forward pass through model
                outputs = net(tokens, v=momenta, ids=id_int, mask=mask, aux=aux)

                #inputs = inputs.to(self.device)
                #outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        mask_bkg = (labels == 0)
        mask_sgn = (labels == 1)
        scores_bkg = np.log10(scores[mask_bkg])
        scores_sgn = np.log10(scores[mask_sgn])

        data_bkg = [[s] for s in scores_bkg]

        table_bkg = wandb.Table(data=data_bkg, columns=["scores"])
        hist_bkg = wandb.plot.histogram(table_bkg, "scores", 
 	  title="Prediction Score Distribution background")

        data_sgn = [[s] for s in scores_sgn]

        table_sgn = wandb.Table(data=data_sgn, columns=["scores"])
        hist_sgn = wandb.plot.histogram(table_sgn, "scores", 
 	  title="Prediction Score Distribution Signal")

        wandb.log({'histogram_1': hist_bkg, 'histogram_2': hist_sgn}) 

        #combined = np.stack((scores, labels), axis=-1)
        #np.save('scores', combined)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        wandb.log({"AUC": 100*self.test_auc})

        #wandb.log({"roc": wandb.plot.roc_curve(y_true=labels, y_probas=scores)})

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data

                aux, tokens, momenta, id_int, mask = inputs

                # Fetch data and move to device
                tokens = tokens.to(self.device) 
                momenta = momenta.to(self.device) 
                id_int = id_int.to(self.device) 
                mask = mask.to(self.device) 
                aux = aux.to(self.device)

                outputs = net(tokens, v=momenta, ids=id_int, mask=mask, aux=aux)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
