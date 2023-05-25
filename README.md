## Pytorch implementation of a Particle Transformer as a SVDD for anomaly detection.
This particular branch is used in the context of Particle Physics, performing unsuperised learning of simulated collisions in the ATLAS detector, for which the DarkMachines dataset is being used. The purpose is to test how competitive a powerfull architecture performs, such as the Transformers, when performing unsupervised learning to search for New Physics.

## Setup at IFIC machines (ARTEMISA)
First, we install python3.9 inside the ATLAS setup (Python 3.9.12 is being used):
``` 
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh -3
lsetup "root 6.26.08-x86_64-centos7-gcc11-opt"
```
Then, we create a python environment and reset the PYTHONPATH to point at the PYTHONPATH in the virtual environment (additionally we add the ROOT libraries to it):
```
python3 -m venv myenv
source myenv/bin/activate
export PYTHONPATH=$VIRTUAL_ENV/lib64/python3.9/:$ROOTSYS/lib
```
Now, all the packages specified in requirements.txt are installed (torch version is 1.12.1):
```
python -m ensurepip --upgrade
pip install -r requirements.txt
```

## Run a training
Interactively, you can use the `run.sh` script in the `src` directory, in which the following command would be run:
```
gpurun python main_iter.py 4tops ftops_Transformer ../log/DarkMachines /path/to/input/h5/ --objective one-class --lr 0.0001 --n_epochs 5 --lr_milestone 50 --batch_size 500 --weight_decay 0.5e-6 --pretrain False --network_name ParT_default_5
```

Running on batch, longer trainings can be achieved. Actually, the default training will loop over a set of latent space dimensions (2, 5, 10, 20, 25),
which will be saved as independent trainings.

## WANDB framework for metrics
The output of the different trainings can be analised using the WANDB (Weights AND Biases) framework.
A link is provided in the log of the training.