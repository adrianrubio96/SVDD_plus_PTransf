## Pytorch implementation of a Particle Transformer as a SVDD for anomaly detection.
This repository is used in the context of Particle Physics, performing unsupervised learning of simulated collisions in the ATLAS detector, for which the DarkMachines dataset is being used. The purpose is to test how competitive the most powerful classifiers are when performing unsupervised learning to search for New Physics.

For the moment, three architectures have been implemented:
* A simple Multi-Layer Perceptron: used as a benchmark to compare with more sophisticated networks.
* Particle Transformer: adaptation of a Transformer architecture to particle physics, which already showed the best performance for jet tagging.
* ParticleNet: graph NN that makes use of the EdgeConv and the Dinamic Graph CNN technique methods, which also showed the best performance for jet tagging before Particle Transformer was developed.

A DeepSVDD approach is used in order to adapt such complex architectures for an anomaly detection task. This is achieved by adding some fully connected layers to the end of these networks, in which the last layer represents the latent space and each event will be represented as a point in such hyper-space. In the same way as a standard SVDD, the training is performed by minimising a loss function computed as the distance to a particular center.

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

## Input dataset
The datasets are downloaded in csv format from zenodo corresponding to the [Unsupervised Hackaton](https://zenodo.org/record/3961917) performed by the DarkMachines collaboration. These files are splitted by channels, with a background file and several signals per channel. 
However, this framework reads h5 files, which have been preprocessed by this [public code](https://github.com/adrianrubio96/DarkMachines). The main information to be taken into account about the input datasets is:
* One file per signal: A single h5 file will be read by the framework. A different h5 exists for each of the signals but the background is the same in all of them (for the same channel). An additional file is also created with all signals merged in order to get an overall evaluation.
* Variables: four-momentum variables for the different objects (jets, b-jets, e+, e+, mu+, mu-, photons), labels for each object type and missing transverse momentum information.

## Run local training
If you have an input file with the charasteristics given above, you can directly run a training. Interactively, you can use the `run.sh` script in the `src` directory, in which a command like the following would be run:
```
gpurun python main_iter.py 4tops ftops_Transformer ../log/DarkMachines /path/to/input/h5/DarkMachines.h5 --objective one-class --lr 0.0001 --n_epochs 5 --lr_milestone 50 --batch_size 500 --weight_decay 0.5e-6 --pretrain False --network_name ParT_DarkM_
```
where `ftops_Transformer`, `ftops_ParticleNET`, `ftops_Mlp` are the names given to the architectures we want to run with.


### Set hyperparameters
The most important information about the definition of the networks and how the training is performed can be found and easily tuned in `src/config.yml`. 

## Run on the batch
To run on batch, the `batchHyper.py` script creates a folder with .sh jobs tu be run in parallel together with the corresponding submission script. This is useful to parallelise jobs when needed to scan over different hyperparameters. To scan over learning rates and batch size:
```
python batchHyper.py --architecture ftops_Transformer --folder-name ParT-scan-test --prefix ParT_DarkM_v21  --lr 1e-3,1e-4,1e-5 --batch_size 50,500,5000
```
where the remaining default hyperparametes will be taken from `src/config.yml`. 

In this example, you will find a folder called `batch__ParT-scan-test` and a submission file `ParT-scan-test.sub`. The job scripts located in the batch folder will be given automatically a name similar to `ParT_DarkM_v21_opadam_e100_lr1e-4_b500_schPlateau_wd0.5e-6_z10-test.sh`, where the main training hyperparameters are already contained in the name.

## Run evaluation locally
The trained models are saved in .tar files. Following previous example, the model would be saved as `../log/DarkMachines/model_ParT_DarkM_v21_opadam_e100_lr1e-4_b500_schPlateau_wd0.5e-6_z10-test.tar`. The models can be loaded in order to evaluate on different signals. The command to run such a local test is:
```
gpurun python main_iter.py 4tops ftops_Transformer ../log/DarkMachines /lustre/ific.uv.es/grid/atlas/t3/adruji/DarkMachines/arrays/v2/chan1/v21/h5/DarkMachines_signal_name.h5  --objective one-class --network_name ParT_DarkM_v21_opadam_e100_lr1e-4_b500_schPlateau_wd0.5e-6_z10-test --test_mode True --test_name signal_name --load_model ../log/DarkMachines/model_ParT_DarkM_v21_opadam_e100_lr1e-4_b500_schPlateau_wd0.5e-6_z10-test.tar
```
where `signal_name` can be any of the signals available in the DarkMachines dataset and for which a h5 file has been created.

## WANDB framework for metrics
The output of the different trainings can be analised using the WANDB (Weights AND Biases) framework.
A link is provided in the log of the training.
