#/usr/bin/env python

import os, sys
import yaml
PWD = os.getcwd()

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------
def main ():
    config = optParser ()

    prefix = config.prefix
    name = config.name if config.name else "test"
    yaml_ = config.config if config.config else None
    batchFolder = "batch__%s" % name
    extraCommand = config.options

    # Read yaml file
    with open(yaml_, 'r') as f:
        yaml_dic = yaml.load(f, Loader=yaml.FullLoader)

    # Get architecture name and read hyperparameters from yaml file
    architecture = config.architecture if config.architecture else None
    default_hypers = yaml_dic[architecture]["training"]

    # Read hyperparameters from command line
    lr = config.lr if config.lr else None
    n_epochs = config.n_epochs if config.n_epochs else None
    lr_milestone = config.lr_milestone if config.lr_milestone else None
    weight_decay = config.weight_decay if config.weight_decay else None
    batch_size = config.batch_size if config.batch_size else None
    rep_dim = config.rep_dim if config.rep_dim else None

    # Make dictionary of hyperparameters to be scanned
    hyperparameters = {}
    if lr: hyperparameters["lr"] = lr.split(",")
    if n_epochs: hyperparameters["n_epochs"] = n_epochs.split(",")
    if lr_milestone: hyperparameters["lr_milestone"] = lr_milestone.split(",")
    if weight_decay: hyperparameters["weight_decay"] = weight_decay.split(",")
    if batch_size: hyperparameters["batch_size"] = batch_size.split(",")
    if rep_dim: hyperparameters["rep_dim"] = rep_dim.split(",")

    # Make the batch folder
    if os.path.exists (batchFolder):
        print ("[info]     .. cleaning ..")
        os.system ("rm %s/*err" % batchFolder)
        os.system ("rm %s/*log" % batchFolder)
        os.system ("rm %s/*out" % batchFolder)
        os.system ("rm %s/*.sh" % batchFolder)
    else: 
        print ("[info]     .. creating ..")
        os.system ("mkdir  %s" % batchFolder)

    # Loop over hyperparameters
    print ("[info] Looping over hyperparameter variations")
    for hypname in hyperparameters.keys():
        print ("[info] . Hyperparameter: %s" % hypname)
        for hypvalue in hyperparameters[hypname]:
            createShell (hypname, hypvalue, default_hypers, architecture, prefix, batchFolder, extraCommand)
    subScript = "%s.sub" % name
    print ("[info] Making the submission script: %s" % subScript)
    createBatch (batchFolder, subScript)


# ---------------------------------------------------------------
# create shell script
# ---------------------------------------------------------------
def createShell(hypname, hypvalue, default_hyps, architecture, prefix, batchFolder, extraCommand):

    # Set default command to run
    #command = "python main_iter.py 4tops ftops_Transformer ../log/DarkMachines /lustre/ific.uv.es/grid/atlas/t3/adruji/DarkMachines/arrays/v1/channel1/v11/h5/DarkMachines.h5  --objective one-class --lr 1e-5 --n_epochs 500 --lr_milestone 50 --batch_size 500 --weight_decay 0.5e-6 --rep_dim 10 --pretrain False --network_name %s" % (default_name)
    command = "python main_iter.py 4tops %s ../log/DarkMachines /lustre/ific.uv.es/grid/atlas/t3/adruji/DarkMachines/arrays/v1/channel1/v11/h5/DarkMachines.h5  --objective one-class  --pretrain False " % architecture

    # Complete command with options
    runname_list = []
    # Loop over default hyperparameters dictionary
    for hdef_name in default_hyps.keys():
        if hdef_name==hypname:
            command += " --%s %s" % (hdef_name, str(hypvalue))
            runname_list.append(hdef_name+str(hypvalue))
        else:
            command += " --%s %s" % (hdef_name,str(default_hyps[hdef_name]))
            runname_list.append(hdef_name+str(default_hyps[hdef_name]))
      
    # Set run name
    runname = "_".join([prefix,"_".join(runname_list)])

    # Simplify run name
    runname = runname.replace("n_epochs","e").replace("lr_milestone","lrm").replace("weight_decay","wd").replace("batch_size","b").replace("rep_dim","z")
    
    # Add run name to command
    command += " --network_name %s_%s" % (prefix,runname)
    
    print("[info] . Run name: %s" % runname)
    shellName = "%s/%s.sh" % (batchFolder, runname)
    runningFolder = os.getcwd()

    s = open (shellName, "w+")
    s.write ("#!/usr/bin/bash\n")
    s.write ('cd %s\n' % PWD)
    s.write ('source /lhome/ific/a/adruji/DarkMachines/unsupervised/Deep_SVDD_PTransf/setup.sh\n' )
    s.write ("cd src\n")
    s.write ("%s %s\n" % (command, extraCommand if extraCommand else ""))
    s.write ("deactivate\n")
    s.close()
    os.system ("chmod +x %s" % (shellName))

# ---------------------------------------------------------------
# create batch submission script
# ---------------------------------------------------------------
def createBatch(folderName, subName):
    print("[info] creating the condor_sub script: {0}".format (subName))
    
    sub = open (subName, "w+")
    sub.write("    universe                = vanilla"+"\n\n")
    sub.write("    executable              = $(filename)"+"\n")
    sub.write("    arguments               = $Fnx(filename)"+"\n")
    sub.write("    output                  = $Fpn(filename).$(ClusterId).$(ProcId).out"+"\n")
    sub.write("    error                   = $Fpn(filename).$(ClusterId).$(ProcId).err"+"\n")
    sub.write("    log                     = $Fp(filename)/$(ClusterID).log"+"\n\n")
    sub.write('    +JobFlavour             = "workday"'+"\n")
    sub.write('    +MaxRuntime             = "14400"'+"\n\n") # 4h    
    sub.write('    #request_Cpus            = 4'+"\n")
    sub.write('    request_Gpus            = 1'+"\n")
    sub.write('    #request_Memory          = 10000'+"\n\n")
    sub.write('    +UseNvidiaV100          = True'+"\n\n") # 4h                                                                                                                                                                   '+"\n")
    sub.write("    queue filename matching ({0}/*.sh)\n".format (folderName))
    sub.close()
    

# ---------------------------------------------------------------
# option parser
# ---------------------------------------------------------------
def optParser():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-o","--options", dest="options", help="Extra options for the command line ",default=None)
    parser.add_option("-l","--lr", dest="lr", help="Comma-separated list of learning rate values for the training",default=None)
    parser.add_option("-e","--n_epochs", dest="n_epochs", help="Comma-separated list of epochs values for the training",default=None)
    parser.add_option("-w","--weight_decay", dest="weight_decay", help="Comma-separated list of weight decay values for the training",default=None)
    parser.add_option("-m","--lr_milestone", dest="lr_milestone", help="Comma-separated list of learning rate milestones for the training",default=None)
    parser.add_option("-b","--batch-size", dest="batch_size", help="Comma-separated list of batch size values for the training",default=None)
    parser.add_option("-z","--rep_dim", dest="rep_dim", help="Comma-separated list of z_dim values for latent space",default=None)
    parser.add_option("-f","--folder-name", dest="name", help="Name of the folder",default=None)
    parser.add_option("-p","--prefix", dest="prefix", help="Default name of the runs",default=None)
    parser.add_option("-a","--architecture", dest="architecture", help="String name of the architecture: ftops_Mlp, ftops_Transformer, ftops_ParticleNET",default=None)
    parser.add_option("-c","--config", dest="config", help="Configuration file for default hyperparameters",default=None)

    (config, sys.argv[1:]) = parser.parse_args(sys.argv[1:])
    return config

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script
    main()
