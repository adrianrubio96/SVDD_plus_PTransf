#/usr/bin/env python

import os, sys
PWD = os.getcwd()

# ---------------------------------------------------------------
# main function
# ---------------------------------------------------------------
def main ():
    config = optParser ()
    if not config.default:
        print ("[ERROR] Please provide a default name for the runs")
        sys.exit ()

    default_name = config.default
    name = config.name if config.name else "test"
    batchFolder = "batch__%s" % name
    extraCommand = config.options

    # Get hyperparameters from parser options
    lr = config.lr if config.lr else None
    epochs = config.epochs if config.epochs else None
    milestones = config.milestones if config.milestones else None
    weight_decay = config.weight_decay if config.weight_decay else None
    batch_size = config.batch_size if config.batch_size else None
    z_dim = config.z_dim if config.z_dim else None

    hyperparameters = {}
    if lr: hyperparameters["lr"] = lr.split(",")
    if epochs: hyperparameters["epochs"] = epochs.split(",")
    if milestones: hyperparameters["milestone"] = milestones.split(",")
    if weight_decay: hyperparameters["wdecay"] = weight_decay.split(",")
    if batch_size: hyperparameters["batch"] = batch_size.split(",")
    if z_dim: hyperparameters["zdim"] = z_dim.split(",")

    if os.path.exists (batchFolder):
        print ("[info]     .. cleaning ..")
        os.system ("rm %s/*err" % batchFolder)
        os.system ("rm %s/*log" % batchFolder)
        os.system ("rm %s/*out" % batchFolder)
        os.system ("rm %s/*.sh" % batchFolder)
    else: 
        print ("[info]     .. creating ..")
        os.system ("mkdir  %s" % batchFolder)

    print ("[info] Looping over hyperparameter variations")
    for hypname in hyperparameters.keys():
        print ("[info] . Hyperparameter: %s" % hypname)
        for hypvalue in hyperparameters[hypname]:
            createShell (hypname, hypvalue, default_name, batchFolder, extraCommand)
    subScript = "%s.sub" % name
    print ("[info] Making the submission script: %s" % subScript)
    createBatch (batchFolder, subScript)


# ---------------------------------------------------------------
# create shell script
# ---------------------------------------------------------------
def createShell(hypname, hypvalue, default_name, batchFolder, extraCommand):

    # Set default command to run
    command = "python main_iter.py 4tops ftops_Transformer ../log/DarkMachines /lustre/ific.uv.es/grid/atlas/t3/adruji/DarkMachines/arrays/v1/channel1/v11/h5/DarkMachines.h5  --objective one-class --lr 1e-4 --n_epochs 100 --lr_milestone 50 --batch_size 500 --weight_decay 0.5e-6 --rep_dim 10 --pretrain False --network_name %s" % (default_name)

    # Set specific name
    default_hyps = default_name.split("_")
    runname_list = []
    for h in default_hyps:
        if hypname in h:
            runname_list.append(hypname+hypvalue)
        else:
            runname_list.append(h)
            
    runname = "_".join(runname_list)
    command = command.replace(default_name, runname)
    
    print("[info] . Run name: %s" % runname)
    shellName = "%s/%s.sh" % (batchFolder, runname)
    runningFolder = os.getcwd()

    # Set specific command
    if hypname=="lr": command = command.replace("--lr 1e-4", "--lr %s" % hypvalue)
    elif hypname=="epochs": command = command.replace("--n_epochs 100", "--n_epochs %s" % hypvalue)
    elif hypname=="milestone": command = command.replace("--lr_milestone 50", "--lr_milestone %s" % hypvalue)
    elif hypname=="wdecay": command = command.replace("--weight_decay 0.5e-6", "--weight_decay %s" % hypvalue)
    elif hypname=="batch": command = command.replace("--batch_size 500", "--batch_size %s" % hypvalue)
    elif hypname=="zdim": command = command.replace("--rep_dim 10", "--rep_dim %s" % hypvalue)
    else:
        print("[WARNING] Hyperparameter %s not found in default command" % hypname)
        return  
    
    s = open (shellName, "w+")
    s.write ("#!/usr/bin/bash\n")
    s.write ('cd %s\n' % PWD)
    s.write ('source setup.sh\n' )
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
    parser.add_option("-l","--learning-rate", dest="lr", help="Comma-separated list of learning rate values for the training",default=None)
    parser.add_option("-e","--epochs", dest="epochs", help="Comma-separated list of epochs values for the training",default=None)
    parser.add_option("-w","--weight-decay", dest="weight_decay", help="Comma-separated list of weight decay values for the training",default=None)
    parser.add_option("-m","--lr-milestones", dest="milestones", help="Comma-separated list of learning rate milestones for the training",default=None)
    parser.add_option("-b","--batch-size", dest="batch_size", help="Comma-separated list of batch size values for the training",default=None)
    parser.add_option("-z","--z-dim", dest="z_dim", help="Comma-separated list of z_dim values for latent space",default=None)
    parser.add_option("-f","--folder-name", dest="name", help="Name of the folder",default=None)
    parser.add_option("-d","--default", dest="default", help="Default name of the runs",default=None)

    (config, sys.argv[1:]) = parser.parse_args(sys.argv[1:])
    return config

# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------
if __name__ == "__main__":
    # execute only if run as a script
    main()
