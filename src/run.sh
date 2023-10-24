dir=$PWD
src_dir=/lhome/ific/a/adruji/DarkMachines/unsupervised/Deep_SVDD_PTransf/src
h5_file=/lustre/ific.uv.es/grid/atlas/t3/adruji/DarkMachines/arrays/v2/chan1/v21/h5/DarkMachines_all.h5
cd $src_dir
#gpurun python main_iter.py 4tops ftops_Transformer ../log/4tops ../data --objective one-class --lr 0.0001 --n_epochs 5 --lr_milestone 50 --batch_size 500 --weight_decay 0.5e-6 --pretrain False 
gpurun python main_iter.py 4tops ftops_Transformer ../log/DarkMachines $h5_file --objective one-class --lr 0.0001 --n_epochs 2 --lr_milestone 50 --batch_size 500 --weight_decay 0.5e-6 --pretrain False 
cd $dir