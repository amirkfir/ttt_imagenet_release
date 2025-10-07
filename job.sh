#!/bin/sh

### General options
### -- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J Resnet_TTT
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=16GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
#BSUB -M 15GB
### -- set walltime limit: hh:mm --
#BSUB -W 48:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u amirkfir93@gmail.com
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o outputs/Output_%J.out
#BSUB -e outputs/Output_%J.err

source /work3/s242954/ttt_imagenet_release/ttt_new/bin/activate


# here follow the commands you want to execute with input.in as the input file
python /work3/s242954/ttt_imagenet_release/main.py --shared layer3 --group_norm 32 --workers 16 --outf results/resnet18_layer3_gn