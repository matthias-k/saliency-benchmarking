#!/bin/bash

set -x

imagename="matthias-k/mit-saliency-benchmark"

make build-docker 
#NV_GPU=$GPU nvidia-docker run --rm -it -v `pwd`:`pwd` --user $(id -u):$(id -g) -w `pwd` matthiask/mit-saliency-benchmark $@
NV_GPU=$GPU nvidia-docker \
        run --rm -it --user root \
        -e NB_UID=`id -u` -e NB_GID=`id -g` -e NB_USER=$USER -e GRANT_SUDO=yes \
        -e USER=`whoami` -e GPU=$GPU \
        -v `pwd`:`pwd` -w `pwd` -v /usr/local/MATLAB/R2018b:/usr/local/MATLAB/R2018b $imagename start.sh $@
#
#run --rm -it -v `pwd`:`pwd` --user $(id -u):$(id -g) -w `pwd` matthiask/mit-saliency-benchmark $@
