cmd=/usr/local/MATLAB/R2018b/bin/matlab
imagename=matthias-k/mit-saliency-benchmark
test:
	echo UID: $(shell id -u)

run: build
	docker run --rm -it --user root -e NB_UID=$(shell id -u) -e NB_GID=$(shell id -g) -e NB_USER=$(USER) -v /usr/local/MATLAB/R2018b:/usr/local/MATLAB/R2018b $(imagename) start.sh $(cmd)

build: copy-libs
	docker build -t $(imagename) .

singularity: build
	docker run -v /var/run/docker.sock:/var/run/docker.sock -v $(shell pwd)/../singularity:/output --privileged -it --rm quay.io/singularity/docker2singularity --name saliency-benchmark.sif $(imagename)
	cd .. && fix_permissions `pwd`/singularity

copy-libs:
	cp /usr/lib64/libXt.so.6 lib
	cp /usr/lib64/libX11.so.6 lib
	cp /usr/lib64/libSM.so.6 lib
	cp /usr/lib64/libICE.so.6 lib
	cp /usr/lib64/libxcb.so.1 lib
	cp /usr/lib64/libXau.so.6 lib

copy-python:
	cp -r /usr/local/MATLAB/R2018b/extern .

copy-matlab:
	cp -r /usr/local/MATLAB/R2018b .

get-scripts:
	rm -f install_bazel.sh
	rm -f install_tensorflow.sh
	wget https://raw.githubusercontent.com/bethgelab/docker-deeplearning/future/deeplearning/install_bazel.sh
	wget https://raw.githubusercontent.com/bethgelab/docker-deeplearning/future/deeplearning/install_tensorflow.sh
