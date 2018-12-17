FROM nvidia/cuda:9.0-cudnn7-devel
#FROM bethgelab/deeplearning:cuda9.0-cudnn7
MAINTAINER Matthias Kümmerer <matthias.kuemmerer@bethgelab.org>

RUN apt-get update -qq \
  && DEBIAN_FRONTEND=noninteractive apt-get install -yq -qq --no-install-recommends \
  ca-certificates \
  gcc \
  git \
  libdpkg-perl \
  python3-pip \
  software-properties-common


# install python 3.6
RUN add-apt-repository ppa:jonathonf/python-3.6 \
 && apt-get update \
 && apt-get install -y python3.6 python3.6-dev \
 && rm /usr/bin/python3 \
 && ln /usr/bin/python3.6 /usr/bin/python3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN cd /usr/lib/python3/dist-packages \
 && ln -s apt_pkg.cpython-35m-x86_64-linux-gnu.so apt_pkg.cpython-36m-x86_64-linux-gnu.so \
 && ln -s apt_inst.cpython-35m-x86_64-linux-gnu.so apt_inst.cpython-36m-x86_64-linux-gnu.so

# We have to use `pip` (instead of `pip2`) to upgrade pip, since these two are
# different and only `pip` works. After the upgrade, we can start using `pip2`.
RUN pip3 install setuptools
RUN pip3 install wheel

RUN pip3 install \
  boltons \
  click \
  cython \
  dill \
  h5py \
  ipython \
  jupyter \
  jupyterlab \
  natsort \
  numpy \
  pandas \
  pillow \
  pycodestyle \
  pylint \
  requests \
  scipy \
  six \
  sklearn \
  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp36-cp36m-linux_x86_64.whl \
  # tensorflow-gpu \
  tqdm

RUN echo "installing pysaliency"

RUN pip3 install git+https://github.com/matthias-k/pysaliency.git

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

