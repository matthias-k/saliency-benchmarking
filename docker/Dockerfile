FROM ubuntu:18.04
MAINTAINER Matthias Kümmerer <matthias.kuemmerer@bethgelab.org>

RUN apt-get update -qq \
  && DEBIAN_FRONTEND=noninteractive apt-get install -yq -qq --no-install-recommends \
  gcc \
  git \
  libdpkg-perl \
  python3 \
  python3-dev \
  python3-pip

RUN pip3 install setuptools
RUN pip3 install \
  boltons \
  click \
  cython \
  dill \
  ipython \
  jupyter \
  jupyterlab \
  natsort \
  numpy \
  pillow \
  requests \
  scipy \
  six \
  tensorflow==1.5 \
  tqdm

RUN pip3 install git+https://github.com/matthias-k/pysaliency.git
