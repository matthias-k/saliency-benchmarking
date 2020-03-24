#FROM ubuntu:latest
#FROM jupyter/base-notebook
FROM jupyter/scipy-notebook

USER root

### install ubuntu packages
RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install \
    -yq -qq --no-install-recommends \
    git \
    unzip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

### install NVIDIA libraries and development tools
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -rf /var/lib/apt/lists/*
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
ENV NCCL_VERSION 2.3.5
ENV CUDNN_VERSION 7.3.1.20
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-0=410.48-1 \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-2+cuda10.0 \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-2+cuda10.0 \
        libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
        libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    ln -s cuda-10.0 /usr/local/cuda && \
    apt-mark hold libnccl2 libcudnn7 && \
    rm -rf /var/lib/apt/lists/*
ENV PATH /usr/local/cuda/bin:${PATH}
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385"
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

### install tensorflow from source
#COPY install_bazel.sh /tmp
#RUN bash /tmp/install_bazel.sh \
# && fix-permissions $CONDA_DIR \
# && fix-permissions /home/$NB_USER

#COPY install_tensorflow.sh /tmp
#RUN bash /tmp/install_tensorflow.sh \
# && fix-permissions $CONDA_DIR \
# && fix-permissions /home/$NB_USER


#ADD lib /tmp/lib
#RUN cp /tmp/lib/* /usr/lib/



### install conda packages
RUN conda install \
    --quiet --yes \
    boltons \
    click \
    cython \
    dill \
    h5py \
    Jinja2 \
    markdown \
    natsort \
    pillow \
    pycodestyle \
    pylint \
    pyyaml \
    requests \
    schema \
    scipy \
    scikit-learn \
    tqdm \
 && conda clean -tipsy \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN pip install --no-cache \
  executor \
  rarfile \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN pip install --no-cache https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.14.0-cp37-cp37m-linux_x86_64.whl \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

# We have to use `pip` (instead of `pip2`) to upgrade pip, since these two are
# different and only `pip` works. After the upgrade, we can start using `pip2`.
#RUN pip3 install setuptools
#RUN pip3 install wheel

#RUN pip3 install \
#  cython \
#  dill \
#  h5py \
#  ipython \
#  jupyter \
#  jupyterlab \
#  natsort \
#  numpy \
#  pandas \
#  pillow \
#  pycodestyle \
#  pylint \
#  pyyaml \
#  requests \
#  schema \
#  scipy \
#  six \
#  sklearn \
#  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.11.0-cp36-cp36m-linux_x86_64.whl \
  # tensorflow-gpu \
#  tqdm



### install pip packages
#RUN pip install --no-cache-dir --upgrade pip \
# && pip --no-cache-dir install \
#    smiler-tools \
# && fix-permissions $CONDA_DIR \
# && fix-permissions /home/$NB_USER

### install smiler

RUN cd /opt \
 && git clone https://github.com/TsotsosLab/SMILER.git

ENV PATH=$PATH:/opt/SMILER



#### MATLAB

#COPY extern /usr/local/MATLAB/R2018b/extern

#RUN cd /usr/local/MATLAB/R2018b/extern/engines/python \
#  && sed -i "s/_supported_versions =.*/_supported_versions = ['2.7', '3.5', '3.6', '3.7']/" setup.py \
#  && python setup.py install

#ENV PATH=$PATH:/usr/local/MATLAB/R2018b/bin

#COPY install-matlab-bridge.sh /usr/local/bin/start-notebook.d/install-matlab-bridge.sh
#RUN chmod +x /usr/local/bin/start-notebook.d/install-matlab-bridge.sh

RUN apt-get update -qq \
 && DEBIAN_FRONTEND=noninteractive apt-get install \
    -yq -qq --no-install-recommends \
    ssh \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ssh /home/$NB_USER/.ssh
RUN chown -R $NB_USER:users /home/$NB_USER/.ssh && fix-permissions /home/$NB_USER && chmod 600 /home/$NB_USER/.ssh/id_rsa

COPY matlab /bin/matlab
RUN chmod +x /bin/matlab

RUN pip install --no-cache \
  staticjinja \
 && fix-permissions $CONDA_DIR \
 && fix-permissions /home/$NB_USER

RUN echo "installing pysaliency >= 2020-03-20 1"
RUN pip install git+https://github.com/matthias-k/pysaliency.git

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
 
USER $NB_UID
