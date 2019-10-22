set -e

pip --no-cache-dir install 'keras_applications >= 1.0.5' 'keras_preprocessing >= 1.0.3' h5py

ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/libcuda.so.1

# clone tensorflow
cd /tmp/
git clone https://github.com/tensorflow/tensorflow.git
cd /tmp/tensorflow
git checkout v1.12.0

# configurate build
export PYTHON_BIN_PATH=/opt/conda/bin/python
export PYTHON_LIB_PATH=/opt/conda/lib/python3.7/site-packages
export CUDA_TOOLKIT_PATH=/usr/local/cuda
export CUDNN_INSTALL_PATH=/usr/local/cuda
export TF_CUDNN_VERSION=7
export TF_NCCL_VERSION=1.3
export TF_NEED_JEMALLOC=1
export TF_NEED_GCP=1
export TF_NEED_HDFS=1
export TF_NEED_NGRAPH=0
export TF_NEED_S3=0
export TF_NEED_AWS=0
export TF_NEED_KAFKA=0
export TF_ENABLE_XLA=1
export TF_NEED_GDR=0
export TF_NEED_VERBS=0
export TF_NEED_OPENCL_SYCL=0
export TF_CUDA_CLANG=0
export TF_NEED_CUDA=1
export TF_CUDA_VERSION=10.0
export CUDA_TOOLKIT_PATH=/usr/local/cuda-$TF_CUDA_VERSION
export TF_CUDA_COMPUTE_CAPABILITIES=3.7,5.2,5.3,6.0,6.1,7.0
export TF_NEED_TENSORRT=0
export TF_NEED_MPI=0
export CC_OPT_FLAGS="-march=native"
export TF_SET_ANDROID_WORKSPACE=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export NCCL_INSTALL_PATH=${CUDA_TOOLKIT_PATH}/targets/x86_64-linux
#export LD_PRELOAD=/usr/lib/libtcmalloc_minimal.so.4
./configure

# build

#echo import %workspace%/tools/bazel.rc >> .bazelrc
bazel build --jobs 24 --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" --config=opt --config=mkl //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ./

# install
pip --no-cache-dir install tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl

# clean up
bazel clean
cd /
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
rm /usr/local/cuda/lib64/libcuda.so.1
rm -rf /home/jovyan/.cache/bazel
