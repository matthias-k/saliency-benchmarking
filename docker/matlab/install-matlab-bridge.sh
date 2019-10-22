#!/bin/bash

set -xe

DIRECTORY="/usr/local/MATLAB/R2018b/extern/engines/python"

sed "s/_supported_versions =.*/_supported_versions = ['2.7', '3.5', '3.6', '3.7']/" $DIRECTORY/setup.py > $DIRECTORY/setup_fixed.py

bash -c "cd $DIRECTORY && python setup_fixed.py install"

#rm setup_fixed.py
