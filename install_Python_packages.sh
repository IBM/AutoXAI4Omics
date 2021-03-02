#! /usr/bin/env bash

pip install autogluon==0.0.15
pip install auto-sklearn==0.12.0
pip uninstall -y numpy
pip install numpy
pip install emcee
pip install pyDOE
# pip install absl-py==0.10.0
# pip install ConfigSpace==0.4.16
