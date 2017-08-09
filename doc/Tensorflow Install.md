# Installation of Tensorflow for Python 3.5/3.6

## Tensorflow for CPU

Just install the python module `tensorflow`. If you want it to be faster, you can compile TensorFlow yourself from source using [this tutorial](https://www.tensorflow.org/install/install\_sources) and then install it as a python module.

## Tensorflow for GPU (cuda v8)

Follow the following [guide](https://github.com/i3/i3status/raw/master/contrib/check_mail.py):

1. Install the following dependencies (ubuntu packages, if you have another system find their counterparts):
    openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev

2. Switch from noveau drivers to the proprietary nvidia drivers and reboot your computer. In Ubuntu this can be done in `Settings > Proprietary Drivers`
3. Download the nvidia toolkit and cuDNN (you must register an account in the nvidia website)
4. Install the python module `tensorflow-gpu` (no need to compile it yourself)
