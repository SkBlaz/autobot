BootStrap: docker
From: ubuntu:latest

%labels

%environment
export LC_ALL=C

%post
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y libhdf5-dev graphviz locales python3-dev python3-pip curl git
apt-get clean

pip3 install autoBOTLib
pip3 install sentence-transformers

unset DEBIAN_FRONTEND