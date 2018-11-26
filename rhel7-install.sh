#!/bin/bash

# install datacube (tested on RHEL 7.6)
# make sure DATACUBE_ENV is set according to something expected by .crossbar/config.json.j2

# echo commands while running
set -x

# need EPEL for redis; this is the recommended way to install
sudo yum install -y wget
wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm -P /tmp
sudo yum install -y /tmp/epel-release-latest-7.noarch.rpm

# datacube specific dependencies
sudo yum install -y redis gcc openssl-devel libffi-devel python-devel

# install and activate miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/ec2-user/miniconda3
source /home/ec2-user/miniconda3/etc/profile.d/conda.sh

# extract datacube artifact provided by file provisioner
sudo yum install -y bzip2
mkdir /home/ec2-user/datacube
mv /tmp/datacube.tgz /home/ec2-user/datacube/
tar xvf /home/ec2-user/datacube/datacube.tgz -C /home/ec2-user/datacube

# (re-)create and activate datacube conda environment
conda env remove --name datacube -y || true
conda env create --file /home/ec2-user/datacube/environment.yml --name datacube
conda activate datacube

# installing numcodecs through conda-forge is a workaround needed for compilation issues on latest RHEL 7.6
conda install -c conda-forge numcodecs
conda deactivate

# extract xrelease artifact provided by file provisioner
mkdir /home/ec2-user/xrelease
mv /tmp/xrelease-master.zip /home/ec2-user/xrelease/
sudo yum install -y unzip
unzip /home/ec2-user/xrelease/xrelease-master.zip -d /home/ec2-user/xrelease/

# get ready to render some jinja templates
conda activate
pip install yasha

# render crossbar config template to make sure we have one for the current DATACUBE_ENV
yasha --DATACUBE_ENV=$DATACUBE_ENV /home/ec2-user/datacube/.crossbar/config.json.j2 -o /home/ec2-user/datacube/.crossbar/config-aws_test.json

# render systemd config template from xrelease
yasha --MINICONDA_DEPLOY_DIR=/home/ec2-user --DATACUBE_DEPLOY_DIR=/home/ec2-user/datacube --DATACUBE_ENV=$DATACUBE_ENV /home/ec2-user/xrelease/templates/_etc_systemd_system_crossbar.service.j2
conda deactivate
# systemd datacube config needs this symlink
ln -s /home/ec2-user/miniconda3 /home/ec2-user/miniconda
# install the config and enable the unit
sudo cp /home/ec2-user/xrelease/templates/_etc_systemd_system_crossbar.service /etc/systemd/system/crossbar.service
sudo systemctl enable crossbar.service

# connect to datacube EFS volume and ensure writable
sudo yum install -y nfs-utils
mkdir /home/ec2-user/efs
sudo echo 'fs-c7438b6f.efs.us-west-2.amazonaws.com:/ /home/ec2-user/efs nfs4 nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport 0 0' | sudo tee -a /etc/fstab > /dev/null
sudo mount /home/ec2-user/efs
sudo chmod go+rw efs

# create symlinks in datacube install directory pointing to efs data directories
ln -s /home/ec2-user/efs/bob_data /home/ec2-user/datacube/bob_data
ln -s /home/ec2-user/efs/conn_data /home/ec2-user/datacube/conn_data
ln -s /home/ec2-user/efs/mouse_ccf_data /home/ec2-user/datacube/mouse_ccf_data
ln -s /home/ec2-user/efs/human_mtg_data /home/ec2-user/datacube/human_mtg_data
ln -s /home/ec2-user/efs/structure_meshes /home/ec2-user/datacube/structure_meshes
ln -s /home/ec2-user/efs/conn_projection_maps /home/ec2-user/datacube/conn_projection_maps

# selinux was preventing the redis process (spawned as a child of crossbar
# under systemd) from reading redis.conf. disable selinux except for logging of
# violations.
sudo sed -i 's/^SELINUX=.*$/SELINUX=permissive/g' /etc/selinux/config
