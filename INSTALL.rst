Installation instructions
=========================

System dependencies
-------------------

::
   
    yum install gcc openssl-devel libffi-devel python-devel

Install redis
-------------

Datacube needs redis>=3.2.

On CentOS 7::

    yum install redis

On CentOS 6::

    wget http://download.redis.io/releases/redis-3.2.9.tar.gz
    tar xzf redis-3.2.9.tar.gz
    pushd redis-3.2.9
    make
    make test
    popd
    export PATH=$PATH:$PWD/redis-3.2.9/src/

Run :code:`redis-server` and heed these warnings if they are present::

    # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
    # WARNING you have Transparent Huge Pages (THP) support enabled in your kernel. This will create latency and memory usage issues with Redis. To fix this issue run the command 'echo never > /sys/kernel/mm/transparent_hugepage/enabled' as root, and add it to your /etc/rc.local in order to retain the setting after a reboot. Redis must be restarted after THP is disabled.

Run Crossbar under Pypy (optional)
----------------------------------

Download and unzip portable pypy::

    wget https://bitbucket.org/squeaky/portable-pypy/downloads/pypy-5.8-1-linux_x86_64-portable.tar.bz2
    tar xvjf pypy-5.8-1-linux_x86_64-portable.tar.bz2

Install crossbar under pypy::

    pypy-5.8-linux_x86_64-portable/bin/pypy -m ensurepip
    pypy-5.8-linux_x86_64-portable/bin/pypy -m pip install -U pip
    pypy-5.8-linux_x86_64-portable/bin/pip install crossbar
    pypy-5.8-linux_x86_64-portable/bin/crossbar version

Install Miniconda
-----------------

Download and install::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

Install Datacube
----------------

Create a conda environment and install required python packages.

The :code:`--ignore-installed` option is a workaround for https://github.com/ContinuumIO/anaconda-issues/issues/542::

    conda create --name datacube python=3
    source activate datacube
    pip install --ignore-installed -r requirements.txt

Verify Install
--------------

Verify the environment and install by running the demo::

    crossbar start --config config-demo.json

Wait for the "Server Ready." message, and then point your browser to http://localhost:8082/ and click on the links.

Run tests
---------

::

    make test
