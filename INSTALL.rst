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

Run ``redis-server`` and heed these warnings if they are present::

    # WARNING overcommit_memory is set to 0! Background save may fail under low memory condition. To fix this issue add 'vm.overcommit_memory = 1' to /etc/sysctl.conf and then reboot or run the command 'sysctl vm.overcommit_memory=1' for this to take effect.
    # WARNING you have Transparent Huge Pages (THP) support enabled in your kernel. This will create latency and memory usage issues with Redis. To fix this issue run the command 'echo never > /sys/kernel/mm/transparent_hugepage/enabled' as root, and add it to your /etc/rc.local in order to retain the setting after a reboot. Redis must be restarted after THP is disabled.

Running the Router
------------------

To run the router::

    cd router
    crossbar start

Run Crossbar Router under Pypy (optional)
-----------------------------------------

Download and unzip portable pypy::

    wget https://bitbucket.org/squeaky/portable-pypy/downloads/pypy3.5-5.10.1-linux_x86_64-portable.tar.bz2
    tar xvjf pypy3.5-5.10.1-linux_x86_64-portable.tar.bz2

Install crossbar under pypy::

    pypy3-5.5-linux_x86_64-portable/bin/pypy -m ensurepip
    pypy3-5.5-linux_x86_64-portable/bin/pypy -m pip install -U pip
    pypy3-5.5-linux_x86_64-portable/bin/pip install --ignore-installed -e .

Run the router under pypy::

    cd router
    ./run.sh

Install Miniconda
-----------------

Download and install::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

Install Datacube
----------------

::

    conda env create -f environment.yml
    conda activate datacube

or

::

    conda env create -n ENV_NAME -f environment.yml
    conda activate ENV_NAME

to use a custom environment name.

Running the Server
------------------

A crossbar `config.json` must be created, as only a template (`.crossbar/config.json.j2`) is checked in. Rendering this template with `DATACUBE_ENV=production` (or `development`, `test`, `demo`) will produce a `.crossbar/config.json` that can be used or modified as needed.

Run all services::

    crossbar start

Wait for the various "ready" messages in the log output.

Running the Demo
----------------

The demo config contains its own router. Generate the config using::

    pip install -e.[dev]
    yasha --DATACUBE_ENV=demo .crossbar/config.json.j2

A Node.js installation is needed in order to install npm packages and to build the client javascript for the demos::

    yum install nodejs

Now an npm install for each of the services is needed::

    ./nested-npm-install.sh

Generate datasets for the demos::

    python recache.py datasets-demo.json

Then run the demo::

    crossbar start

Wait for the "Server Ready." message, and then point your browser to http://localhost:8080/demo and click on the links.

Run tests
---------

Follow normal install steps, then do::

    pip install -r requirements.txt
    make test
