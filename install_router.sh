wget https://bitbucket.org/squeaky/portable-pypy/downloads/pypy-5.8-linux_x86_64-portable.tar.bz2
sha256sum -c pypy-5.8-linux_x86_64-portable.tar.bz2.sha256 || exit 1
tar xvjf pypy-5.8-linux_x86_64-portable.tar.bz2
pypy-5.8-linux_x86_64-portable/bin/pypy -m ensurepip
pypy-5.8-linux_x86_64-portable/bin/pypy -m pip install -U pip
pypy-5.8-linux_x86_64-portable/bin/pip install crossbar
pypy-5.8-linux_x86_64-portable/bin/crossbar version
