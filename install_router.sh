wget https://bitbucket.org/squeaky/portable-pypy/downloads/pypy3.5-5.10.1-linux_x86_64-portable.tar.bz2
sha256sum -c <(echo "b7c7b0e0905208ce8a8061b1a0ae136a702e5218d0d350cb5216ad5a7c20d12e  pypy3.5-5.10.1-linux_x86_64-portable.tar.bz2") || exit 1
tar xvjf pypy3.5-5.10.1-linux_x86_64-portable.tar.bz2
pypy3.5-5.10.1-linux_x86_64-portable/bin/pypy -m ensurepip
pypy3.5-5.10.1-linux_x86_64-portable/bin/pypy -m pip install -U pip
pypy3.5-5.10.1-linux_x86_64-portable/bin/pip install --ignore-installed -r requirements-global-prod.txt
