Development
===========

Dependencies
------------

When adding python dependencies, the package should be added (with or without version) to ``requirements-{env}.txt`` where ``{env}`` is the appropriate ``prod``, ``test``, or ``dev``. Note that ``test`` includes ``prod`` and ``dev`` includes ``prod`` and ``test``.

If ``requirements-prod.txt`` is modified, ``requirements.txt`` should be updated by way of a pip install and pip freeze within a fresh conda env, and checked in.

.. code-block::

    source deactivate datacube
    conda env remove -n datacube
    conda create -n datacube python=3
    source activate datacube
    pip install --ignore-installed -r requirements-prod.txt
    pip freeze > requirements.txt


Subtree
-------

The development environment (e.g. building and running the demos) expects datacube-js to appear under clients/ and datacubesdk to appear under sdk/. `git subtree`_ can be used for this.

.. _git subtree: https://www.atlassian.com/blog/git/alternatives-to-git-submodule-git-subtree


Adding
^^^^^^

.. code-block::

    git remote add -f datacube-js http://chrisba@stash.corp.alleninstitute.org/scm/~chrisba/datacube-js.git
    git subtree push --prefix=clients/ datacube-js master

    git remote add -f datacubesdk http://chrisba@stash.corp.alleninstitute.org/scm/~chrisba/datacubesdk.git
    git subtree push --prefix=sdk/ datacubesdk master


Pulling
^^^^^^^

.. code-block::

    git fetch datacube-js master
    git subtree pull --prefix=clients/ datacube-js master --squash
    
    git fetch datacubesdk master
    git subtree pull --prefix=sdk/ datacubesdk master --squash


Pushing
^^^^^^^

.. code-block::

    git subtree push --prefix=clients/ datacube-js master

    git subtree push --prefix=sdk/ datacubesdk master
