Development
===========

Dependencies
------------

Top-level requirements are tracked individually per-service and for the base application. Packages are pinned into a single production environment installable via conda.

environment-base.yml
^^^^^^^^^^^^^^^^^^^^

Top-level requirements (not pinned). This is a hand-created conda *environment.yml* which installs some common packages via conda, followed by pip requirements. The semantics of the ``pip`` section in conda *environment.yml* files closely resemble that of a pip *requirements.txt* file, including the ability to use ``-r`` and ``-e``. Install the latest packages into a fresh conda environment using:

::

    conda env create -f environment-base.yml

environment.yml
^^^^^^^^^^^^^^^

Pinned requirements for running any/all services and base crossbar application in production. Initially created via:

::

    conda env create -f environment-base.yml
    conda activate datacube
    conda env export > environment.yml

And used in production like:

::

    conda env create -f environment.yml

Finer-grained package management (adding, updating, removing) should be done through ``conda`` (see https://conda.io/docs/commands.html#conda-vs-pip-vs-virtualenv-commands), then re-exporting to ``environment.yml`` and checking in to source-control. ``environment-base.yml`` should be manually edited to reflect top-level dependency changes.

pip/setuptools
^^^^^^^^^^^^^^

Individual sets of requirements can be managed by *requirements.txt* or *setup.py* as long as they can be added to ``environment-base.yml`` as included requirements (``-r``) or editable package installs (``-e``).

.. note:: ``conda env export`` lacks a ``--exclude-editable`` option like ``pip freeze``. To keep unpublished packages from appearing in the pinned requirements, a workaround is to supply the ``--multi-version`` option to *setup.py develop*. This is done by way of *setup.cfg* files, since pip does not honor ``--install-option`` options on editable (``-e``) requirements in *requirements.txt* files, nor by extension does conda do so within the ``pip`` section of *environment.yml* files.


test dependencies
^^^^^^^^^^^^^^^^^

The root ``./requirements.txt`` should install test dependencies on top of the base environment (e.g. by referencing separate pip *requirements-test.txt* files with ``-r``, or via setuptools *extras* e.g. ``-e ./service/service-name[test]``); these dependencies aren't pinned, but could be.

constraints.txt
^^^^^^^^^^^^^^^

A pip *constraints.txt* file applying any common constraints needed to customize the pinned production environment (``./environment.yml``).

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
