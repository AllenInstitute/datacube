Development
===========

Dependencies
------------

The requirements files are set up in an inheritance/include heirarchy to support per-service dependency specification along with common global dependencies, across prod, test, and dev environments. The inheritance structure looks like this:

::

    global-prod ------> svc{n}-prod
        |                   |
        v                   v
    global-test ------> svc{n}-test   ...
        |                   |
        v                   v
    global-dev -------> svc{n}-dev


Note that it is possible to introduce version conflicts either explicitly or implicitly given this structure. It may be advisable to build an environment across all services periodically in order to ensure conflicts don't arise (see requirements-txt_).


Service dependencies
^^^^^^^^^^^^^^^^^^^^

When adding python dependencies to a service, the package should be added (with or without version) to ``services/{servicename}/requirements-{env}.txt`` where ``{env}`` is the appropriate ``prod``, ``test``, or ``dev``. Note that ``test`` includes ``prod`` and ``dev`` includes ``prod`` and ``test``.

Global dependencies
^^^^^^^^^^^^^^^^^^^

In some cases a package may be needed as part of the base install across all services. In such cases the package can be added to the root-level ``requirements-{env}.txt``.

.. _requirements-txt:

requirements.txt
^^^^^^^^^^^^^^^^

``requirements.txt`` of locked package versions are maintained for purposes of deploying to production (not dev or test). These files can be materialized at different levels of the project. For example, if a single unified deploy environment is desired, ``requirements.txt`` can be maintained at the root-level, based off the union of the requirements of all the services. Alternatively, or in addition, individual ``requirements.txt`` files can be maintained within each service for leaner environments when deploying separately.

If any production dependencies are modified which would affect it, the corresponding ``requirements.txt`` should be updated by way of a ``pip install`` and ``pip freeze`` within a fresh conda env, and checked in. For example, this is how to update a root-level ``requirements.txt`` file from a root-level ``requirements-prod.txt`` file containing includes for each of the services' ``services/{servicename}/requirements-prod.txt`` files:

::

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
