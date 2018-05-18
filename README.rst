Datacube
========
Serves up data from large multidimensional datasets using an `xarray`_-based query language. 

.. _xarray: https://xarray.pydata.org/en/stable/

Please see the `installation instructions`_ and the `development guide`_.

.. _installation instructions: INSTALL.rst
.. _development guide: DEVELOPMENT.rst

What's New (05/17/2018)
-----------------------
- Connectivity data is now stored with zarr. Connectivity data generation has an --internal flag for pulling data from Allen Institute-internal resources.
- removed deprecated services/datacube/ in favor of services/pandas