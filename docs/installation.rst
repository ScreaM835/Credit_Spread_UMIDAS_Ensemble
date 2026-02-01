Installation
============

Install from source (editable):

.. code-block:: bash

   pip install -e .

Optional extras:

.. code-block:: bash

   # For LightGBM / XGBoost
   pip install -e ".[boosting]"

   # For documentation builds
   pip install -e ".[docs]"

Command-line entry point
------------------------

The package exposes a console script:

.. code-block:: bash

   umidas-ensemble --help

Data requirements
-----------------

The panel CSV must contain at least:

- ``cusip``: bond identifier
- ``date``: timestamp
- ``cs``: credit spread level (decimal units)

An optional Excel variable dictionary can be supplied to infer which predictor
columns are monthly vs quarterly. If not provided, a JSON cache of variable lists
must be supplied.
