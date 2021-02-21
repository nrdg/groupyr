############
Installation
############

*Groupyr* requires Python 3.6, 3.7, or 3.8 and depends on

    copt
    numpy
    scikit-learn
    scipy
    scikit-optimize
    tqdm

Installing the release version
------------------------------

The recommended way to install *groupyr* is from PyPI,

.. code-block:: console

    $ pip install groupyr

This will install *groupyr* and all of its dependencies.

Installing the development version
----------------------------------

The development version is less stable but may include new features.
You can install the development version using ``pip``:

.. code-block:: console

    pip install git+https://github.com/richford/groupyr.git 

Alternatively, you can clone the source code from the `github repository
<https://github.com/richford/groupyr>`_:

.. code-block:: console

    $ git clone git@github.com:richford/groupyr.git
    $ cd groupyr
    $ pip install .

If you would like to contribute to *groupyr*, see the `contributing guidelines
<contributing.html>`_.

Next, go to the `user guide <user_guide.html>`_ or see the `example gallery
<auto_examples/index.html>`_ for further information on how to use *groupyr*.

Dependencies
------------

Installing *groupyr* using either of the methods above will install all of
its dependencies: copt, numpy, scikit-learn, scipy, scikit-optimize, and
tqdm.