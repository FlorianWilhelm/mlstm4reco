==========
mlstm4reco
==========


Benchmark multiplicative LSTM vs. ordinary LSTM


Description
===========

Create a conda environment with::

    conda env create -f environment-abstract.yml

or use::

    conda env create -f environment-concrete.yml

to perfectly replicate the environment.
Then activate the environment with::

    source activate mlstm4reco

and install it with::

    python setup.py develop

Then change into the ``experiments`` directory and run:

   ./run.py 10m -m mlstm

to run the ``mlstm`` model on the Movielens 10m dataset. Check out
``./run.py -h`` for more help.

Note
====

This project has been set up using PyScaffold 3.0.2. For details and usage
information on PyScaffold see http://pyscaffold.org/.
