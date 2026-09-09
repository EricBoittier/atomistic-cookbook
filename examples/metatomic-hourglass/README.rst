The metatomic hourglass design
==============================

This example takes three machine-learning models with very different origins,
PET-MAD trained with metatrain, a MACE foundation model, and a branch of the
multitask DPA-3.1-3M model from deepmd-kit, and exports them all to the
common metatomic format. It then uses them to run the same molecular dynamics
simulation with five different engines: ASE, LAMMPS, GROMACS, i-PI, and
TorchSim.

``lammps-metatomic`` and ``gromacs-metatomic`` currently pin incompatible
``libtorch`` versions, so they cannot live in the same conda prefix as each
other (or as the Python engines). Create three prefixes, then run from the
Python one; the script locates ``lmp`` / ``gmx`` in the sibling prefixes:

.. code-block:: bash

   bash create-engine-envs.sh
   conda activate ./.hourglass-env
   python metatomic-hourglass.py
