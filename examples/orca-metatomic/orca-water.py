"""
Calling a metatomic model from ORCA
====================================

:Authors: Eric Boittier `@EricBoittier <https://github.com/EricBoittier>`_

ORCA can pass each geometry step to another program. It writes an
``extinp`` file and reads an ``engrad`` file: energy in Hartree, gradient
in Hartree/Bohr. ``prepare.py`` is that program for a metatomic model.
The file is ``data/prepare.py`` in ``orca-water.zip``.

This page does not launch ORCA. It writes the files ORCA would write for
water, then calls the same function. The model is PET-MAD-XS, exported
the first time ``prepare.py`` is imported.
"""

# %%
# One water evaluation
# --------------------
#
# The extinp file is five fields: the XYZ name, the charge, the
# multiplicity, ``NCores``, and ``1`` if a gradient is required.
# ``NCores`` is also how many threads the wrapper gives PyTorch.

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "data"))
from prepare import (  # noqa: E402
    MetatomicOrcaSettings,
    model_path,
    run_orca_job,
)

water = Path("water.xyz")
water.write_text(
    "3\n"
    "water\n"
    "O 0.000000 0.000000 0.117300\n"
    "H 0.000000 0.757200 -0.469200\n"
    "H 0.000000 -0.757200 -0.469200\n"
)
extinp = Path("water_EXT.extinp.tmp")
extinp.write_text(f"{water.name}\n0\n1\n1\n1\n")
engrad = run_orca_job(
    extinp,
    MetatomicOrcaSettings(model=model_path, device="cpu"),
)
print(engrad.read_text())

# %%
# The ORCA input
# --------------
#
# ORCA 6 selects the program with ``ProgExt``. The path has to be
# absolute, because ORCA's working directory is not always the directory
# you started from.
#
# .. code-block:: text
#
#    ! ExtOpt Opt
#    %method
#      ProgExt "python /path/to/data/prepare.py"
#      Ext_Params "--model /path/to/pet-mad-xs.pt --device cpu"
#    end
#
#    * xyzfile 0 1 water.xyz
#
# Each step starts a new Python process and loads the model again.
# The file protocol is the one in the
# `ORCA external optimizer tutorial <https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html>`_
# and in
# `orca-external-tools <https://github.com/faccts/orca-external-tools#interface>`_.
# `This request <https://github.com/metatensor/metatomic/issues/228>`_
# is what the wrapper is for.
