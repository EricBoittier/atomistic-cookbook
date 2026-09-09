#!/usr/bin/env bash
# Create the recipe Python env plus sibling LAMMPS / GROMACS prefixes.
# Those two engines pin incompatible libtorch versions, so they cannot
# share one conda environment. The Python script finds the binaries in
# the sibling prefixes (or HOURGLASS_{LAMMPS,GROMACS}_PREFIX).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

create() {
  local prefix="$1"
  local file="$2"
  if [ -d "$prefix/conda-meta" ]; then
    echo "already exists: $prefix"
    return 0
  fi
  echo "==> conda env create --prefix $prefix --file $file"
  conda env create --prefix "$prefix" --file "$file"
}

create "$HERE/.hourglass-env" "$HERE/environment.yml"
create "$HERE/.hourglass-env-lammps" "$HERE/environment-lammps.yml"
create "$HERE/.hourglass-env-gromacs" "$HERE/environment-gromacs.yml"

echo
echo "Activate the Python env, then run the recipe:"
echo "  conda activate $HERE/.hourglass-env"
echo "  python $HERE/metatomic-hourglass.py"
