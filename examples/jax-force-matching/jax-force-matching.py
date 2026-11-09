"""
Force matching a water model with JAX
=====================================

:Authors: Eric Boittier `@EricBoittier <https://github.com/EricBoittier>`_

This recipe fits the parameters of a classical, flexible water model to the forces of
hybrid DFT (revPBE0-D3) simulations of liquid water, and then runs molecular dynamics
with the fitted model. Everything is written with `JAX <https://docs.jax.dev>`_: the
model is a `metatomic <https://docs.metatensor.org/metatomic>`_ ``JaxModel``, the
neighbor lists are computed by ``vesin-jax`` inside :py:func:`jax.jit`, and the fit
differentiates the forces with respect to the parameters of the model.

Force matching needs neighbor lists that are both computed inside JAX transformations
and differentiable, which is what ``metatomic.JaxPairs`` provides:

- the frames have different numbers of pairs, and are evaluated together. Pairs given
  as ``metatensor.Labels`` are static data, so each frame would have a different
  structure and need its own compilation, while ``JaxPairs`` stores the pairs as arrays
  of fixed capacity;
- the loss is computed on the forces, which are derivatives of the energy with respect
  to the positions. Fitting differentiates them again with respect to the parameters,
  which goes through the second derivatives of the pair vectors;
- the molecular dynamics recomputes the neighbor list at every step, inside
  :py:func:`jax.lax.scan`.
"""

# sphinx_gallery_thumbnail_number = 1

# %%
from functools import partial
from pathlib import Path

import jax


# vesin-jax must be imported before metatensor (see the note at the end)
from vesin_jax import NeighborList

import ase.io
import ase.units
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from atomistic_cookbook_utils import download_with_retry
from jax.scipy.special import erfc
from metatensor import Labels, TensorBlock, TensorMap
from metatomic import (
    JaxModel,
    JaxPairs,
    JaxSystem,
    ModelCapabilities,
    ModelMetadata,
    PairListOptions,
    Quantity,
    unit_conversion_factor,
)


# the model and the fit use double precision
jax.config.update("jax_enable_x64", True)

# %%
#
# The reference data
# ------------------
#
# We use the 1593 frames of 64 water molecules computed with revPBE0-D3 by `Cheng et
# al., PNAS 116, 1110 (2019) <https://doi.org/10.1073/pnas.1815117116>`_. The frames
# cover a range of densities and temperatures, and some of them contain broken or
# transferred protons, which a classical water model can not describe: we only keep
# the frames where every hydrogen belongs to a molecule with two hydrogens.
#
# The model below finds the molecules from the order of the atoms, so we also reorder
# the atoms as O, H, H for each molecule, and make the molecules whole (the positions
# of the hydrogens are moved to the periodic image closest to their oxygen). The file
# uses atomic units: positions in bohr, and forces in Hartree/bohr.

URL = (
    "https://raw.githubusercontent.com/BingqingCheng/ab-initio-thermodynamics-of-water/"
    "master/training-set/dataset_1593.xyz"
)
N_MOLECULES = 64


def load_frames(path):
    """frames with intact molecules, as O, H, H with whole molecules"""
    positions, cells, forces = [], [], []
    for frame in ase.io.read(path, ":"):
        oxygens = np.flatnonzero(frame.numbers == 8)
        hydrogens = np.flatnonzero(frame.numbers == 1)
        box = frame.cell.lengths()
        # O-H vectors for all O and H, with the minimum image convention
        oh = frame.positions[hydrogens] - frame.positions[oxygens][:, None]
        oh -= box * np.round(oh / box)
        owner = np.linalg.norm(oh, axis=-1).argmin(0)
        if np.bincount(owner, minlength=N_MOLECULES).tolist() != [2] * N_MOLECULES:
            continue

        mine = [np.flatnonzero(owner == o) for o in range(N_MOLECULES)]
        order = np.concatenate(
            [[o, *hydrogens[h]] for o, h in zip(oxygens, mine, strict=True)]
        )
        whole = frame.positions[order].reshape(N_MOLECULES, 3, 3)
        whole[:, 1:] = (
            whole[:, :1] + oh[np.arange(N_MOLECULES)[:, None], np.stack(mine)]
        )

        positions.append(whole.reshape(-1, 3))
        cells.append(frame.cell[:])
        forces.append(frame.arrays["force"][order])
    return np.stack(positions), np.stack(cells), np.stack(forces)


path = download_with_retry(URL, "tmp/water-cheng-2019.xyz")
positions, cells, reference = load_frames(path)
TYPES = np.tile([8, 1, 1], N_MOLECULES)

print(f"{len(positions)} frames with intact molecules")
print(f"box sizes from {cells[:, 0, 0].min():.1f} to {cells[:, 0, 0].max():.1f} bohr")

# %%
#
# A flexible water model made of pair terms
# -----------------------------------------
#
# The model is a variant of `SPC/Fw <https://doi.org/10.1063/1.2136877>`_ where every
# term depends on a single distance, so that it can be computed from a pair list:
#
# - inside a molecule, a Morse potential for the O-H bonds (the bonds of water are
#   strongly anharmonic) and a harmonic H-H term instead of the H-O-H angle;
# - between molecules, Lennard-Jones between oxygens, and Coulomb interactions between
#   fixed charges :math:`q_\mathrm{O} = -2 q_\mathrm{H}`. The Coulomb interactions use
#   the `damped shifted force <https://doi.org/10.1063/1.2206581>`_ approximation, with
#   an 8 Å cutoff, instead of an Ewald sum.
#
# The 8 parameters :math:`\theta` start from SPC/Fw, with the Morse depth and width of
# `q-TIP4P/F <http://dx.doi.org/10.1063/1.3167790>`_. The positive parameters are
# stored as logarithms, which keeps them positive during the fit.

CUTOFF = 8.0  # Å
ALPHA = 0.2  # Å^-1, damping of the Coulomb interactions
COULOMB = unit_conversion_factor("Hartree", "eV") * unit_conversion_factor(
    "bohr", "angstrom"
)  # e² / (4 π ε0) in eV Å

KCAL = unit_conversion_factor("kcal/mol", "eV")
EPSILON, SIGMA = 0.1554253 * KCAL, 3.165492
NAMES = ["D_OH", "a_OH", "r_OH", "k_HH", "r_HH", "q_H", "C12_OO", "C6_OO"]
UNITS = ["eV", "1/Å", "Å", "eV/Å²", "Å", "e", "eV Å¹²", "eV Å⁶"]
INITIAL = np.array(
    [116.09 * KCAL, 2.287, 1.012, 245.0 * KCAL, 1.690, 0.41]
    + [4 * EPSILON * SIGMA**12, 4 * EPSILON * SIGMA**6]
)
LOG = np.array([True, True, False, True, False, False, True, True])


def to_theta(parameters):
    return np.where(LOG, np.log(parameters), parameters)


def from_theta(theta):
    return jnp.where(LOG, jnp.exp(theta), theta)


def pair_energy(theta, oxygen_i, oxygen_j, r, intra):
    """energy of pairs of atoms at distance ``r`` (in Å), which are oxygens or
    hydrogens, in the same molecule or not"""
    d_oh, a_oh, r_oh, k_hh, r_hh, q_h, c12, c6 = from_theta(theta)
    morse = d_oh * (1 - jnp.exp(-a_oh * (r - r_oh))) ** 2
    bonds = jnp.where(oxygen_i ^ oxygen_j, morse, k_hh * (r - r_hh) ** 2 / 2)

    q_i, q_j = (jnp.where(oxygen, -2 * q_h, q_h) for oxygen in (oxygen_i, oxygen_j))

    def dsf(r):
        return erfc(ALPHA * r) / r

    gaussian = 2 * ALPHA / np.sqrt(np.pi) * np.exp(-((ALPHA * CUTOFF) ** 2))
    slope = dsf(CUTOFF) / CUTOFF + gaussian / CUTOFF
    coulomb = COULOMB * q_i * q_j * (dsf(r) - dsf(CUTOFF) + slope * (r - CUTOFF))

    def lennard_jones(r):
        return c12 / r**12 - c6 / r**6

    lj = jnp.where(oxygen_i & oxygen_j, lennard_jones(r) - lennard_jones(CUTOFF), 0)
    return jnp.where(intra, bonds, coulomb + lj)


# %%
#
# The ``JaxModel`` gets the pairs and their vectors from ``JaxSystem.pair_arrays()``,
# which works for pair lists given as ``JaxPairs`` (computed inside JAX) and as
# ``Labels``. The vectors are computed from the positions and cell, so their
# derivatives are available to JAX. Pair lists of fixed capacity end with padding
# entries, which have zero vectors and are removed with ``pairs.mask``: the padding
# distances are set to one before computing :math:`1/r`, otherwise the gradients would
# contain ``NaN``.
#
# Two atoms belong to the same molecule when their indices divided by three are the
# same (the atoms come as O, H, H), and when the pair does not cross the cell (the
# molecules are whole).

OPTIONS = PairListOptions(cutoff=CUTOFF, full_list=False, strict=True)


class FlexibleWater(JaxModel):
    def __init__(self, theta=to_theta(INITIAL)):  # noqa: B008
        self.theta = theta

    def capabilities(self):
        return ModelCapabilities(
            atomic_types=[1, 8],
            interaction_range=CUTOFF,
            length_unit="angstrom",
            supported_devices=["cpu"],
            dtype="float64",
            outputs=[Quantity(name="energy", unit="eV", sample_kind="system")],
        )

    def metadata(self):
        return ModelMetadata(name="force-matched flexible water")

    def requested_pair_lists(self, outputs):
        return [OPTIONS]

    def energy(self, system):
        pairs, vectors = system.pair_arrays(OPTIONS)
        i, j = (jnp.where(pairs.mask, k, 0) for k in (pairs.first, pairs.second))
        r = jnp.sqrt(jnp.where(pairs.mask, jnp.sum(vectors**2, axis=1), 1.0))

        oxygen = jnp.asarray(system.types) == 8
        intra = (i // 3 == j // 3) & jnp.all(pairs.shifts == 0, axis=1)
        energies = pair_energy(self.theta, oxygen[i], oxygen[j], r, intra)
        return jnp.sum(jnp.where(pairs.mask, energies, 0.0))

    def forward(self, systems, selected_atoms, outputs):
        if selected_atoms is not None:
            raise ValueError("this model does not support selected_atoms")

        energies = jnp.stack([self.energy(system) for system in systems])
        block = TensorBlock(
            energies.reshape(-1, 1),
            Labels.range("system", len(systems)),
            [],
            Labels.range("energy", 1),
        )
        return [TensorMap(Labels.range("_", 1), [block])]


# %%
#
# Neighbor lists inside JAX
# -------------------------
#
# ``vesin_jax.NeighborList`` computes the pairs inside :py:func:`jax.jit`, as arrays
# with at most ``max_pairs`` entries, with the same layout as ``JaxPairs``. The systems
# are given in bohr like the data (``JaxModel.evaluate`` converts them to Å for the
# model), so the cutoff of this neighbor list is in bohr as well.
#
# We build the systems for 100 frames used for the fit and 200 frames used for testing
# with :py:func:`jax.vmap`: the result is a single ``JaxSystem`` whose arrays have one
# more dimension, for the frames.

BOHR = unit_conversion_factor("angstrom", "bohr")
CAPACITY = 2**15


def make_system(positions, cell, calculator, length_unit):
    pairs = JaxPairs(
        *calculator.compute(positions, cell, True, "ijS", max_pairs=CAPACITY)
    )
    system = JaxSystem(
        TYPES,
        positions,
        cell,
        [True, True, True],
        length_unit=length_unit,
        pairs={OPTIONS: pairs},
    )
    return system, pairs.n_pairs, pairs.overflow


calculator = NeighborList(cutoff=CUTOFF * BOHR, full_list=False)
make_systems = jax.jit(
    jax.vmap(partial(make_system, calculator=calculator, length_unit="bohr"))
)

rng = np.random.default_rng(0)
fit, test = np.split(rng.permutation(len(positions))[:300], [100])
data = {}
for name, frames in [("fit", fit), ("test", test)]:
    systems, n_pairs, overflow = make_systems(positions[frames], cells[frames])
    assert not overflow.any(), "increase CAPACITY"
    data[name] = (systems, jnp.asarray(reference[frames]))
    print(f"{name}: {len(frames)} frames, {n_pairs.min()} to {n_pairs.max()} pairs")

# %%
#
# Forces and the fit
# ------------------
#
# ``JaxModel.evaluate`` computes the forces as the gradient of the energy with respect
# to the positions, using JAX automatic differentiation. Requesting the energy in
# Hartree for systems in bohr gives forces in Hartree/bohr, the units of the data.

FORCES = Quantity(
    name="energy", unit="Hartree", sample_kind="system", gradients=["positions"]
)


def forces(theta, system):
    (result,) = FlexibleWater(theta).evaluate([system], [FORCES])
    return -result.block().gradient("positions").values.reshape(-1, 3)


# %%
#
# The parameters are fitted with the `Levenberg-Marquardt
# <https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_ algorithm,
# which minimizes the sum of the squared force errors :math:`\|r(\theta)\|^2` with
# steps :math:`\delta` solving :math:`(J^T J + \lambda\, \mathrm{diag}(J^T J))\,\delta =
# -J^T r`, where :math:`J = \partial r / \partial \theta` is the jacobian of the force
# errors. :math:`J` is a forward-mode derivative (:py:func:`jax.jacfwd`) of forces that
# are already reverse-mode derivatives of the energy, i.e. a second derivative going
# through the pair vectors.
#
# The jacobian of all the frames would need a lot of memory, so :py:func:`jax.lax.map`
# computes :math:`J^T J` and :math:`J^T r` ten frames at a time, and sums them.


def normal_equations(theta, system, reference):
    def residuals(theta):
        return (forces(theta, system) - reference).ravel()

    r, jacobian = residuals(theta), jax.jacfwd(residuals)(theta)
    return jacobian.T @ jacobian, jacobian.T @ r, r @ r


def squared_error(theta, system, reference):
    return jnp.sum((forces(theta, system) - reference) ** 2)


@partial(jax.jit, static_argnums=0)
def sum_over_frames(function, theta, systems, reference):
    results = jax.lax.map(
        lambda args: function(theta, *args), (systems, reference), batch_size=10
    )
    return jax.tree.map(lambda x: x.sum(axis=0), results)


EV_A = unit_conversion_factor("Hartree/bohr", "eV/angstrom")


def rmse(theta, name):
    error = sum_over_frames(squared_error, theta, *data[name])
    return EV_A * float(jnp.sqrt(error / data[name][1].size))


theta = initial = jnp.asarray(to_theta(INITIAL))
damping = 1e-3
for iteration in range(40):
    curvature, gradient, cost = sum_over_frames(normal_equations, theta, *data["fit"])
    while damping < 1e8:
        scaled = curvature + damping * jnp.diag(jnp.diag(curvature))
        step = jnp.linalg.solve(scaled, -gradient)
        if sum_over_frames(squared_error, theta + step, *data["fit"]) < cost:
            theta, damping = theta + step, damping / 3
            break
        damping *= 3
    if damping >= 1e8 or jnp.abs(step).max() < 1e-7:
        break

print(f"converged after {iteration + 1} iterations")
print(f"{'':>8} {'initial':>10} {'fitted':>10}")
for name, unit, before, after in zip(
    NAMES, UNITS, from_theta(initial), from_theta(theta), strict=True
):
    print(f"{name:>8} {before:10.4g} {after:10.4g} {unit}")

before, after = rmse(initial, "test"), rmse(theta, "test")
print(f"force RMSE on the test frames: {before:.3f} eV/Å initially, {after:.3f} fitted")
print(f"force RMSE on the frames used for the fit: {rmse(theta, 'fit'):.3f} eV/Å")

# %%
#
# The fit reduces the force errors about seven times. The H-H term (which stands for
# the angle) becomes much softer, the O-H bonds shorter, the charges larger, and the
# :math:`r^{-6}` attraction between oxygens goes to zero: with these charges, the forces
# do not need it. Most of the remaining error comes from what a pair potential can not
# describe, such as the polarization of the molecules and the coupling between the
# bonds and the angle.

predicted = jax.lax.map(lambda s: forces(theta, s), data["test"][0], batch_size=10)
hydrogen = np.arange(len(TYPES)) % 3 > 0

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
for element, atoms in [("O", ~hydrogen), ("H", hydrogen)]:
    dft = EV_A * np.asarray(data["test"][1][:, atoms]).ravel()
    fitted = EV_A * np.asarray(predicted[:, atoms]).ravel()
    error = np.sqrt(np.mean((fitted - dft) ** 2))
    ax.scatter(dft, fitted, s=1, alpha=0.2, label=f"{element}, RMSE {error:.2f} eV/Å")
ax.axline((0, 0), slope=1, c="k", lw=0.8)
ax.legend(markerscale=10)
ax.set(title="forces on the test frames (eV/Å)", xlabel="DFT", ylabel="fitted model")
plt.show()

# %%
#
# The intermolecular pair energies are stronger after the fit: the larger charges
# increase both the O-H attraction and the O-O and H-H repulsion, and the
# Lennard-Jones term between oxygens loses its attraction.

r = jnp.linspace(1.6, CUTOFF, 300)
fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
for (label, o_i, o_j), color in zip(
    [("O-O", True, True), ("O-H", True, False), ("H-H", False, False)],
    ["C0", "C1", "C2"],
    strict=True,
):
    ax.plot(r, pair_energy(theta, o_i, o_j, r, False), c=color, label=label)
    ax.plot(r, pair_energy(initial, o_i, o_j, r, False), ":", c=color)
ax.legend()
ax.set(
    title="intermolecular pair energies (dotted: initial)",
    xlabel="r (Å)",
    ylabel="energy (eV)",
    ylim=(-1.5, 1.5),
)
plt.show()

# %%
#
# Molecular dynamics with the fitted model
# ----------------------------------------
#
# The fitted model can run molecular dynamics directly in JAX. Each step of the
# velocity Verlet integrator recomputes the neighbor list with ``vesin-jax`` and the
# forces with ``JaxModel.evaluate``, and :py:func:`jax.lax.scan` compiles the whole
# trajectory as a single function. The number of pairs changes along the trajectory,
# without recompiling anything. This time the systems are in Å, and the energies in eV.

model = FlexibleWater(theta)
ENERGY = Quantity(
    name="energy", unit="eV", sample_kind="system", gradients=["positions"]
)
masses = jnp.asarray(np.tile([15.999, 1.008, 1.008], N_MOLECULES))[:, None]
cell = jnp.asarray(cells[test[0]]) / BOHR
md_calculator = NeighborList(cutoff=CUTOFF, full_list=False)
dt, n_steps = 0.25 * ase.units.fs, 4000


def energy_and_forces(positions):
    system, _, _ = make_system(positions, cell, md_calculator, "angstrom")
    (result,) = model.evaluate([system], [ENERGY])
    gradient = result.block().gradient("positions").values.reshape(-1, 3)
    return result.block().values[0, 0], -gradient


def verlet_step(state, _):
    positions, velocities, forces = state
    velocities = velocities + 0.5 * dt * forces / masses
    positions = positions + dt * velocities
    potential, forces = energy_and_forces(positions)
    velocities = velocities + 0.5 * dt * forces / masses
    total = potential + 0.5 * jnp.sum(masses * velocities**2)
    return (positions, velocities, forces), (total, positions[::3])


@jax.jit
def simulate(positions, velocities):
    _, forces = energy_and_forces(positions)
    state = (positions, velocities, forces)
    return jax.lax.scan(verlet_step, state, None, length=n_steps)[1]


# start from a test frame, with velocities at 300 K and no total momentum
start = jnp.asarray(positions[test[0]]) / BOHR
kT = 300 * ase.units.kB
velocities = rng.normal(size=(len(TYPES), 3)) * np.sqrt(kT / np.asarray(masses))
velocities -= np.sum(masses * velocities, axis=0) / np.sum(masses)

total, oxygens = simulate(start, jnp.asarray(velocities))

# %%
#
# The total energy is conserved, apart from a drop during the first tens of
# femtoseconds, while the strongly distorted DFT frame relaxes in the fitted model (a
# shorter time step makes this drop smaller).

time = np.arange(n_steps) * dt / ase.units.fs
fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.plot(time, 1e3 * (total - total[0]))
ax.set(
    title="total energy, 1 ps of NVE dynamics",
    xlabel="time (fs)",
    ylabel="energy change (meV)",
)
plt.show()

# %%
#
# Finally, we compare the oxygen-oxygen radial distribution function of the second
# half of the trajectory with the one of the DFT test frames. The fitted model puts
# the first neighbors slightly too close, and makes them too structured, but gets the
# second shell right. The DFT frames come from a range of densities and temperatures,
# so this is only a qualitative comparison.


def radial_distribution(frames, edges=np.linspace(2.0, 5.7, 75)):
    """O-O g(r) averaged over (positions, box lengths) frames, in Å"""
    histograms = []
    for x, box in frames:
        d = x[:, None] - x
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=-1)[np.triu_indices(len(x), 1)]
        density = len(x) * (len(x) - 1) / 2 / np.prod(box)
        histograms.append(np.histogram(r, edges)[0] / density)
    shells = 4 / 3 * np.pi * np.diff(edges**3)
    return (edges[1:] + edges[:-1]) / 2, np.mean(histograms, axis=0) / shells


md_frames = [(x, np.diag(cell)) for x in np.asarray(oxygens[n_steps // 2 :: 20])]
dft_frames = [
    (x[::3] / BOHR, np.diag(c) / BOHR)
    for x, c in zip(positions[test], cells[test], strict=True)
]

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.plot(*radial_distribution(md_frames), label="MD with the fitted model")
ax.plot(*radial_distribution(dft_frames), label="DFT test frames")
ax.legend()
ax.set(title="O-O radial distribution function", xlabel="r (Å)", ylabel="g(r)")
plt.show()

# %%
#
# Notes
# -----
#
# - Importing ``vesin_jax`` calls ``jax.ffi.pycapsule()``, which changes the argument
#   types of ``ctypes.pythonapi.PyCapsule_New`` for the whole process. Metatensor uses
#   the same function to share arrays with JAX, so ``vesin_jax`` has to be imported
#   first.
# - ``JaxPairs`` can also be built from ``metatensor.Labels`` with
#   ``JaxPairs.from_labels()``, for example to use neighbor lists computed outside of
#   JAX; ``JaxSystem.pair_arrays()`` returns the same arrays in both cases.
# - The same model also runs through ``metatomic.execute_model``, the path used by
#   simulation engines, where ``forward()`` runs eagerly on the neighbor lists given
#   by the engine.
