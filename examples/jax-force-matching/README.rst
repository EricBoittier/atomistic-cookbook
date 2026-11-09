Force matching a water model with JAX
=====================================

This example fits a flexible, pair-only water model to the forces of hybrid DFT
simulations of liquid water with a metatomic ``JaxModel``, using neighbor lists
computed inside JAX with ``vesin-jax``, and then runs molecular dynamics with the
fitted model.
