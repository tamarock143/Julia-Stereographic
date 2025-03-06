# Adaptive Stereographic MCMC algorithms 

Code attached to the [Adaptive Stereographic MCMC]([url](https://arxiv.org/abs/2408.11780)) project. Please cite this paper if you use this code or found it helpful.
Example sample paths can be found at:
- SRW: https://youtu.be/1p3cE2QfVFE
- SSS: https://youtu.be/EkEuZtfvXW8
- SBPS: https://youtu.be/8Hi2jW-2eBw

## Preliminary Code

- The file 'Stereographic Projection.jl' contains a large number of preliminary functions, such as the stereographic projection map, density and gradient calculations, and many smaller functions required for the stereographic bouncy particle sampler.
- The file 'Optimisation.jl' contains contains several optimisation and root finding routines necessary for other algorithms.

## Stereographic Random Walk

- The file 'SRW.jl' contains the function for simulating stereographic random walk sample paths.
- The file 'Adaptive SRW.jl' uses this to simulating adaptive SRW sample paths.

## Stereographic Slice Sampler

- The file 'SSS.jl' contains the function for simulating stereographic slice sampler sample paths.
- The file 'Adaptive SSS.jl' uses this to simulating adaptive SSS sample paths.

## Stereographic Bouncy Particle Sampler

- The file 'SBPS.jl' contains several functions for simulating stereographic random walk sample paths. We recommend using SBPSGeom, and this is the function used in our tests.
- The file 'Adaptive SRW.jl' uses this to simulating adaptive SRW sample paths. We recommend using SBPSAdaptiveGeom, and this is the function used in our tests.

## (Euclidean) HMC

- The file 'Hamiltonian MC.jl' contains our implementation of the vanilla Hamiltonian Monte Carlo algorithm, and is used to benchmark our methods against off-theshelf algorithms.

## Works in Progress

- The file 'SHMC.jl' is an attempt at creating a stereographic version of HMC. This is a work in progress.
- The file 'JuliaTests.jl' is the active test suite, and is therefore subject to regular changes.
