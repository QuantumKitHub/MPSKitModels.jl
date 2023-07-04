# MPSKitModels.jl

[![docs][docs-dev-img]][docs-dev-url] [![codecov][codecov-img]][codecov-url] ![CI][ci-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://maartenvd.github.io/MPSKitModels.jl/dev/

[codecov-img]: https://codecov.io/gh/maartenvd/MPSKitModels.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/maartenvd/MPSKitModels.jl

[ci-url]: https://github.com/maartenvd/MPSKitModels.jl/workflows/CI/badge.svg


```julia
import Pkg
Pkg.add("MPSKitModels")
```

MPSKitModels.jl provides operators, tools and utilities for [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl).
The main goal is to facilitate the definition and readability of Hamiltonians on (1+0)-dimensional quantum systems, as well as their quasi-one-dimensional extensions, such as cylinders, ladders, etc.
Additionally, some models from (2+0)-dimensional statistical mechanics are provided.

## Overview

The main building blocks of these Hamiltonians are local ``N``-body operators, which are provided in the form of an ``AbstractTensorMap{N,N}`` (see [TensorKit.jl](https://github.com/Jutho/TensorKit.jl)).
Several often-used operators are defined and exported within MPSKitModels.jl:

* spin operators (`S_x`, `S_y`, `S_z`, `S_plus`, `S_min`)
* spin exchange operators (`S_xx`, `S_yy`, `S_zz`, `S_exchange`, `S_plusmin`, `S_minplus`)
* bosonic operators (`a_plus`, `a_min`, `a_number`)
* fermionic operators (`c_plus`, `c_min`, `c_number`)
* fermionic spin operators (`e_plus`, `e_min`, `e_number`, `e_number_up`, `e_number_down`, `e_number_updown`)

These operators can then be combined to define Hamiltonians by way of the ``@mpoham`` macro.
This transforms ``{}`` expressions to denote the site-indices upon which the operators act, and generates site-indices for various geometries.
Some examples to showcase this:

```julia
using MPSKitModels, TensorKit

g = 1.0
H_ising = @mpoham sum(S_xx(){i, i + 1} + g * S_z(){i} for i in -Inf:Inf)

J = [1.0 -1.0]  # staggered couplings over unit cell of length 2
H_heisenberg_ = @mpoham sum(J[i] * S_exchange(SU2Irrep; spin=1){i, i + 1} for i in vertices(InfiniteChain(2)))

H_heisenberg_cylinder =
    @mpoham sum(J1 * S_exchange(; spin=1){i, j} for (i, j) in nearest_neighbours(InfiniteCylinder(3)))

J1 = 0.8
J2 = 0.2

H_J1J2 = @mpoham sum(J1 * S_exchange(){i, j} for (i, j) in nearest_neighbours(InfiniteCylinder(4))) +
    sum(J2 * S_exchange(){i,j} for (i, j) in next_nearest_neighbours(InfiniteCylinder(4)))
```

For convenience, several models have already been defined. The full list can be found in the [docs]([docs-dev-url]).
