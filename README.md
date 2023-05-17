# MPSKitModels.jl


[![docs][docs-dev-img]][docs-dev-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://maartenvd.github.io/MPSKitModels.jl/dev/

[![codecov](https://codecov.io/gh/maartenvd/MPSKitModels.jl/branch/master/graph/badge.svg?token=MDGR0SONEI)](https://codecov.io/gh/maartenvd/MPSKitModels.jl)


```julia
import Pkg
Pkg.add("MPSKitModels")
```

MPSKitModels.jl provides operators, tools and utilities for MPSKit.jl.
The main goal is to facilitate the definition and readability of hamiltonians on 1-dimensional systems, as well as quasi-1-dimensional systems defined on a cylinder or infinite strips.

## Overview

The main building blocks of these Hamiltonians are local ``N``-body operators, which should be provided in the form of an ``AbstractTensorMap{N,N}`` (see TensorKit.jl).
Several often-used operators are defined and exported within MPSKitModels.jl:

* spin operators (`sigma_x`, `sigma_y`, `sigma_z`, `sigma_plus`, `sigma_min`)
* spin exchange couplings (`sigma_xx`, `sigma_yy`, `sigma_zz`, `sigma_exchange`, `sigma_plusmin`, `sigma_minplus`)
* fermionic creation and annihilation operators (`cc`, `ccdag`, `cdagc`, `cdagcdag`, `number`)
* ...

These operators can then be combined to define Hamiltonians by way of the ``@mpoham`` macro.
This transforms ``{}`` expressions to denote the site-indices upon which the operators act, and generates site-indices for various geometries.
Some examples to showcase this:

```julia
using MPSKitModels
h = 0.5
H_ising = @mpoham sum(sigma_xx(){i, i + 1} + h * sigma_z(){i} for i in -Inf:Inf)

J = [1.0 -1.0]  # staggered couplings over unit cell of length 2
H_heisenberg = @mpoham sum(J[i] * sigma_exchange(){i, i + 1} for i in -Inf:2:Inf)

H_heisenberg_cylinder =
    @mpoham sum(J1 * sigma_exchange(){i, j} for (i, j) in nearest_neighbours(InfiniteCylinder(3)))

J1 = 0.8
J2 = 0.2

H_J1J2 = @mpoham sum(J1 * sigma_exchange(){i, j} for (i, j) in nearest_neighbours(InfiniteCylinder(4))) +
    sum(J2 * sigma_exchange(){i,j} for (i, j) in next_nearest_neighbours(InfiniteCylinder(4)))
```

For convenience, several models have already been defined. The full list can be found, along with all information in the docs.
