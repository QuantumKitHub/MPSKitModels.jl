# MPSKitModels.jl

```julia
] add MPSKitModels
```

MPSKitModels.jl provides operators, tools and utilities for MPSKit.jl.
The main goal is to facilitate the definition and readability of hamiltonians on 1-dimensional systems, as well as quasi-1-dimensional systems defined on a cylinder or infinite strips.

## Overview

The main building blocks of these Hamiltonians are local ``N``-body operators, which should be provided in the form of an ``AbstractTensorMap{N,N}`` (see TensorKit.jl).
Several often-used operators are defined and exported within MPSKitModels.jl:

* spin operators (Sx, Sy, Sz, S+, S-)
* spin exchange couplings (Sxx, Syy, Szz, SS, S+S-, S-S+)
* fermionic creation and annihilation operators (c+, c, n)
* ...

These operators can then be combined to define Hamiltonians by way of the ``@mpoham`` macro.
This transforms ``{}`` expressions to denote the site-indices upon which the operators act, and generates site-indices for various geometries.
Some examples to showcase this:

```julia
using MPSKitModels
h = 0.5
H_ising = @mpoham sum(Sxx(){i, i+1} + h * Sz(){i} for i in -Inf:Inf)

J = [1.0 -1.0]  # staggered couplings over unit cell of length 2
H_heisenberg = @mpoham sum(J[i] * SS(){i, i+1} for i in -Inf:2:Inf)

H_heisenberg_cylinder =
    @mpoham sum(J1 * (SS{(i,j), (i+1,j)} + SS{(i,j), (i,j+1)}) for i in 1:N, j in -Inf:Inf)

J1 = 0.8
J2 = 0.2

H_J1J2 = @mpoham sum(J1 * SS{i,j} for (i,j) in nearest_neighbours(Cylinder(4))) +
    sum(J2 * SS{i,j} for (i,j) in next_nearest_neighbours(Cylinder(4)))
```

For convenience, several models have already been defined. The full list can be found, along with all information in the docs.