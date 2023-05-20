# MPSKitModels.jl

*Operators, models and QOL for working with MPSKit.jl*

## Table of contents

```@contents
Pages = ["home.md", "man/operators.md", "man/mpoham.md", "man/models.md", "index.md"]
Depth = 4
```

## Installation

Install with the package manager:

```julia
import Pkg
Pkg.add("MPSKitModels")
```

## Package features

*   A macro `@mpoham` for conveniently specifying (quasi-) 1D hamiltonians.
*   A list of predefined operators, optionally with enforced symmetry.
*   A list of predefined models

MPSKitModels.jl is centered around specifying MPOs through the combination of local operators that act on a finite number of sites, along with a specification of allowed sites.
The former are implemented using `AbstractTensorMap`s from TensorKit.jl, while the latter are defined through some geometry, such as a chain, strip or cylinder, and some notion of neighbours on this geometry.
Additionally, several commonly used models are provided.

## To do list

*   Add support for finite systems
*   Add support for non-local operators and partition functions