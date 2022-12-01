# Operators

```@meta
CurrentModule = MPSKitModels
```

## Spin operators

The spin operators `sigma_x`, `sigma_y` and `sigma_z` are defined such that they obey the spin commutation relations ``[Sⱼ, Sₖ] = i ɛⱼₖₗ Sₗ``.
Additionally, the ladder operators are defined as ``S± = Sˣ ± i Sʸ``.
Several combinations are defined that act on two spins.

When imposing symmetries, by convention we choose `sigma_z` as the diagonal operator, such that for non-trivial symmetry only the combinations that are invariant under this symmetry are implemented.
As such, when defining the other single-site operators with a symmetry, an additional virtual space is required to carry the charge, which is by convention chosen as the second space in a (2,1) tensor.

```@docs
sigma_x
sigma_y
sigma_z
sigma_plus
sigma_min
sigma_xx
sigma_yy
sigma_zz
sigma_plusmin
sigma_minplus
sigma_exchange
```

For convenience, the spin 1/2 case, which reduces to the pauli matrices, have the exported unicode symbols:

```@docs
σˣ
σʸ
σᶻ
σ⁺
σ⁻
σˣˣ
σʸʸ
σᶻᶻ
σ⁺⁻
σ⁻⁺
σσ
```

## Bosonic operators
```@docs
```

## Fermionic operators

```@docs
```