# Operators

```@meta
CurrentModule = MPSKitModels
```

There are several different operators defined, which all follow an interface similar to the following:
```julia
operator([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial]; kwargs...)
```
Here, the scalar type of the operator is defined by `elt`, while the symmetry can be chosen through the `symmetry` argument.
Other parameters are supplied as keywords.
The special keyword argument `side` can be used for operators that require an additional virtual space to satisfy the symmetry constraints, in which case it determines where this auxiliary space is located, either to the left `:L` (default) or to the right `:R`.

## Spin operators

The spin operators `S_x`, `S_y` and `S_z` are defined such that they obey the spin commutation relations ``[S^j, S^k] = i \varepsilon_{jkl} S^l``.
Additionally, the ladder operators are defined as ``S^{\pm} = S^x \pm i S^y``.
Several combinations are defined that act on two spins.

Supported values of `symmetry` for spin operators are `Trivial`, `Z2Irrep` and `U1Irrep`. 
When imposing symmetries, by convention we choose `S_z` as the diagonal operator for
``\mathrm{U}(1)``, and `S_x` as the diagonal operator for ``\mathbb{Z}_2``.

```@docs
S_x
S_y
S_z
S_plus
S_min
S_xx
S_yy
S_zz
S_plusmin
S_minplus
S_exchange
spinmatrices
```

For convenience, the Pauli matrices can also be recovered as ``σⁱ = 2 Sⁱ``.

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

## Q-state Potts operators

The Q-state Potts operators `potts_X` and `potts_Z` are defined to fulfill the braiding relation ``ZX = \omega XZ`` with ``\omega = e^{2\pi i/Q}``. 

Supported values of `symmetry` for the `X` operator are `Trivial` and `ZNIrrep{Q}`, while for the `Z` operator only `Trivial` is supported.

```@docs
potts_X
potts_Z
potts_ZZ
potts_field
weyl_heisenberg_matrices
```

## Bosonic operators

The bosonic creation and annihilation operators `a_plus` ($$a^+$$) and `a_min` ($$a^-$$) are defined such that the following holds:

$$a^+ \left|n\right> = \sqrt{n + 1} \left|n+1\right>$$
$$a^- \left|n\right> = \sqrt{n} \left|n-1\right>$$

From these, a number operator `a_number` ($$N$$) can be defined:

$$N = a^+ a^-$$
$$N\left|n\right> = n \left|n\right>$$

With these, the following commutators can be obtained:

$$\left[a^-, a^+\right] = 1$$
$$\left[N, a^+\right] = a^+$$
$$\left[N, a^-\right] = -a^-$$

Supported values of `symmetry` for bosonic operators are `Trivial` and `U1Irrep`.

```@docs
a_plus
a_min
a_number
```

## Fermionic operators

Spinless fermions.

```@docs
c_plus
c_min
c_number
```

Spinful fermions.

```@autodocs
Modules = [MPSKitModels.HubbardOperators]
```

TJ-model operators.

```@autodocs
Modules = [MPSKitModels.TJOperators]
```
