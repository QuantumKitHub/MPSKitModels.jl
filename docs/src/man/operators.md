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

The spin operators `S_x`, `S_y` and `S_z` are defined such that they obey the spin commutation relations ``[Sⱼ, Sₖ] = i ɛⱼₖₗ Sₗ``.
Additionally, the ladder operators are defined as ``S± = Sˣ ± i Sʸ``.
Several combinations are defined that act on two spins.

When imposing symmetries, by convention we choose `S_z` as the diagonal operator for U₁, and `S_x` as the diagonal operator for ℤ₂.

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

## Bosonic operators

The bosonic creation and annihilation operators `a_plus` ($$a^\dagger$$) and `a_min` ($$a$$) are defined such that the following holds:

$$a^\dagger \left|n\right> = \sqrt(n + 1) \left|n+1\right>$$
$$a \left|n\right> = \sqrt(n) \left|n-1\right>$$

From these, a number operator ``a_number`` ($$N$$) can be defined:

$$N = a^\dagger a$$
$$N\left|n\right> = n \left|n\right>$$

With these, the following commutators can be obtained:

$$\left[a, a^\dagger\right] = 1$$
$$\left[N,a^\dagger\right] = a^\dagger$$
$$\left[N,a\right] = -a$$

```@docs
a_plus
a_min
a_number
```

## Fermionic operators

```@docs
c_plus
c_min
c_number
e_plus
e_min
e_number
```