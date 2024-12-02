# Operators that act on Hubbard-type models
# i.e. the local hilbert space consists of |∅⟩, |↑⟩, |↓⟩, |↑↓⟩
module HubbardOperators

using TensorKit

export hubbard_space
export e_plusmin, e_plusmin_up, e_plusmin_down
export e_minplus, e_minplus_up, e_minplus_down
export e_number, e_number_up, e_number_down, e_number_updown

export e⁺e⁻, e⁺e⁻ꜛ, e⁺e⁻ꜜ, e⁻e⁺, e⁻e⁺ꜛ, e⁻e⁺ꜜ
export nꜛ, nꜜ, nꜛnꜜ
# not exported because namespace: export n

"""
    hubbard_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the local hilbert space for a Hubbard-type model with the given particle and spin symmetries.
The possible symmetries are `Trivial`, `U1Irrep`, and `SU2Irrep`, for both particle number and spin.
"""
function hubbard_space(::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial)
    return Vect[FermionParity](0 => 2, 1 => 2)
end
function hubbard_space(::Type{Trivial}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
end
function hubbard_space(::Type{Trivial}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 2, (1, 1 // 2) => 1)
end
function hubbard_space(::Type{U1Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
end
function hubbard_space(::Type{U1Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1,
                                                   (1, 1, -1 // 2) => 1, (0, 2, 0) => 1)
end
function hubbard_space(::Type{U1Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ U1Irrep ⊠ SU2Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1,
                                                    (0, 2, 0) => 1)
end
function hubbard_space(::Type{SU2Irrep}, ::Type{Trivial})
    return Vect[FermionParity ⊠ SU2Irrep]((0, 0) => 2, (1, 1 // 2) => 1)
end
function hubbard_space(::Type{SU2Irrep}, ::Type{U1Irrep})
    return Vect[FermionParity ⊠ SU2Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1 // 2, 1) => 1)
end
function hubbard_space(::Type{SU2Irrep}, ::Type{SU2Irrep})
    return Vect[FermionParity ⊠ SU2Irrep ⊠ SU2Irrep]((1, 1 // 2, 1 // 2) => 1)
end

function single_site_operator(T, particle_symmetry::Type{<:Sector},
                              spin_symmetry::Type{<:Sector})
    V = hubbard_space(particle_symmetry, spin_symmetry)
    return TensorMap(zeros, T, V ← V)
end

function two_site_operator(T, particle_symmetry::Type{<:Sector},
                           spin_symmetry::Type{<:Sector})
    V = hubbard_space(particle_symmetry, spin_symmetry)
    return TensorMap(zeros, T, V ⊗ V ← V ⊗ V)
end

"""
    e_plusmin_up(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e†_{1,↑}, e_{2,↑}`` that creates a spin-up electron at the first site and annihilates a spin-up electron at the second.
"""
e_plusmin_up(P::Type{<:Sector}, S::Type{<:Sector}) = e_plusmin_up(ComplexF64, P, S)
function e_plusmin_up(T, ::Type{Trivial}, ::Type{Trivial})
    t = two_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 1, 1] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 1, 2] = 1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 2, 1] = -1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 2, 2, 2] = -1
    return t
end
function e_plusmin_up(T, ::Type{Trivial}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_plusmin_up(T, ::Type{Trivial}, ::Type{SU2Irrep})
    return error("Not implemented")
end
function e_plusmin_up(T, ::Type{U1Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_plusmin_up(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = two_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, 1 // 2), I(0, 0, 0), dual(I(0, 0, 0)), dual(I(1, 1, 1 // 2)))] .= 1
    t[(I(1, 1, 1 // 2), I(1, 1, -1 // 2), dual(I(0, 0, 0)), dual(I(0, 2, 0)))] .= 1
    t[(I(0, 2, 0), I(0, 0, 0), dual(I(1, 1, -1 // 2)), dual(I(1, 1, 1 // 2)))] .= -1
    t[(I(0, 2, 0), I(1, 1, -1 // 2), dual(I(1, 1, -1 // 2)), dual(I(0, 2, 0)))] .= -1
    return t
end
function e_plusmin_up(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    return error("Not implemented")
end
function e_plusmin_up(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_plusmin_up(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_plusmin_up(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    return error("Not implemented")
end
const e⁺e⁻ꜛ = e_plusmin_up

"""
    e_plusmin_down(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator ``e†_{1,↓}, e_{2,↓}`` that creates a spin-down electron at the first site and annihilates a spin-down electron at the second.
"""
e_plusmin_down(P::Type{<:Sector}, S::Type{<:Sector}) = e_plusmin_down(ComplexF64, P, S)
function e_plusmin_down(T, ::Type{Trivial}, ::Type{Trivial})
    t = two_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 1, 2] = -1
    t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 1, 2] = 1
    t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 1, 1, 2] = -1
    return t
end
function e_plusmin_down(T, ::Type{Trivial}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_plusmin_down(T, ::Type{Trivial}, ::Type{SU2Irrep})
    return error("Not implemented")
end
function e_plusmin_down(T, ::Type{U1Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_plusmin_down(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = two_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, -1 // 2), I(0, 0, 0), dual(I(0, 0, 0)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(1, 1, -1 // 2), I(1, 1, 1 // 2), dual(I(0, 0, 0)), dual(I(0, 2, 0)))] .= -1
    t[(I(0, 2, 0), I(0, 0, 0), dual(I(1, 1, 1 // 2)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(0, 2, 0), I(1, 1, 1 // 2), dual(I(1, 1, 1 // 2)), dual(I(0, 2, 0)))] .= -1
    return t
end
function e_plusmin_down(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    return error("Not implemented")
end
function e_plusmin_down(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_plusmin_down(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_plusmin_down(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    return error("Not implemented")
end
const e⁺e⁻ꜜ = e_plusmin_down

"""
    e_minplus_up(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the Hermitian conjugate of `e_plusmin_up`, i.e.
``(e†_{1,↑}, e_{2,↑})† = -e_{1,↑}, e†_{2,↑}`` (note the extra minus sign). 
It annihilates a spin-up electron at the first site and creates a spin-up electron at the second.
"""
e_minplus_up(P::Type{<:Sector}, S::Type{<:Sector}) = e_minplus_up(ComplexF64, P, S)
function e_minplus_up(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return copy(adjoint(e_plusmin_up(T, particle_symmetry, spin_symmetry)))
end
const e⁻⁺ꜛ = e_minplus_up

"""
    e_minplus_down(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the Hermitian conjugate of `e_plusmin_down`, i.e.
``(e†_{1,↓}, e_{2,↓})† = -e_{1,↓}, e†_{2,↓}`` (note the extra minus sign). 
It annihilates a spin-down electron at the first site and creates a spin-down electron at the second.
"""
e_minplus_down(P::Type{<:Sector}, S::Type{<:Sector}) = e_minplus_down(ComplexF64, P, S)
function e_minplus_down(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return copy(adjoint(e_plusmin_down(T, particle_symmetry, spin_symmetry)))
end
const e⁻e⁺ꜜ = e_minplus_down

"""
    e_plusmin(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `e_plusmin_up` and `e_plusmin_down`.
"""
e_plusmin(P::Type{<:Sector}, S::Type{<:Sector}) = e_plusmin(ComplexF64, P, S)
function e_plusmin(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return e_plusmin_up(T, particle_symmetry, spin_symmetry) +
           e_plusmin_down(T, particle_symmetry, spin_symmetry)
end
function e_plusmin(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = two_site_operator(T, U1Irrep, SU2Irrep)
    I = sectortype(t)
    f1 = only(fusiontrees((I(0, 0, 0), I(1, 1, 1 // 2)), I(1, 1, 1 // 2)))
    f2 = only(fusiontrees((I(1, 1, 1 // 2), I(0, 0, 0)), I(1, 1, 1 // 2)))
    t[f1, f2] .= 1
    f3 = only(fusiontrees((I(1, 1, 1 // 2), I(0, 2, 0)), I(1, 3, 1 // 2)))
    f4 = only(fusiontrees((I(0, 2, 0), I(1, 1, 1 // 2)), I(1, 3, 1 // 2)))
    t[f3, f4] .= -1
    f5 = only(fusiontrees((I(0, 0, 0), I(0, 2, 0)), I(0, 2, 0)))
    f6 = only(fusiontrees((I(1, 1, 1 // 2), I(1, 1, 1 // 2)), I(0, 2, 0)))
    t[f5, f6] .= sqrt(2)
    f7 = only(fusiontrees((I(1, 1, 1 // 2), I(1, 1, 1 // 2)), I(0, 2, 0)))
    f8 = only(fusiontrees((I(0, 2, 0), I(0, 0, 0)), I(0, 2, 0)))
    t[f7, f8] .= sqrt(2)
    return t
end
const e⁺e⁻ = e_plusmin

"""
    e_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `e_minplus_up` and `e_minplus_down`.
"""
e_minplus(P::Type{<:Sector}, S::Type{<:Sector}) = e_minplus(ComplexF64, P, S)
function e_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return copy(adjoint(e_plusmin(T, particle_symmetry, spin_symmetry)))
end
const e⁻e⁺ = e_minplus

"""
    e_number_up(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the one-body operator that counts the number of spin-up electrons.
"""
e_number_up(P::Type{<:Sector}, S::Type{<:Sector}) = e_number_up(ComplexF64, P, S)
function e_number_up(T::Type{<:Number}, ::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial)
    t = single_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(1))][1, 1] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
function e_number_up(T, ::Type{Trivial}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_number_up(T, ::Type{Trivial}, ::Type{SU2Irrep})
    throw(ArgumentError("`e_number_up` is not symmetric under `SU2Irrep` spin symmetry"))
end
function e_number_up(T, ::Type{U1Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_number_up(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = single_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    block(t, I(1, 1, 1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 1
    return t
end
function e_number_up(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`e_number_up` is not symmetric under `SU2Irrep` spin symmetry"))
end
function e_number_up(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_number_up(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_number_up(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`e_number_up` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜛ = e_number_up

"""
    e_number_down(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the one-body operator that counts the number of spin-down electrons.
"""
e_number_down(P::Type{<:Sector}, S::Type{<:Sector}) = e_number_down(ComplexF64, P, S)
function e_number_down(T::Type{<:Number}, ::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial)
    t = single_site_operator(T, Trivial, Trivial)
    I = sectortype(t)
    t[(I(1), I(1))][2, 2] = 1
    t[(I(0), I(0))][2, 2] = 1
    return t
end
function e_number_down(T, ::Type{Trivial}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_number_down(T, ::Type{Trivial}, ::Type{SU2Irrep})
    throw(ArgumentError("`e_number_down` is not symmetric under `SU2Irrep` spin symmetry"))
end
function e_number_down(T, ::Type{U1Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_number_down(T, ::Type{U1Irrep}, ::Type{U1Irrep})
    t = single_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    block(t, I(1, 1, -1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 1
    return t
end
function e_number_down(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`e_number_down` is not symmetric under `SU2Irrep` spin symmetry"))
end
function e_number_down(T, ::Type{SU2Irrep}, ::Type{Trivial})
    return error("Not implemented")
end
function e_number_down(T, ::Type{SU2Irrep}, ::Type{U1Irrep})
    return error("Not implemented")
end
function e_number_down(T, ::Type{SU2Irrep}, ::Type{SU2Irrep})
    throw(ArgumentError("`e_number_down` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜜ = e_number_down

"""
    e_number(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the one-body operator that counts the number of particles.
"""
e_number(P::Type{<:Sector}, S::Type{<:Sector}) = e_number(ComplexF64, P, S)
function e_number(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return e_number_up(T, particle_symmetry, spin_symmetry) +
           e_number_down(T, particle_symmetry, spin_symmetry)
end
function e_number(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = single_site_operator(T, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(1, 1, 1 // 2)) .= 1
    block(t, I(0, 2, 0)) .= 2
    return t
end
const n = e_number

"""
    e_number_updown(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})

Return the one-body operator that counts the number of doubly occupied sites.
"""
e_number_updown(P::Type{<:Sector}, S::Type{<:Sector}) = e_number_updown(ComplexF64, P, S)
function e_number_updown(T, particle_symmetry::Type{<:Sector},
                         spin_symmetry::Type{<:Sector})
    return e_number_up(T, particle_symmetry, spin_symmetry) *
           e_number_down(T, particle_symmetry, spin_symmetry)
end
function e_number_updown(T, ::Type{U1Irrep}, ::Type{SU2Irrep})
    t = single_site_operator(T, U1Irrep, SU2Irrep)
    I = sectortype(t)
    block(t, I(0, 2, 0)) .= 1
    return t
end
const nꜛnꜜ = e_number_updown

end
