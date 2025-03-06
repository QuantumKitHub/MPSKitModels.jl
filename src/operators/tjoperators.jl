#= Operators that act on t-J-type models
i.e. the local hilbert space consists of 

- usual basis states: 
    |∅⟩, |↑⟩, |↓⟩
- slave-fermion basis states (c_σ = h† b_σ; holon h is fermionic, spinon b_σ is bosonic): 
    |h⟩ = h†|∅⟩, |↑'⟩ = (b↑)†|∅⟩, |↓'⟩ = (b↓)†|∅⟩
=#
module TJOperators

using TensorKit

export tj_space
export e_plusmin, e_plusmin_up, e_plusmin_down
export e_minplus, e_minplus_up, e_minplus_down
export e_minmin_updown, e_minmin_downup, e_singlet
export e_number, e_number_up, e_number_down, e_number_hole
export S_x, S_y, S_z, S_plus, S_min
export S_plusmin, S_minplus, S_exchange

export e⁺e⁻, e⁺e⁻ꜛ, e⁺e⁻ꜜ, e⁻e⁺, e⁻e⁺ꜛ, e⁻e⁺ꜜ
export nꜛ, nꜜ, nʰ
export Sˣ, Sʸ, Sᶻ, S⁺, S⁻
# not exported because namespace: export n

"""
    tj_space(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the local hilbert space for a t-J-type model with the given particle and spin symmetries.
The possible symmetries are 
- Particle number: `Trivial`, `U1Irrep`;
- Spin: `Trivial`, `U1Irrep`, `SU2Irrep`.

Setting `slave_fermion = true` switches to the slave-fermion basis. 
"""
function tj_space(::Type{Trivial}=Trivial, ::Type{Trivial}=Trivial;
                  slave_fermion::Bool=false)
    return slave_fermion ? Vect[FermionParity](0 => 2, 1 => 1) :
           Vect[FermionParity](0 => 1, 1 => 2)
end
function tj_space(::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1, (0, 1 // 2) => 1, (0, -1 // 2) => 1)
    else
        Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    end
end
function tj_space(::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return error("Not implemented")
end
function tj_space(::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep]((1, 0) => 1, (0, 1) => 2)
    else
        Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 2)
    end
end
function tj_space(::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    return if slave_fermion
        Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((1, 0, 0) => 1, (0, 1, 1 // 2) => 1,
                                                (0, 1, -1 // 2) => 1)
    else
        Vect[FermionParity ⊠ U1Irrep ⊠ U1Irrep]((0, 0, 0) => 1, (1, 1, 1 // 2) => 1,
                                                (1, 1, -1 // 2) => 1)
    end
end
function tj_space(::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return error("Not implemented")
end

function single_site_operator(T, particle_symmetry::Type{<:Sector},
                              spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return zeros(T, V ← V)
end

function two_site_operator(T, particle_symmetry::Type{<:Sector},
                           spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)
    V = tj_space(particle_symmetry, spin_symmetry; slave_fermion)
    return zeros(T, V ⊗ V ← V ⊗ V)
end

"""
    e_plusmin_up(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↑}, e_{2,↑}`` that creates a spin-up electron at the first site and annihilates a spin-up electron at the second.
The only nonzero matrix element corresponds to `|↑0⟩ <-- |0↑⟩`.
"""
function e_plusmin_up(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_plusmin_up(ComplexF64, P, S; slave_fermion)
end
function e_plusmin_up(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    #= The extra minus sign in slave-fermion basis:
    c†_{1,↑} c_{2,↑} |0↑⟩
    = h_1 b†_{1,↑} h†_2 b_{2,↑} h†_1 b†_{2,↑}|vac⟩
    = -b†_{1,↑} h†_2 h_1 h†_1 b_{2,↑} b†_{2,↑}|vac⟩
    = -b†_{1,↑} h†_2 |vac⟩
    = -|↑0⟩
    =#
    t[(I(b), I(h), dual(I(h)), dual(I(b)))][1, 1, 1, 1] = sgn * 1
    return t
end
function e_plusmin_up(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1 // 2), I(h, 0), dual(I(h, 0)), dual(I(b, 1 // 2)))] .= sgn * 1
    return t
end
function e_plusmin_up(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return error("Not implemented")
end
function e_plusmin_up(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1), I(h, 0), dual(I(h, 0)), dual(I(b, 1)))][1, 1, 1, 1] = sgn * 1
    return t
end
function e_plusmin_up(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1, 1 // 2), I(h, 0, 0), dual(I(h, 0, 0)), dual(I(b, 1, 1 // 2)))] .= sgn * 1
    return t
end
function e_plusmin_up(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return error("Not implemented")
end
const e⁺e⁻ꜛ = e_plusmin_up

"""
    e_plusmin_down(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e†_{1,↓}, e_{2,↓}`` that creates a spin-down electron at the first site and annihilates a spin-down electron at the second.
The only nonzero matrix element corresponds to `|↓0⟩ <-- |0↓⟩`.
"""
function e_plusmin_down(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_plusmin_down(ComplexF64, P, S; slave_fermion)
end
function e_plusmin_down(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b), I(h), dual(I(h)), dual(I(b)))][2, 1, 1, 2] = sgn * 1
    return t
end
function e_plusmin_down(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, -1 // 2), I(h, 0), dual(I(h, 0)), dual(I(b, -1 // 2)))] .= sgn * 1
    return t
end
function e_plusmin_down(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return error("Not implemented")
end
function e_plusmin_down(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1), I(h, 0), dual(I(h, 0)), dual(I(b, 1)))][2, 1, 1, 2] = sgn * 1
    return t
end
function e_plusmin_down(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(b, 1, -1 // 2), I(h, 0, 0), dual(I(h, 0, 0)), dual(I(b, 1, -1 // 2)))] .= sgn * 1
    return t
end
function e_plusmin_down(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    return error("Not implemented")
end
const e⁺e⁻ꜜ = e_plusmin_down

"""
    e_minplus_up(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the Hermitian conjugate of `e_plusmin_up`, i.e.
``(e†_{1,↑}, e_{2,↑})† = -e_{1,↑}, e†_{2,↑}`` (note the extra minus sign). 
It annihilates a spin-up electron at the first site and creates a spin-up electron at the second.
The only nonzero matrix element corresponds to `|0↑⟩ <-- |↑0⟩`.
"""
function e_minplus_up(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_minplus_up(ComplexF64, P, S; slave_fermion)
end
function e_minplus_up(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                      slave_fermion::Bool=false)
    return copy(adjoint(e_plusmin_up(T, particle_symmetry, spin_symmetry; slave_fermion)))
end
const e⁻⁺ꜛ = e_minplus_up

"""
    e_minplus_down(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the Hermitian conjugate of `e_plusmin_down`, i.e.
``(e†_{1,↓}, e_{2,↓})† = -e_{1,↓}, e†_{2,↓}`` (note the extra minus sign). 
It annihilates a spin-down electron at the first site and creates a spin-down electron at the second.
The only nonzero matrix element corresponds to `|0↓⟩ <-- |↓0⟩`.
"""
function e_minplus_down(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_minplus_down(ComplexF64, P, S; slave_fermion)
end
function e_minplus_down(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                        slave_fermion::Bool=false)
    return copy(adjoint(e_plusmin_down(T, particle_symmetry, spin_symmetry; slave_fermion)))
end
const e⁻e⁺ꜜ = e_minplus_down

"""
    e_plusmin(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator that creates a particle at the first site and annihilates a particle at the second.
This is the sum of `e_plusmin_up` and `e_plusmin_down`.
"""
function e_plusmin(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_plusmin(ComplexF64, P, S; slave_fermion)
end
function e_plusmin(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                   slave_fermion::Bool=false)
    return e_plusmin_up(T, particle_symmetry, spin_symmetry; slave_fermion) +
           e_plusmin_down(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const e⁺e⁻ = e_plusmin

"""
    e_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator that annihilates a particle at the first site and creates a particle at the second.
This is the sum of `e_minplus_up` and `e_minplus_down`.
"""
function e_minplus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_minplus(ComplexF64, P, S; slave_fermion)
end
function e_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                   slave_fermion::Bool=false)
    return copy(adjoint(e_plusmin(T, particle_symmetry, spin_symmetry; slave_fermion)))
end
const e⁻e⁺ = e_minplus

"""
    e_minmin_updown(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↑} e_{2,↓}`` that annihilates a spin-up particle at the first site and a spin-down particle at the second site.
The only nonzero matrix element corresponds to `|00⟩ <-- |↑↓⟩`.
"""
function e_minmin_updown(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_minmin_updown(ComplexF64, P, S; slave_fermion)
end
function e_minmin_updown(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 1, 2] = -sgn * 1
    return t
end
function e_minmin_updown(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h, 0), I(h, 0), dual(I(b, 1 // 2)), dual(I(b, -1 // 2)))] .= -sgn * 1
    return t
end
function e_minmin_updown(T, ::Type{U1Irrep}, ::Type{<:Sector}; slave_fermion::Bool=false)
    throw(ArgumentError("`e_minmin_updown` is not symmetric under `U1Irrep` particle symmetry"))
end

"""
    e_minmin_downup(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator ``e_{1,↓} e_{2,↑}`` that annihilates a spin-down particle at the first site and a spin-up particle at the second site.
The only nonzero matrix element corresponds to `|00⟩ <-- |↓↑⟩`.
"""
function e_minmin_downup(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_minmin_downup(ComplexF64, P, S; slave_fermion)
end
function e_minmin_downup(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h), I(h), dual(I(b)), dual(I(b)))][1, 1, 2, 1] = -sgn * 1
    return t
end
function e_minmin_downup(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    (h, b, sgn) = slave_fermion ? (1, 0, -1) : (0, 1, 1)
    t[(I(h, 0), I(h, 0), dual(I(b, -1 // 2)), dual(I(b, 1 // 2)))] .= -sgn * 1
    return t
end
function e_minmin_downup(T, ::Type{U1Irrep}, ::Type{<:Sector}; slave_fermion::Bool=false)
    throw(ArgumentError("`e_minmin_downup` is not symmetric under `U1Irrep` particle symmetry"))
end

"""
    e_singlet(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body singlet operator ``(e_{1,↓} e_{2,↑} - e_{1,↓} e_{2,↑}) / sqrt(2)``.
"""
function e_singlet(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_singlet(ComplexF64, P, S; slave_fermion)
end
function e_singlet(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                   slave_fermion::Bool=false)
    return (e_minmin_updown(T, particle_symmetry, spin_symmetry; slave_fermion) -
            e_minmin_downup(T, particle_symmetry, spin_symmetry; slave_fermion)) / sqrt(2)
end

"""
    e_number_up(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the one-body operator that counts the number of spin-up electrons.
"""
function e_number_up(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_number_up(ComplexF64, P, S; slave_fermion)
end
function e_number_up(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                     slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 1] = 1
    return t
end
function e_number_up(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), dual(I(b, 1 // 2)))][1, 1] = 1
    return t
end
function e_number_up(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`e_number_up` is not symmetric under `SU2Irrep` spin symmetry"))
end
function e_number_up(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 1] = 1
    return t
end
function e_number_up(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), dual(I(b, 1, 1 // 2)))] .= 1
    return t
end
function e_number_up(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`e_number_up` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜛ = e_number_up

"""
    e_number_down(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of spin-down electrons.
"""
function e_number_down(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_number_down(ComplexF64, P, S; slave_fermion)
end
function e_number_down(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial};
                       slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][2, 2] = 1
    return t
end
function e_number_down(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, -1 // 2), dual(I(b, -1 // 2)))][1, 1] = 1
    return t
end
function e_number_down(T, ::Type{Trivial}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`e_number_down` is not symmetric under `SU2Irrep` spin symmetry"))
end
function e_number_down(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][2, 2] = 1
    return t
end
function e_number_down(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)))] .= 1
    return t
end
function e_number_down(T, ::Type{U1Irrep}, ::Type{SU2Irrep}; slave_fermion::Bool=false)
    throw(ArgumentError("`e_number_down` is not symmetric under `SU2Irrep` spin symmetry"))
end
const nꜜ = e_number_down

"""
    e_number(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of particles.
"""
function e_number(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_number(ComplexF64, P, S; slave_fermion)
end
function e_number(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                  slave_fermion::Bool=false)
    return e_number_up(T, particle_symmetry, spin_symmetry; slave_fermion) +
           e_number_down(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const n = e_number

"""
    e_number_hole(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body operator that counts the number of holes.
"""
function e_number_hole(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return e_number_hole(ComplexF64, P, S; slave_fermion)
end
function e_number_hole(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                       slave_fermion::Bool=false)
    iden = TensorKit.id(tj_space(particle_symmetry, spin_symmetry; slave_fermion))
    return iden - e_number(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const nʰ = e_number_hole

"""
    S_x(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 x-operator on the electrons.
"""
function S_x(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_x(ComplexF64, P, S; slave_fermion)
end
function S_x(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 2] = 0.5
    t[(I(b), dual(I(b)))][2, 1] = 0.5
    return t
end
function S_x(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 2] = 0.5
    t[(I(b, 1), dual(I(b, 1)))][2, 1] = 0.5
    return t
end
const Sˣ = S_x

"""
    S_y(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 x-operator on the electrons (only defined for `Trivial` symmetry). 
"""
function S_y(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_y(ComplexF64, P, S; slave_fermion)
end
function S_y(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 2] = -0.5im
    t[(I(b), dual(I(b)))][2, 1] = 0.5im
    return t
end
function S_y(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 2] = -0.5im
    t[(I(b, 1), dual(I(b, 1)))][2, 1] = 0.5im
    return t
end
const Sʸ = S_y

"""
    S_z(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the one-body spin-1/2 z-operator on the electrons. 
"""
function S_z(P::Type{<:Sector}=Trivial, S::Type{<:Sector}=Trivial;
             slave_fermion::Bool=false)
    return S_z(ComplexF64, P, S; slave_fermion)
end
function S_z(T::Type{<:Number}, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), dual(I(b)))][1, 1] = 0.5
    t[(I(b), dual(I(b)))][2, 2] = -0.5
    return t
end
function S_z(T::Type{<:Number}, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), dual(I(b, 1 // 2)))] .= 0.5
    t[(I(b, -1 // 2), dual(I(b, -1 // 2)))] .= -0.5
    return t
end
function S_z(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), dual(I(b, 1)))][1, 1] = 0.5
    t[(I(b, 1), dual(I(b, 1)))][2, 2] = -0.5
    return t
end
function S_z(T::Type{<:Number}, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = single_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), dual(I(b, 1, 1 // 2)))] .= 0.5
    t[(I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)))] .= -0.5
    return t
end
const Sᶻ = S_z

"""
    S_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the spin-plus operator.
"""
function S_plus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_plus(ComplexF64, P, S; slave_fermion)
end
function S_plus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                slave_fermion::Bool=false)
    return S_x(T, particle_symmetry, spin_symmetry; slave_fermion) +
           1im * S_y(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const S⁺ = S_plus

"""
    S_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool=false)

Return the spin-minus operator.
"""
function S_min(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_min(ComplexF64, P, S; slave_fermion)
end
function S_min(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
               slave_fermion::Bool=false)
    return S_x(T, particle_symmetry, spin_symmetry; slave_fermion) -
           1im * S_y(T, particle_symmetry, spin_symmetry; slave_fermion)
end
const S⁻ = S_min

"""
    S_plusmin(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator S⁺S⁻.
The only nonzero matrix element corresponds to `|↑↓⟩ <-- |↓↑⟩`.
"""
function S_plusmin(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_plusmin(ComplexF64, P, S; slave_fermion)
end
function S_plusmin(T, ::Type{Trivial}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b), I(b), dual(I(b)), dual(I(b)))][1, 2, 2, 1] = 1
    return t
end
function S_plusmin(T, ::Type{Trivial}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, Trivial, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1 // 2), I(b, -1 // 2), dual(I(b, -1 // 2)), dual(I(b, 1 // 2)))] .= 1
    return t
end
function S_plusmin(T, ::Type{U1Irrep}, ::Type{Trivial}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, Trivial; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1), I(b, 1), dual(I(b, 1)), dual(I(b, 1)))][1, 2, 2, 1] = 1
    return t
end
function S_plusmin(T, ::Type{U1Irrep}, ::Type{U1Irrep}; slave_fermion::Bool=false)
    t = two_site_operator(T, U1Irrep, U1Irrep; slave_fermion)
    I = sectortype(t)
    b = slave_fermion ? 0 : 1
    t[(I(b, 1, 1 // 2), I(b, 1, -1 // 2), dual(I(b, 1, -1 // 2)), dual(I(b, 1, 1 // 2)))] .= 1
    return t
end

"""
    S_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the two-body operator S⁻S⁺.
The only nonzero matrix element corresponds to `|↓↑⟩ <-- |↑↓⟩`.
"""
function S_minplus(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_minplus(ComplexF64, P, S; slave_fermion)
end
function S_minplus(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                   slave_fermion::Bool=false)
    return copy(adjoint(S_plusmin(T, particle_symmetry, spin_symmetry; slave_fermion)))
end

"""
    S_exchange(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; slave_fermion::Bool = false)

Return the spin exchange operator S⋅S.
"""
function S_exchange(P::Type{<:Sector}, S::Type{<:Sector}; slave_fermion::Bool=false)
    return S_exchange(ComplexF64, P, S; slave_fermion)
end
function S_exchange(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                    slave_fermion::Bool=false)
    Sz = S_z(T, particle_symmetry, spin_symmetry; slave_fermion)
    return (1 / 2) * (S_plusmin(T, particle_symmetry, spin_symmetry; slave_fermion)
                      +
                      S_minplus(T, particle_symmetry, spin_symmetry; slave_fermion)) +
           Sz ⊗ Sz
end

end
