#===========================================================================================
    Spinless fermions
===========================================================================================#

"""
    c_plus(elt=ComplexF64, ::Type{fℤ₂}; side=:L)

fermionic creation operator.
"""
function c_plus(elt=ComplexF64; side=:L)
    if side === :L
        pspace = Vect[fℤ₂](0 => 1, 1 => 1)
        vspace = Vect[fℤ₂](1 => 1)
        c⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(c⁺)[fℤ₂(1)] .= one(elt)
    elseif side === :R
        C = c_plus(elt; side=:L)
        F = isometry(flip(space(C, 3)), space(C, 3))
        @plansor c⁺[-1 -2; -3] := C[-2; 1 2] * τ[1 2; 3 -3] * F[-1; 3]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return c⁺
end
const c⁺ = c_plus

"""
    c_min(elt=ComplexF64, ::Type{fℤ₂}; side=:L)

fermionic creation operator.
"""
function c_min(elt=ComplexF64; side=:L)
    if side === :L
        C = c_plus(elt; side=:L)'
        F = isometry(flip(space(C, 2)), space(C, 2))
        @plansor c⁻[-1; -2 -3] := C[-1 1; -2] * F[-3; 1]
    elseif side === :R
        c⁻ = permute(c_plus(elt; side=:L)', (2, 1), (3,))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return c⁻
end
const c⁻ = c_min

c_plusmin(elt=ComplexF64) = contract_twosite(c⁺(elt; side=:L), c⁻(elt; side=:R))
const c⁺c⁻ = c_plusmin
c_minplus(elt=ComplexF64) = contract_twosite(c⁻(elt; side=:L), c⁺(elt; side=:R))
const c⁻c⁺ = c_minplus
c_plusplus(elt=ComplexF64) = contract_twosite(c⁺(elt; side=:L), c⁺(elt; side=:R))
const c⁺c⁺ = c_plusplus
c_minmin(elt=ComplexF64) = contract_twosite(c⁻(elt; side=:L), c⁻(elt; side=:R))
const c⁻c⁻ = c_minmin

"""
    c_number(elt=ComplexF64, ::Type{fℤ₂}=fℤ₂)

fermionic number operator.
"""
function c_number(elt=ComplexF64)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    n = TensorMap(zeros, elt, pspace ← pspace)
    blocks(n)[fℤ₂(1)] .= one(elt)
    return n
end

#===========================================================================================
    spin 1/2 fermions
===========================================================================================#

"""
    e_plus(elt=Complexf64, particle_symmetry, spin_symmetry; side=:L)

The creation operator for electron-like fermions.
"""
function e_plus end
function e_plus(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    return e_plus(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
end

function e_plus(elt::Type{<:Number}=ComplexF64, ::Type{<:Union{fℤ₂,Z2Irrep}}=fℤ₂,
                ::Type{Trivial}=Trivial; side=:L)
    pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    vspace = Vect[fℤ₂](1 => 2)
    if side == :L
        e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(e⁺)[fℤ₂(0)][2, 2:3] .= [one(elt), -one(elt)]
        blocks(e⁺)[fℤ₂(1)][:, 1:2] .= [one(elt) zero(elt); zero(elt) one(elt)]
    elseif side == :R
        e⁺ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(e⁺)[fℤ₂(0)][[1, 3], :] .= [one(elt) zero(elt); zero(elt) one(elt)]
        blocks(e⁺)[fℤ₂(1)][3:4, :] .= [zero(elt) -one(elt); one(elt) zero(elt)]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁺
end
const e⁺ = e_plus

"""
    e_min(elt=Complexf64, particle_symmetry, spin_symmetry; side=:L)

The annihilation operator for electron-like fermions.
"""
function e_min end
function e_min(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...)
    return e_min(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
end

function e_min(elt::Type{<:Number}=ComplexF64, ::Type{<:Union{fℤ₂,Z2Irrep}}=fℤ₂,
               ::Type{Trivial}=Trivial; side=:L)
    pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    vspace = Vect[fℤ₂](1 => 2)
    if side == :L
        e⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(e⁻)[fℤ₂(0)][:, 1:2] .= [one(elt) zero(elt); zero(elt) one(elt)]
        blocks(e⁻)[fℤ₂(1)][:, [2, 4]] .= [zero(elt) -one(elt); one(elt) zero(elt)]
    elseif side == :R
        e⁻ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(e⁻)[fℤ₂(0)][2:3, 2] .= [one(elt), -one(elt)]
        blocks(e⁻)[fℤ₂(1)][[1, 3], :] .= [one(elt) zero(elt); zero(elt) one(elt)]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁻
end
const e⁻ = e_min

function e_plusmin(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁)
    return contract_twosite(e⁺(elt, particle_symmetry, spin_symmetry; side=:L),
                            e⁻(elt, particle_symmetry, spin_symmetry; side=:R))
end
const e⁺e⁻ = e_plusmin
function e_minplus(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁)
    return contract_twosite(e⁻(elt, particle_symmetry, spin_symmetry; side=:L),
                            e⁺(elt, particle_symmetry, spin_symmetry; side=:R))
end
const e⁻e⁺ = e_minplus

"""
    e_number(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁)

The number operator for electron-like fermions.
"""
function e_number end
function e_number(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})
    return e_number(ComplexF64, particle_symmetry, spin_symmetry)
end

function e_number(elt::Type{<:Number}=ComplexF64, ::Type{<:Union{fℤ₂,Z2Irrep}}=fℤ₂,
                  ::Type{Trivial}=Trivial)
    pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    n = TensorMap(zeros, elt, pspace ← pspace)
    blocks(n)[fℤ₂(1)][1, 1] = 1
    blocks(n)[fℤ₂(1)][2, 2] = 1
    blocks(n)[fℤ₂(0)][2, 2] = 2
    return n
end

function e_number_up(elt::Type{<:Number}=ComplexF64, ::Type{<:Union{fℤ₂,Z2Irrep}}=fℤ₂,
                     ::Type{Trivial}=Trivial)
    pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    n = TensorMap(zeros, elt, pspace ← pspace)
    blocks(n)[fℤ₂(1)][1, 1] = 1
    blocks(n)[fℤ₂(0)][2, 2] = 1
    return n
end

function e_number_down(elt::Type{<:Number}=ComplexF64, ::Type{<:Union{fℤ₂,Z2Irrep}}=fℤ₂,
                     ::Type{Trivial}=Trivial)
    pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    n = TensorMap(zeros, elt, pspace ← pspace)
    blocks(n)[fℤ₂(1)][2, 2] = 1
    blocks(n)[fℤ₂(0)][2, 2] = 1
    return n
end

function e_number_updown(elt::Type{<:Number}=ComplexF64, ::Type{<:Union{fℤ₂,Z2Irrep}}=fℤ₂,
                       ::Type{Trivial}=Trivial)
    pspace = Vect[fℤ₂](0 => 2, 1 => 2)
    n = TensorMap(zeros, elt, pspace ← pspace)
    blocks(n)[fℤ₂(0)][2, 2] = 1
    return n
end

const nꜛnꜜ = e_number_updown