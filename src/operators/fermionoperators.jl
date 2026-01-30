#===========================================================================================
    Spinless fermions
===========================================================================================#

"""
    c_plus([elt::Type{<:Number}=ComplexF64]; side=:L)
    c⁺([elt::Type{<:Number}=ComplexF64]; side=:L)

Fermionic creation operator.
"""
function c_plus(::Type{TorA}; side = :L) where {TorA}
    vspace = Vect[fℤ₂](1 => 1)
    if side === :L
        pspace = Vect[fℤ₂](0 => 1, 1 => 1)
        c⁺ = zeros(TorA, pspace ← pspace ⊗ vspace)
        block(c⁺, fℤ₂(1)) .= one(elt)
    elseif side === :R
        C = c_plus(TorA; side = :L)
        F = isomorphism(TorA, vspace, flip(vspace))
        @planar c⁺[-1 -2; -3] := C[-2; 1 2] * τ[1 2; 3 -3] * F[3; -1]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return c⁺
end
const c⁺ = c_plus

"""
    c_min([elt::Type{<:Number}=ComplexF64]; side=:L)
    c⁻([elt::Type{<:Number}=ComplexF64]; side=:L)

Fermionic annihilation operator.
"""
function c_min(::Type{TorA}; side = :L) where {TorA}
    if side === :L
        C = c_plus(TorA; side = :L)'
        F = isomorphism(TorA, flip(space(C, 2)), space(C, 2))
        @planar c⁻[-1; -2 -3] := C[-1 1; -2] * F[-3; 1]
    elseif side === :R
        c⁻ = permute(c_plus(TorA; side = :L)', ((2, 1), (3,)))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return c⁻
end

const c⁻ = c_min

c_plusmin(::Type{TorA}) where {TorA} = contract_twosite(c⁺(TorA; side = :L), c⁻(TorA; side = :R))
const c⁺c⁻ = c_plusmin
c_minplus(::Type{TorA}) where {TorA} = contract_twosite(c⁻(TorA; side = :L), c⁺(TorA; side = :R))
const c⁻c⁺ = c_minplus
c_plusplus(::Type{TorA}) where {TorA} = contract_twosite(c⁺(TorA; side = :L), c⁺(TorA; side = :R))
const c⁺c⁺ = c_plusplus
c_minmin(::Type{TorA}) where {TorA} = contract_twosite(c⁻(TorA; side = :L), c⁻(TorA; side = :R))
const c⁻c⁻ = c_minmin

"""
    c_number([elt::Type{<:Number}=ComplexF64])

Fermionic number operator.
"""
function c_number(::Type{TorA}) where {TorA}
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    n = zeros(TorA, pspace ← pspace)
    block(n, fℤ₂(1)) .= one(eltype(TorA))
    return n
end
