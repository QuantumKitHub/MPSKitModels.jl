#===========================================================================================
    Spinless fermions
===========================================================================================#

"""
    c_plus([elt::Type{<:Number}=ComplexF64]; side=:L)
    c⁺([elt::Type{<:Number}=ComplexF64]; side=:L)

Fermionic creation operator.
"""
function c_plus(elt::Type{<:Number}=ComplexF64; side=:L)
    vspace = Vect[fℤ₂](1 => 1)
    if side === :L
        pspace = Vect[fℤ₂](0 => 1, 1 => 1)
        c⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(c⁺)[fℤ₂(1)] .= one(elt)
    elseif side === :R
        C = c_plus(elt; side=:L)
        F = isomorphism(storagetype(C), vspace, flip(vspace))
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
function c_min(elt::Type{<:Number}=ComplexF64; side=:L)
    if side === :L
        C = c_plus(elt; side=:L)'
        F = isomorphism(flip(space(C, 2)), space(C, 2))
        @planar c⁻[-1; -2 -3] := C[-1 1; -2] * F[-3; 1]
    elseif side === :R
        c⁻ = permute(c_plus(elt; side=:L)', ((2, 1), (3,)))
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
    c_number([elt::Type{<:Number}=ComplexF64])

Fermionic number operator.
"""
function c_number(elt::Type{<:Number}=ComplexF64)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    n = TensorMap(zeros, elt, pspace ← pspace)
    blocks(n)[fℤ₂(1)] .= one(elt)
    return n
end
