"""
    cc([eltype])

pair of fermionic annihilation operators at two different sites.
"""
function cc(elt::Type{<:Number}=ComplexF64)
    pspace = Vect[FermionParity](0 => 1, 1 => 1)
    cc = TensorMap(zeros, elt, pspace^2 ← pspace^2)
    blocks(cc)[fℤ₂(0)][1, 2] = -one(elt)
    return cc
end

"""
    ccdag([eltype])

pair of fermionic annihilation and creation operator at two different sites.
"""
function ccdag(elt::Type{<:Number}=ComplexF64)
    pspace = Vect[FermionParity](0 => 1, 1 => 1)
    ccdag = TensorMap(zeros, elt, pspace^2 ← pspace^2)
    blocks(ccdag)[fℤ₂(1)][2, 1] = -one(elt)
    return ccdag
end

"""
    cdagc([eltype])

pair of fermionic creation and annihilation operator at two different sites.
"""
function cdagc(elt::Type{<:Number}=ComplexF64)
    pspace = Vect[FermionParity](0 => 1, 1 => 1)
    cdagc = TensorMap(zeros, elt, pspace^2 ← pspace^2)
    blocks(cdagc)[fℤ₂(1)][1, 2] = one(elt)
    return cdagc
end

"""
    cdagcdag([eltype])

pair of fermionic creation operators at two different sites.
"""
function cdagcdag(elt::Type{<:Number}=ComplexF64)
    pspace = Vect[FermionParity](0 => 1, 1 => 1)
    cdagcdag = TensorMap(zeros, elt, pspace^2 ← pspace^2)
    blocks(cdagcdag)[fℤ₂(0)][2, 1] = one(elt)
    return cdagcdag
end

"""
    number([eltype])

fermionic number operator.
"""
function number(elt::Type{<:Number}=ComplexF64)
    pspace = Vect[FermionParity](0 => 1, 1 => 1)
    n = TensorMap(zeros, elt, pspace, pspace)
    blocks(n)[fℤ₂(1)] .= one(elt)
    return n
end