
function nonsym_spintensors(s)
    (Sxd, Syd, Szd) = spinmatrices(s)
    sp = ComplexSpace(size(Sxd, 1))

    Sx = TensorMap(Sxd, sp, sp)
    Sy = TensorMap(Syd, sp, sp)
    Sz = TensorMap(Szd, sp, sp)

    return Sx, Sy, Sz, one(Sx)
end

"""
bosonic creation anihilation operators with a cutoff
cutoff = maximal number of bosons at one location
"""
function nonsym_bosonictensors(cutoff::Int, elt=ComplexF64)
    creadat = zeros(elt, cutoff + 1, cutoff + 1)

    for i in 1:cutoff
        creadat[i + 1, i] = sqrt(i)
    end

    a⁺ = TensorMap(creadat, ℂ^(cutoff + 1), ℂ^(cutoff + 1))
    a⁻ = TensorMap(collect(creadat'), ℂ^(cutoff + 1), ℂ^(cutoff + 1))
    return (a⁺, a⁻)
end

# check all elements are equal -> only defined in 1.8+
@static if !isdefined(Base, :allequal)
    allequal(itr) = isempty(itr) ? true : all(isequal(first(itr)), itr)
end

#===========================================================================================
    Contractions
===========================================================================================#

"""
    contract_onesite(L, R)

contract two single-site operators inta a single-site operator.
"""
function contract_onesite(L::AbstractTensorMap{<:Any,1,2}, R::AbstractTensorMap{<:Any,2,1})
    @plansor H[-1; -2] := L[-1; 1 2] * τ[1 2; 3 4] * R[3 4; -2]
    return H
end
contract_onesite(L::AbstractTensorMap{<:Any,1,1}, R::AbstractTensorMap{<:Any,1,1}) = L * R

"""
    contract_twosite(L, R)

contract two single-site operators into a two-site operator.
"""
function contract_twosite(L::AbstractTensorMap{<:Any,1,2}, R::AbstractTensorMap{<:Any,2,1})
    @plansor H[-1 -2; -3 -4] := L[-1; -3 1] * R[1 -2; -4]
    return H
end
contract_twosite(L::AbstractTensorMap{<:Any,1,1}, R::AbstractTensorMap{<:Any,1,1}) = L ⊗ R
