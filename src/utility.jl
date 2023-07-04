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
