"""
    struct LocalOperator{T,G}

`N`-body operator acting on `N` sites, indexed through lattice points of type `G`. The
operator is represented as a vector of `MPOTensor`s, each of which acts on a single site.

# Fields
- `opp::Vector{T}`: `N`-body operator represented by an MPO.
- `inds::Vector{G}`: `N` site indices.
"""
struct LocalOperator{T<:AbstractTensorMap{<:Any,2,2},G<:LatticePoint}
    opp::Vector{T}
    inds::Vector{G}
    function LocalOperator{T,G}(O::Vector{T}, inds::Vector{G}) where {T<:AbstractTensorMap{<:Any,2,2},G<:LatticePoint}
        length(O) == length(inds) ||
            throw(ArgumentError("number of operators and indices should be the same"))
        issorted(inds) && allunique(inds) ||
            throw(ArgumentError("indices should be ascending and unique"))
        allequal(getfield.(inds, :lattice)) ||
            throw(ArgumentError("points should be defined on the same lattice"))
        return new{T,G}(O, inds)
    end
end

function LocalOperator(t::AbstractTensorMap, inds::Vector{G}) where {G<:LatticePoint}
    numin(t) == numout(t) == length(inds) || throw(ArgumentError("number of indices should match number of incoming and outgoing indices of the operator"))
    linds = linearize_index.(inds)
    p = sortperm(linds)
    t = permute(t, (tuple(p...), tuple((p .+ numin(t))...)))
    t_mpo = collect(MPSKit.decompose_localmpo(MPSKit.add_util_leg(t)))
    return LocalOperator{eltype(t_mpo),G}(t_mpo, inds[p])
end

function LocalOperator(t::AbstractTensorMap, inds::Vector)
    allequal(typeof.(inds)) || throw(ArgumentError("indices should be of the same type"))
    G = typeof(first(inds))
    G <: LatticePoint || throw(ArgumentError("indices should be lattice points"))
    return LocalOperator(t, convert(Vector{G}, inds))
end

# function _fix_order(O::LocalOperator)
#     linds = linearize_index.(O.inds)
#     p = sortperm([linds...])
#     return LocalOperator(permute(O.opp, tuple(p...), tuple((p .+ numin(O.opp))...)),
#                          O.inds[p])
# end

Base.copy(O::LocalOperator{T,G}) where {T,G} = LocalOperator{T,G}(copy(O.opp), copy(O.inds))
Base.deepcopy(O::LocalOperator{T,G}) where {T,G} = LocalOperator{T,G}(deepcopy(O.opp), deepcopy(O.inds))

# Linear Algebra
# --------------
function LinearAlgebra.rmul!(a::LocalOperator, b::Number) 
    rmul!(first(a.opp), b)
    return a
end
function LinearAlgebra.lmul!(a::Number, b::LocalOperator) 
    lmul!(a, first(b.opp))
    return b
end

Base.:*(a::LocalOperator, b::Number) = rmul!(deepcopy(a), b)
Base.:*(a::Number, b::LocalOperator) = lmul!(a, deepcopy(b))

function LinearAlgebra.rdiv!(a::LocalOperator, b::Number)
    rdiv!(first(a.opp), b)
    return a
end
function LinearAlgebra.ldiv!(a::Number, b::LocalOperator)
    ldiv!(a, first(b.opp))
    return b
end

Base.:/(a::LocalOperator, b::Number) = deepcopy(a) / b
Base.:\(a::Number, b::LocalOperator) = a \ deepcopy(b)

# TODO: implement this
# function Base.:*(a::LocalOperator, b::LocalOperator)
#     inds = [a.inds..., b.inds]
#     a.inds == b.inds && return LocalOperator(a.opp * b.opp, a.inds)
#     allunique(inds) && return LocalOperator(a.opp ⊗ b.opp, (a.inds..., b.inds...))
#     error("operator multiplication with partial overlapping indices not implemented")
# end

lattice(O::LocalOperator) = first(O.inds).lattice
latticetype(O::LocalOperator) = latticetype(typeof(O))
latticetype(::Type{<:LocalOperator{T,G}}) where {T,G} = G
tensortype(::Union{O,Type{O}}) where {T,O<:LocalOperator{T}} = T
TensorKit.spacetype(O::Union{LocalOperator,Type{<:LocalOperator}}) = spacetype(tensortype(O))

# MPSKit.decompose_localmpo(O::LocalOperator) = MPSKit.decompose_localmpo(add_util_leg(O.opp))

"""
    SumOfLocalOperators{L<:LocalOperator}

Lazy sum of local operators.
"""
struct SumOfLocalOperators{L<:LocalOperator}
    opps::Vector{L}
    function SumOfLocalOperators(opps::Vector{<:LocalOperator})
        if length(opps) > 1
            allequal(lattice.(opps)) ||
                throw(ArgumentError("sum of operators only defined on the same lattice"))
            allequal(spacetype.(opps)) ||
                throw(ArgumentError("sum of operators only defined for same spacetypes"))
        end
        return new{eltype(opps)}(opps)
    end
end

SumOfLocalOperators() = SumOfLocalOperators([])
Base.:+(a::LocalOperator, b::LocalOperator) = SumOfLocalOperators(vcat(a, b))
Base.:+(a::SumOfLocalOperators, b::LocalOperator) = SumOfLocalOperators(vcat(a.opps, b))
Base.:+(a::LocalOperator, b::SumOfLocalOperators) = SumOfLocalOperators(vcat(a, b.opps))
function Base.:+(a::SumOfLocalOperators, b::SumOfLocalOperators)
    return SumOfLocalOperators(vcat(a.opps, b.opps))
end

Base.:*(a::Number, b::SumOfLocalOperators) = SumOfLocalOperators(a .* b.opps)
Base.:*(a::SumOfLocalOperators, b::Number) = SumOfLocalOperators(a.opps .* b)
Base.:\(a::Number, b::SumOfLocalOperators) = SumOfLocalOperators(a .\ b.opps)
Base.:/(a::SumOfLocalOperators, b::Number) = SumOfLocalOperators(a.opps ./ b)

const LocalOrSumOfLocal = Union{LocalOperator,SumOfLocalOperators}
Base.:-(a::LocalOrSumOfLocal) = -1 * a
Base.:-(a::LocalOrSumOfLocal, b::LocalOrSumOfLocal) = a + (-b)

TensorKit.spacetype(Os::SumOfLocalOperators) = spacetype(typeof(Os))
TensorKit.spacetype(::Type{<:SumOfLocalOperators{L}}) where {L} = spacetype(L)

lattice(Os::SumOfLocalOperators) = lattice(first(Os.opps))

latticetype(Os::SumOfLocalOperators) = latticetype(typeof(Os))
latticetype(::Type{<:SumOfLocalOperators{L}}) where {L} = latticetype(L)
tensortype(::Union{Os,Type{<:Os}}) where {L,Os<:SumOfLocalOperators{L}} = tensortype(L)