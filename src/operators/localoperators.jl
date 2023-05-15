"""
    LocalOperator{N,T,G}

`N`-body operator acting on `N` sites, indexed through lattice points of type `G`.

# Fields
- `opp::T`: `N`-body operator.
- `inds::NTuple{N, G}`: site indices.
"""
struct LocalOperator{N,T<:AbstractTensorMap,G}
    opp::T
    inds::NTuple{N,G}
    function LocalOperator{N,T,G}(O::T, inds::NTuple{N,G}) where {N,T,G}
        allequal(getfield.(inds, :lattice)) ||
            throw(ArgumentError("points should be defined on the same lattice"))
        numin(O) == numout(O) == N ||
            throw(ArgumentError("Operator should be an $N,$N TensorMap"))
        return new{N,T,G}(O, inds)
    end
end

function LocalOperator(t::T, inds::NTuple{<:Any,G}) where {T<:AbstractTensorMap,G<:LatticePoint}
    return LocalOperator{length(inds),T,G}(t, inds)
end
LocalOperator(t, inds::LatticePoint...) = LocalOperator(t, inds)


function _fix_order(O::LocalOperator)
    linds = linearize_index.(O.inds)
    p = sortperm([linds...])
    return LocalOperator(permute(O.opp, tuple(p...), tuple((p .+ numin(O.opp))...)),
                         O.inds[p])
end

Base.:*(a::LocalOperator, b::Number) = LocalOperator(a.opp * b, a.inds)
Base.:*(a::Number, b::LocalOperator) = LocalOperator(a * b.opp, b.inds)

Base.:/(a::LocalOperator, b::Number) = LocalOperator(a.opp / b, a.inds)
Base.:\(a::Number, b::LocalOperator) = LocalOperator(a \ b.opp, b.inds)

function Base.:*(a::LocalOperator, b::LocalOperator)
    inds = [a.inds..., b.inds]
    a.inds == b.inds && return LocalOperator(a.opp * b.opp, a.inds)
    allunique(inds) && return LocalOperator(a.opp ⊗ b.opp, (a.inds..., b.inds...))
    error("operator multiplication with partial overlapping indices not implemented")
end

lattice(O::LocalOperator) = first(O.inds).lattice
latticetype(O::LocalOperator) = latticetype(typeof(O))
latticetype(::Type{<:LocalOperator{N,T,G}}) where {N,T,G} = G
tensortype(::Union{O,Type{O}}) where {T,O<:LocalOperator{<:Any,T}} = T
TensorKit.spacetype(O::Union{LocalOperator,Type{<:LocalOperator}}) = spacetype(tensortype(O))

MPSKit.decompose_localmpo(O::LocalOperator) = MPSKit.decompose_localmpo(add_util_leg(O.opp))

"""
    SumOfLocalOperators{T<:Tuple}

Lazy sum of local operators.
"""
struct SumOfLocalOperators{T<:Tuple}
    opps::T
    function SumOfLocalOperators(opps::Tuple)
        if length(opps) > 1
            allequal(lattice.(opps)) ||
                throw(ArgumentError("sum of operators only defined on the same lattice"))
            allequal(spacetype.(opps)) ||
                throw(ArgumentError("sum of operators only defined for same spacetypes"))
        end
        return new{typeof(opps)}(opps)
    end
end

SumOfLocalOperators() = SumOfLocalOperators(())
Base.:+(a::LocalOperator, b::LocalOperator) = SumOfLocalOperators((a, b))
Base.:+(a::SumOfLocalOperators, b::LocalOperator) = SumOfLocalOperators((a.opps..., b))
Base.:+(a::LocalOperator, b::SumOfLocalOperators) = SumOfLocalOperators((a, b.opps...))
function Base.:+(a::SumOfLocalOperators, b::SumOfLocalOperators)
    return SumOfLocalOperators((a.opps..., b.opps...))
end

Base.:*(a::Number, b::SumOfLocalOperators) = SumOfLocalOperators(a .* b.opps)
Base.:*(a::SumOfLocalOperators, b::Number) = SumOfLocalOperators(a.opps .* b)
Base.:\(a::Number, b::SumOfLocalOperators) = SumOfLocalOperators(a .\ b.opps)
Base.:/(a::SumOfLocalOperators, b::Number) = SumOfLocalOperators(a.opps ./ b)

const LocalOrSumOfLocal = Union{LocalOperator,SumOfLocalOperators}
Base.:-(a::LocalOrSumOfLocal) = -1 * a
Base.:-(a::LocalOrSumOfLocal, b::LocalOrSumOfLocal) = a + (-b)

TensorKit.spacetype(Os::SumOfLocalOperators) = spacetype(typeof(Os))
TensorKit.spacetype(::Type{<:SumOfLocalOperators{T}}) where {T} = spacetype(fieldtype(T, 1))

lattice(Os::SumOfLocalOperators) = lattice(first(Os.opps))

latticetype(Os::SumOfLocalOperators) = latticetype(typeof(Os))
latticetype(::Type{<:SumOfLocalOperators{T}}) where {T} = latticetype(fieldtype(T, 1))