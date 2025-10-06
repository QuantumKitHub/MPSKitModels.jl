"""
    struct LocalOperator{T,G}

`N`-body operator acting on `N` sites, indexed through lattice points of type `G`. The
operator is represented as a vector of `MPOTens, instantiate_operatoror`s, each of which acts on a single site.

# Fields
- `opp::Vector{T}`: `N`-body operator represented by an MPO.
- `inds::Vector{G}`: `N` site indices.
"""
struct LocalOperator{T <: AbstractTensorMap{<:Number, <:Any, 2, 2}, G <: LatticePoint}
    opp::Vector{T}
    inds::Vector{G}
    function LocalOperator{T, G}(
            O::Vector{T}, inds::Vector{G}
        ) where {T <: AbstractTensorMap{<:Number, <:Any, 2, 2}, G <: LatticePoint}
        length(O) == length(inds) ||
            throw(ArgumentError("number of operators and indices should be the same"))
        issorted(inds) && allunique(inds) ||
            throw(ArgumentError("indices should be ascending and unique"))
        allequal(getfield.(inds, :lattice)) ||
            throw(ArgumentError("points should be defined on the same lattice"))
        return new{T, G}(O, inds)
    end
end

function LocalOperator(
        t::AbstractTensorMap{<:Number, <:Any, N, N}, inds::Vararg{G, N}
    ) where {N, G <: LatticePoint}
    p = TupleTools.sortperm(linearize_index.(inds))
    t = permute(t, (p, p .+ N))
    t_mpo = collect(MPSKit.decompose_localmpo(MPSKit.add_util_leg(t)))

    return LocalOperator{eltype(t_mpo), G}(t_mpo, collect(getindex.(Ref(inds), p)))
end

function MPSKit.instantiate_operator(lattice, O::LocalOperator)
    inds = linearize_index.(O.inds)
    mpo = FiniteMPO(O.opp)
    return MPSKit.instantiate_operator(lattice, inds => mpo)
end

# function LocalOperator(t::AbstractTensorMap{<:Number,<:Any,N,N}, inds::Vector) where {N}
#     allequal(typeof.(inds)) || throw(ArgumentError("indices should be of the same type"))
#     G = typeof(first(inds))
#     G <: LatticePoint || throw(ArgumentError("indices should be lattice points"))
#     return LocalOperator(t, convert(Vector{G}, inds))
# end

# function _fix_order(O::LocalOperator)
#     linds = linearize_index.(O.inds)
#     p = sortperm([linds...])
#     return LocalOperator(permute(O.opp, tuple(p...), tuple((p .+ numin(O.opp))...)),
#                          O.inds[p])
# end

Base.copy(O::LocalOperator{T, G}) where {T, G} = LocalOperator{T, G}(copy(O.opp), copy(O.inds))
function Base.deepcopy(O::LocalOperator{T, G}) where {T, G}
    return LocalOperator{T, G}(deepcopy(O.opp), deepcopy(O.inds))
end

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

Base.:/(a::LocalOperator, b::Number) = a * inv(b)
Base.:\(a::Number, b::LocalOperator) = inv(a) * b

function Base.:*(a::LocalOperator{T₁, G}, b::LocalOperator{T₂, G}) where {T₁, T₂, G}
    inds = sort!(union(a.inds, b.inds))
    T = promote_type(T₁, T₂)
    operators = Vector{T}(undef, length(inds))
    M = storagetype(T)

    left_vspace_A = space(first(a.opp), 1)
    left_vspace_B = space(first(b.opp), 1)

    for (i, ind) in enumerate(inds)
        i_A = findfirst(==(ind), a.inds)
        i_B = findfirst(==(ind), b.inds)

        right_vspace_A = isnothing(i_A) ? left_vspace_A : space(a.opp[i_A], 4)'
        right_vspace_B = isnothing(i_B) ? left_vspace_B : space(b.opp[i_B], 4)'

        left_fuse = unitary(
            M, fuse(left_vspace_B, left_vspace_A),
            left_vspace_B ⊗ left_vspace_A
        )
        right_fuse = unitary(
            M, fuse(right_vspace_B, right_vspace_A),
            right_vspace_B ⊗ right_vspace_A
        )

        if !isnothing(i_A) && !isnothing(i_B)
            @plansor operators[i][-1 -2; -3 -4] := b.opp[i_B][1 2; -3 4] *
                a.opp[i_A][3 -2; 2 5] * left_fuse[-1; 1 3] * conj(right_fuse[-4; 4 5])
        elseif !isnothing(i_A)
            @plansor operators[i][-1 -2; -3 -4] := τ[1 2; -3 4] *
                a.opp[i_A][3 -2; 2 5] * left_fuse[-1; 1 3] * conj(right_fuse[-4; 4 5])
        elseif !isnothing(i_B)
            @plansor operators[i][-1 -2; -3 -4] := b.opp[i_B][1 2; -3 4] *
                τ[3 -2; 2 5] * left_fuse[-1; 1 3] * conj(right_fuse[-4; 4 5])
        else
            error("this should not happen")
        end

        left_vspace_A = right_vspace_A
        left_vspace_B = right_vspace_B
    end

    return LocalOperator{T, G}(operators, inds)
end

lattice(O::LocalOperator) = first(O.inds).lattice
latticetype(O::LocalOperator) = latticetype(typeof(O))
latticetype(::Type{<:LocalOperator{T, G}}) where {T, G} = G
tensortype(::Union{O, Type{O}}) where {T, O <: LocalOperator{T}} = T
function TensorKit.spacetype(O::Union{LocalOperator, Type{<:LocalOperator}})
    return spacetype(tensortype(O))
end

# MPSKit.decompose_localmpo(O::LocalOperator) = MPSKit.decompose_localmpo(add_util_leg(O.opp))

"""
    SumOfLocalOperators{L<:LocalOperator}

Lazy sum of local operators.
"""
struct SumOfLocalOperators{L <: LocalOperator}
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

const LocalOrSumOfLocal = Union{LocalOperator, SumOfLocalOperators}
Base.:-(a::LocalOrSumOfLocal) = -1 * a
Base.:-(a::LocalOrSumOfLocal, b::LocalOrSumOfLocal) = a + (-b)

TensorKit.spacetype(Os::SumOfLocalOperators) = spacetype(typeof(Os))
TensorKit.spacetype(::Type{<:SumOfLocalOperators{L}}) where {L} = spacetype(L)

lattice(Os::SumOfLocalOperators) = lattice(first(Os.opps))

latticetype(Os::SumOfLocalOperators) = latticetype(typeof(Os))
latticetype(::Type{<:SumOfLocalOperators{L}}) where {L} = latticetype(L)
tensortype(::Union{Os, Type{<:Os}}) where {L, Os <: SumOfLocalOperators{L}} = tensortype(L)
