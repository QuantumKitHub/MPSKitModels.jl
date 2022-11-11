"""
    @mpoham(block)

Specify a Matrix Product Operator that is represented by a sum of local operators.

This macro converts expressions of the form `O{i...}` to an operator acting on sites `i...`
where `O` is an operator, and `i` can be an integer or a lattice point. The macro recognizes
expressions of the following forms:

*   `O{i...}` to indicate local operators `O` acting on sites `i...`
*   `-Inf:Inf`, `-∞:∞`, `-Inf:step:Inf`, `-∞:step:∞` to indicate infinite chains.
*   `1:L` to indicate finite chains.
*   `∑` to represent sums.

# Examples
```julia
H_ising = @mpoham sum(σᶻᶻ{i, i+1} + h * σˣ{i} for i in -Inf:Inf)
H_heisenberg = @mpoham ∑(sigma_exchange(){i,j} for (i,j) in nearest_neighbours(-∞:∞))
```
"""
macro mpoham(ex)
    for processor in (process_geometries_sugar, process_operators, process_sums,
                      addoperations)
        ex = postwalk(processor, ex)
    end
    return esc(ex)
end

function process_geometries_sugar(ex)
    @capture(ex, ((-Inf):(Inf | -∞):∞)) && return :(vertices(InfiniteChain()))
    @capture(ex, (((-Inf):step_:(Inf | -∞)):step_:∞)) &&
        return :(vertices(InfiniteChain($step)))
    return ex
end

function process_operators(ex)
    return @capture(ex, O_{inds__}) ? Expr(:call, :LocalOperator, O, inds...) : ex
end

function process_sums(ex)
    if @capture(ex, (sum([term_ for i_ in range_]))|(sum(term_ for i_ in range_)))
        return :(MPOHamiltonian(sum($term for $i in $range)))
    end
    return ex
end

const operations = (:LocalOperator, :MPOHamiltonian)

function addoperations(ex::Expr)
    if ex.head == :call && ex.args[1] in operations
        return Expr(ex.head, GlobalRef(MPSKitModels, ex.args[1]), ex.args[2:end]...)
    else
        return ex
    end
end
addoperations(ex) = ex

import MPSKit: MPOHamiltonian
MPSKitModels.MPOHamiltonian(args...) = MPSKit.MPOHamiltonian(args...)

"""
    LocalOperator{T,N}

`N`-body operator acting on `N` site indices.

# Fields
- `opp::T`: `N`-body operator.
- `inds::NTuple{N, Int}`: site indices.
"""
struct LocalOperator{T <: AbstractTensorMap, N}
    opp::T
    inds::NTuple{N, Int} # should be sorted
    function LocalOperator{T, N}(O::T, inds::NTuple{N, Int}) where {T, N}
        length(inds) == numind(O) // 2 ||
            error("length of indices should be compatible with operator")
        return new{T, N}(O, inds)
    end
end
LocalOperator(t, inds::NTuple{N, Int}) where {N} = LocalOperator{typeof(t), N}(t, inds)
LocalOperator(t, inds::Int...) = LocalOperator(t, inds)
LocalOperator(t, inds::LatticePoint...) = LocalOperator(t, inds)
function LocalOperator(t, inds::NTuple{N, LatticePoint}) where {N}
    LocalOperator{typeof(t), N}(t, linearize_index.(inds))
end

function _fix_order(O::LocalOperator)
    issorted(O.inds) && return O
    p = sortperm([O.inds...])
    return LocalOperator(permute(O.opp, tuple(p...), tuple((p .+ numin(O.opp))...)),
                         O.inds[p])
end

Base.:*(a::LocalOperator, b::Number) = LocalOperator(a.opp * b, a.inds)
Base.:*(a::Number, b::LocalOperator) = LocalOperator(a * b.opp, b.inds)
function Base.:*(a::LocalOperator, b::LocalOperator)
    return LocalOperator(a.opp ⊗ b.opp, (a.inds..., b.inds...))
end

"""
    SumOfLocalOperators{T<:Tuple}

Lazy sum of local operators.
"""
struct SumOfLocalOperators{T <: Tuple}
    opps::T
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

function _deduce_physical_spaces(inp::SumOfLocalOperators, unitcell)
    toret = PeriodicArray(Vector{Any}(missing, unitcell);)
    for lopp in inp.opps
        opp = lopp.opp
        for (i, j) in enumerate(lopp.inds)
            cs = space(opp, i)
            (ismissing(toret[j]) || toret[j] == cs) ||
                throw(SpaceMismatch("different spaces at site $(j)"))
            toret[j] = cs
        end
    end

    example = findfirst(x -> !x, ismissing.(toret))
    @assert !isnothing(example) "not a single physical operator is present"

    for i in 1:unitcell
        ismissing(toret[i]) || continue
        @warn "couldn't deduce physical space on site $(i), assuming $(oneunit(example))"
        toret[i] = oneunit(example)
    end

    return toret
end

function MPSKit.decompose_localmpo(O::LocalOperator, pspaces)
    O = _fix_order(O)
    mpo = MPSKit.decompose_localmpo(add_util_leg(O.opp))
    toret = [mpo[1]]
    mpo = mpo[2:end]

    li = O.inds[1]
    for ni in O.inds[2:end]
        virt = space(mpo[1], 1)
        for j in (li + 1):(ni - 1)
            push!(toret, convert(TensorMap, TensorKit.BraidingTensor(pspaces[j], virt)))
        end
        push!(toret, mpo[1])
        mpo = mpo[2:end]
        li = ni
    end
    return toret
end

function _find_free_channel(data, loc)
    hit = findfirst(map(x -> all(ismissing.(data[mod1(loc, end), :, x])),
                        2:(size(data, 2) - 1)))
    #hit = findfirst(ismissing.(data[loc,1,2:end-1]));
    if isnothing(hit)
        ndata = Array{Any, 3}(missing, size(data, 1), size(data, 2) + 1, size(data, 2) + 1)
        ndata[:, 1:(end - 1), 1:(end - 2)] .= data[:, :, 1:(end - 1)]
        ndata[:, 1:(end - 2), end] .= data[:, 1:(end - 1), end]
        ndata[:, end, end] .= data[:, end, end]
        hit = size(data, 2)
        data = ndata
    else
        hit += 1
    end
    return hit, data
end

function MPSKit.MPOHamiltonian(o::LocalOperator, unitcell = minimum(o.inds))
    return MPOHamiltonian(SumOfLocalOperators((o,)), unitcell)
end

function MPSKit.MPOHamiltonian(opps::SumOfLocalOperators,
                               unitcell = maximum(first.(map(i -> i.inds, opps.opps))),
                               pspaces = _deduce_physical_spaces(opps, unitcell))
    data = Array{Any, 3}(missing, unitcell, 2, 2)
    data[:, 1, 1] .= 1
    data[:, end, end] .= 1

    for opp in opps.opps
        start = minimum(opp.inds)
        stop = maximum(opp.inds)
        mpo = MPSKit.decompose_localmpo(opp, pspaces)

        if length(mpo) == 1
            if ismissing(data[start, 1, end])
                data[start, 1, end] = mpo[1]
            else
                data[start, 1, end] += mpo[1]
            end
            continue
        end

        hit, data = _find_free_channel(data, start)
        data[start, 1, hit] = mpo[1]
        for (s, o) in zip((start + 1):(stop - 1), mpo[2:(end - 1)])
            if ismissing(data[mod1(s, end), hit, hit]) && unitcell > 1
                data[mod1(s, end), hit, hit] = o
            else
                (nhit, data) = _find_free_channel(data, s)
                data[mod1(s, end), hit, nhit] = o
                hit = nhit
            end
        end

        data[mod1(stop, end), hit, end] = mpo[end]
    end

    return MPOHamiltonian(data)
end
