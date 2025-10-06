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
    for processor in
        (process_operators, process_sums, process_geometries_sugar, addoperations)
        ex = postwalk(processor, ex)
    end
    return Expr(:call, GlobalRef(MPSKit, :MPOHamiltonian), esc(ex))
end

function process_geometries_sugar(ex)
    @capture(ex, (((-Inf):Inf) | ((-∞):∞))) && return :(vertices(InfiniteChain()))
    @capture(ex, (((-Inf):step_:(Inf | -∞)):step_:∞)) && return :(vertices(InfiniteChain($step)))
    return ex
end

function process_operators(ex)
    return @capture(ex, O_{inds__}) ? Expr(:call, :LocalOperator, O, inds...) : ex
end

function process_sums(ex)
    if @capture(ex, (sum([term_ for i_ in range_])) | (sum(term_ for i_ in range_)))
        # note: extra comma is necessary for destructuring arguments
        return :(sum(map(($i,) -> $term, $range)))
    end
    if @capture(ex, (∑([term_ for i_ in range_])) | (∑(term_ for i_ in range_)))
        # note: extra comma is necessary for destructuring arguments
        return :(sum(map(($i,) -> $term, $range)))
    end
    return ex
end

const operations = (:LocalOperator, :SumOfLocalOperators)

function addoperations(ex::Expr)
    if ex.head == :call && ex.args[1] in operations
        return Expr(ex.head, GlobalRef(MPSKitModels, ex.args[1]), ex.args[2:end]...)
    else
        return ex
    end
end
addoperations(ex) = ex

"""
    deduce_pspaces(opps::SumOfLocalOperators)

Attempt to automatically deduce the physical spaces for all sites of the lattice
"""
function deduce_pspaces(opps::SumOfLocalOperators)
    S = spacetype(opps)
    MissingS = Union{S, Missing}
    pspaces = MPSKit.PeriodicVector{MissingS}(missing, length(lattice(opps)))
    for opp in opps.opps
        for (ind, O) in zip(opp.inds, opp.opp)
            lind = linearize_index(ind)
            if !ismissing(pspaces[lind])
                pspaces[lind] == MPSKit.physicalspace(O) ||
                    error("incompatible physical spaces at $ind:\n$(pspaces[lind]) != $(MPSKit.physicalspace(O))")
            else
                pspaces[lind] = MPSKit.physicalspace(O)
            end
        end
    end

    not_missing = filter(!ismissing, pspaces)

    if length(not_missing) != length(pspaces) # Some spaces were not defined / not able to be deduced
        if allequal(not_missing) # all non-missing spaces are equal
            # fill in the missing spaces with the unique non-missing space
            uniquespace = first(not_missing)
            for i in eachindex(pspaces)
                pspaces[i] = uniquespace
            end
        else # Not all non-missing spaces are equal
            error("cannot automatically deduce physical spaces at $(findall(map(ismissing, pspaces)))")
        end
    end
    return collect(S, pspaces)
end

# define a partial order on local operators, sorting them by starting site
# and then by decreasing length.
function _isless(a::L, b::L) where {L <: LocalOperator}
    return first(a.inds) == first(b.inds) ? length(a.inds) < length(b.inds) :
        first(a.inds) < first(b.inds)
end

MPSKit.MPOHamiltonian(o::LocalOperator) = MPOHamiltonian(SumOfLocalOperators([o]))
function MPSKit.MPOHamiltonian(opps::SumOfLocalOperators)
    G = lattice(opps)
    pspaces = deduce_pspaces(opps)
    if isfinite(G)
        return FiniteMPOHamiltonian(pspaces, opps.opps)
    else
        return InfiniteMPOHamiltonian(pspaces, opps.opps)
    end
end
