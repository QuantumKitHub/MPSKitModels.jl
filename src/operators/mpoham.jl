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
    for processor in (process_operators, process_sums, process_geometries_sugar, addoperations)
        ex = postwalk(processor, ex)
    end
    return Expr(:call, GlobalRef(MPSKit, :MPOHamiltonian), esc(ex))
end

function process_geometries_sugar(ex)
    @capture(ex, (((-Inf):Inf) | (-∞:∞))) && return :(vertices(InfiniteChain()))
    @capture(ex, (((-Inf):step_:(Inf | -∞)):step_:∞)) &&
        return :(vertices(InfiniteChain($step)))
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

import MPSKit: MPOHamiltonian

MPSKit.MPOHamiltonian(o::LocalOperator) = MPOHamiltonian(SumOfLocalOperators([o]))

"""
    deduce_pspaces(opps::SumOfLocalOperators)

Attempt to automatically deduce the physical spaces for all sites of the lattice
"""
function deduce_pspaces(opps::SumOfLocalOperators)
    pspaces = MPSKit.PeriodicVector{Union{spacetype(opps),Missing}}(missing,
                                                                    length(lattice(opps)))
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

    non_deduced = map(ismissing, pspaces)
    any(non_deduced) &&
        error("cannot automatically deduce physical spaces at $(findall(non_deduced))")

    return pspaces
end

# define a partial order on local operators, sorting them by starting site
# and then by decreasing length.
function _isless(a::L1, b::L2) where {L1<:LocalOperator,L2<:LocalOperator}
    return first(a.inds) == first(b.inds) ? length(a.inds) < length(b.inds) :
           first(a.inds) < first(b.inds)
end

function MPSKit.MPOHamiltonian(opps::SumOfLocalOperators, pspaces=deduce_pspaces(opps))
    T = tensortype(opps)
    T = AbstractTensorMap{scalartype(T),spacetype(T),numin(T),numout(T)}
    Tτ = TensorKit.BraidingTensor{scalartype(T),spacetype(T)}
    L = length(lattice(opps))
    data = PeriodicArray([Dict{Tuple{Int,Int},Union{T,Tτ}}() for _ in 1:L])
    vspaces = PeriodicArray([[oneunit(eltype(pspaces))] for _ in 1:L])

    # add operators
    for opp in sort(opps.opps; lt=_isless) # sort to minimize virtual size (?)
        linds = linearize_index.(opp.inds)
        mpo = opp.opp

        if length(mpo) == 1
            if haskey(data[linds[1]], (1, 0))
                data[linds[1]][(1, 0)] += mpo[1]
            else
                data[linds[1]][(1, 0)] = mpo[1]
            end
            continue
        end

        start, stop = first(linds), last(linds)
        push!(vspaces[start + 1], MPSKit.right_virtualspace(mpo[1])')
        lvl = length(vspaces[start + 1])
        data[start][1, lvl] = mpo[1]

        for site in (start + 1):(stop - 1)
            # add trivial spaces to avoid entries below diagonal
            while lvl > length(vspaces[site + 1]) + 1
                push!(vspaces[site + 1], vspaces[site + 1][1])
            end
            lvl′ = length(vspaces[site + 1]) + 1

            mpo_ind = findfirst(linds .== site)
            if isnothing(mpo_ind)
                push!(vspaces[site + 1], vspaces[site][end])
                data[site][lvl, lvl′] = Tτ(pspaces[site], vspaces[site + 1][end])
            else
                push!(vspaces[site + 1], MPSKit.right_virtualspace(o)')
                data[site][lvl, lvl′] = mpo[mpo_ind]
            end

            lvl = lvl′
        end
        
        data[stop][lvl, 0] = mpo[end]
    end

    # add identities
    maxlvl = maximum(length, vspaces) + 1
    for i in 1:L
        while length(vspaces[i]) < maxlvl
            push!(vspaces[i], vspaces[i][1])
        end
        τ = Tτ(pspaces[i], vspaces[i + 1][1])
        data[i][0, 0] = τ
        data[i][1, 1] = τ
    end

    # convert to BlockTensorMap
    Ws = map(1:L) do i
        P = SumSpace(pspaces[i])
        Vₗ = SumSpace(vspaces[i])
        Vᵣ = SumSpace(vspaces[i + 1])
        tdst = SparseBlockTensorMap{T}(undef, Vₗ ⊗ P ← P ⊗ Vᵣ)
        for ((i, j), t) in data[i]
            tdst[i == 0 ? maxlvl : i, 1, 1, j == 0 ? maxlvl : j] = t
        end
        return tdst
    end

    return MPOHamiltonian(PeriodicArray(Ws))
end
