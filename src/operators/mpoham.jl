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
    if @capture(ex, (sum([term_ for i_ in range_])) | (sum(term_ for i_ in range_)))
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

function _find_free_channel(data, loc)
    hit = findfirst(map(x -> _is_free_channel(data, loc, x), 2:(size(data, 2) - 1)))
    #hit = findfirst(ismissing.(data[loc,1,2:end-1]));
    if isnothing(hit)
        ndata = Array{Any,3}(missing, size(data, 1), size(data, 2) + 1, size(data, 2) + 1)
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

function _is_free_channel(data, loc, channel)
    return all(ismissing.(data[mod1(loc, end), :, channel]))
end

function MPSKit.MPOHamiltonian(o::LocalOperator)
    return MPOHamiltonian(SumOfLocalOperators((o,)))
end

function MPSKit.MPOHamiltonian(opps::SumOfLocalOperators)
    data = Array{Any,3}(missing, length(lattice(opps)), 2, 2)
    data[:, 1, 1] .= 1
    data[:, end, end] .= 1

    for opp in opps.opps
        linds = linearize_index.(opp.inds)
        if !issorted(linds)
            opp = _fix_order(opp)
        end
        mpo = MPSKit.decompose_localmpo(opp)

        if length(mpo) == 1
            if ismissing(data[linds[1], 1, end])
                data[mod1(linds[1], end), 1, end] = mpo[1]
            else
                data[mod1(linds[1], end), 1, end] += mpo[1]
            end
            continue
        end
        
        start, stop = first(linds), last(linds)
        hit, data = _find_free_channel(data, start)
        
        data[mod1(start, end), 1, hit] = mpo[1]
        for site in (start + 1):(stop - 1)
            mpo_ind = findfirst(linds .== site)
            o = isnothing(mpo_ind) ? 1 : mpo[mpo_ind]
            
            if length(lattice(opps)) > 1 && _is_free_channel(data, site, hit)
                data[mod1(site, end), hit, hit] = o
            else
                nhit, data = _find_free_channel(data, site)
                data[mod1(site, end), hit, nhit] = o
                hit = nhit
            end
        end
        
        data[mod1(stop, end), hit, end] = mpo[end]
    end
    
    return MPOHamiltonian(data)
end
