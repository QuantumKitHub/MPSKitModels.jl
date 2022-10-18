struct SumOfLocalOperators{T<:Tuple}
    opps::T
end

struct LocalOperator{T<:AbstractTensorMap,N}
    opp::T
    inds::NTuple{N,Int} # should be sorted
end

Base.:+(a::Nothing,b::LocalOperator) = b;
Base.:+(a::LocalOperator,b::LocalOperator) = SumOfLocalOperators((a,b));
Base.:+(a::SumOfLocalOperators,b::LocalOperator) = SumOfLocalOperators((a.opps...,b));
Base.:+(a::LocalOperator,b::SumOfLocalOperators) = SumOfLocalOperators((a,b.opps...));
Base.:+(a::SumOfLocalOperators,b::SumOfLocalOperators) = SumOfLocalOperators((a.opps...,b.opps...));

function _deduce_physical_spaces(inp::SumOfLocalOperators,unitcell)
    toret = PeriodicArray(Vector{Any}(missing,unitcell);)
    for lopp in inp.opps
        opp = lopp.opp;
        
        for (i,j) in enumerate(lopp.inds)
            cs = space(opp,1);
            @assert ismissing(toret[i]) || toret[i] == cs # space mismatch on physical site j
            toret[i] = cs;

            cs = space(opp,2);
            @assert ismissing(toret[j]) || toret[j] == cs # space mismatch on physical site j
            toret[j] = cs;
        end
    end

    example = findfirst(x->!x,ismissing.(toret));
    @assert !isnothing(example) # not a single physical operator is present

    for i in 1:unitcell
        ismissing(toret[i]) || continue
        @warn "couldn't deduce physical space on site $(i), assuming $(oneunit(example))"
        toret[i] = oneunit(example)
    end

    return toret
end

# decomposes the LocalOperator in an MPO, but also creates identity tensors when inp.inds is longer ranged
function MPSKit.decompose_localmpo(inp::LocalOperator,pspaces)
    mpo = MPSKit.decompose_localmpo(add_util_leg(inp.opp));
    toret = [mpo[1]];
    mpo = mpo[2:end];

    li = inp.inds[1];
    for ni in inp.inds[2:end]
        virt = space(mpo[1],1);
        for j in li+1:ni-1
            push!(toret,convert(TensorMap,TensorKit.BraidingTensor(pspaces[j],virt)))
        end
        push!(toret,mpo[1]);
        mpo = mpo[2:end]
        li = ni
    end
    toret
end

function _find_free_channel(data,loc)
    hit = findfirst(map(x->all(ismissing.(data[mod1(loc,end),:,x])),2:size(data,2)-1))
    #hit = findfirst(ismissing.(data[loc,1,2:end-1]));
    if isnothing(hit)
        ndata = Array{Any,3}(missing,size(data,1),size(data,2)+1,size(data,2)+1);
        ndata[:,1:end-1,1:end-2].=data[:,:,1:end-1]
        ndata[:,1:end-2,end] .= data[:,1:end-1,end]
        ndata[:,end,end] .= data[:,end,end];
        hit = size(data,2);
        data = ndata;
    else
        hit+=1;
    end
    (hit,data)
end
MPSKit.MPOHamiltonian(o::LocalOperator,unitcell=o.inds[1]) = MPOHamiltonian(SumOfLocalOperators((o,)),unitcell);
function MPSKit.MPOHamiltonian(opps::SumOfLocalOperators,
        unitcell = maximum(first.(map(i->i.inds,opps.opps))),
        pspaces = _deduce_physical_spaces(opps,unitcell))

    data = Array{Any,3}(missing,unitcell,2,2);
    data[:,1,1] .= 1;
    data[:,end,end] .= 1;
    
    for opp in opps.opps
        start = opp.inds[1];
        stop = opp.inds[end];
        mpo = MPSKit.decompose_localmpo(opp,pspaces);

        if length(mpo) == 1
            if ismissing(data[start,1,end])
                data[start,1,end] = mpo[1]
            else
                data[start,1,end] += mpo[1]
            end
            continue;
        end

        (hit,data) = _find_free_channel(data,start);
        data[start,1,hit] = mpo[1];
        for (s,o) in zip(start+1:stop-1,mpo[2:end-1])
            if ismissing(data[mod1(s,end),hit,hit])
                data[mod1(s,end),hit,hit] = o
            else
                (nhit,data) = _find_free_channel(data,s);
                data[mod1(s,end),hit,nhit] = o
                hit = nhit;
            end
        end
        data[mod1(stop,end),hit,end] = mpo[end]
    end

    MPOHamiltonian(data)
end
