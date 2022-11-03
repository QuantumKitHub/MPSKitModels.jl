macro mpoham(ex)
    for processor in (process_operators, process_sums, addoperations)
        ex = postwalk(processor, ex)
    end
    return esc(ex)
end

function process_operators(ex)
    return @capture(ex, O_{inds__}) ? Expr(:call, :LocalOperator, O, inds...) : ex
end

function process_sums(ex)
    if @capture(ex, (sum([term_ for i_ in range_])) | (sum(term_ for i_ in range_)))
        start, step, stop = destruct_range(range)
        if start in (:(-Inf), :(-∞)) && stop in (:(Inf), :(∞))
            return :(MPOHamiltonian(sum($term for $i in 1:$step)))
        else
            return :(sum($term for $i in $range))
        end
        error("finite sums not implemented")
    end
    return ex
end

function destruct_range(ex)
    if @capture(ex, start_:stop_)
        step = 1
    else
        @capture(ex, start_:step_:stop_) || error("invalid range expression $ex")
    end
    return start, step, stop
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