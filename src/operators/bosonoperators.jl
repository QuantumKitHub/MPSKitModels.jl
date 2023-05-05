"""
    a_plus(cutoff, [elt], [symmetry])

bosonic creation operator.
"""
function a_plus(cutoff::Int, elt=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; side=:L)
    a⁺ = TensorMap(zeros, elt, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁺[n + 1, n] = sqrt(n)
    end
    return a⁺
end
function a_plus(cutoff::Int, elt, ::Type{U₁}; side=:L)
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :L
        aspace = U1Space(+1 => 1)
        a⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ aspace)
        for (c, b) in blocks(a⁺)
            b .= sqrt(c.charge)
        end
    elseif side === :R
        aspace = U1Space(-1 => 1)
        a⁺ = TensorMap(zeros, elt, aspace ⊗ pspace ← pspace)
        for (c, b) in blocks(a⁺)
            b .= sqrt(c.charge + 1)
        end
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return a⁺
end


"""
    a_min(cutoff, [elt], [symmetry])

bosonic annihilation operator.
"""
function a_min(cutoff::Int, elt=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; side=:L)
    a⁻ = TensorMap(zeros, elt, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁻[n, n + 1] = sqrt(n)
    end
    return a⁻
end

function a_min(cutoff::Int, elt, ::Type{U₁}; side=:L)
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :L
        aspace = U1Space(-1 => 1)
        a⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ aspace)
        for (c, b) in blocks(a⁻)
            b .= sqrt(c.charge + 1)
        end
    elseif side === :R
        aspace = U1Space(+1 => 1)
        a⁻ = TensorMap(zeros, elt, aspace ⊗ pspace ← pspace)
        for (c, b) in blocks(a⁻)
            b .= sqrt(c.charge)
        end
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return a⁻
end
