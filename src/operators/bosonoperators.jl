"""
    a_plus(cutoff, [eltype], [symmetry])

bosonic creation operator.
"""
function a_plus(cutoff::Int, eltype=ComplexF64, ::Type{ℤ{1}}=ℤ{1})
    a⁺ = TensorMap(zeros, eltype, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁺[n + 1, n] = sqrt(n)
    end
    return a⁺
end
function a_plus(cutoff::Int, eltype, ::Type{U₁}; side=:left)
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :left
        aspace = U1Space(+1 => 1)
        a⁺ = TensorMap(zeros, eltype, pspace ← pspace ⊗ aspace)
        for (c, b) in blocks(a⁺)
            b .= sqrt(c.charge)
        end
    elseif side === :right
        aspace = U1Space(-1 => 1)
        a⁺ = TensorMap(zeros, eltype, aspace ⊗ pspace ← pspace)
        for (c, b) in blocks(a⁺)
            b .= sqrt(c.charge + 1)
        end
    else
        throw(ArgumentError("invalid side (:$side)"))
    end
    return a⁺
end


"""
    a_min(cutoff, [eltype], [symmetry])

bosonic annihilation operator.
"""
function a_min(cutoff::Int, eltype=ComplexF64, ::Type{ℤ{1}}=ℤ{1})
    a⁻ = TensorMap(zeros, eltype, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁻[n, n + 1] = sqrt(n)
    end
    return a⁻
end
function a_min(cutoff::Int, eltype, ::Type{U₁}; side=:left)
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :left
        aspace = U1Space(-1 => 1)
        a⁻ = TensorMap(zeros, eltype, pspace ← pspace ⊗ aspace)
        for (c, b) in blocks(a⁻)
            b .= sqrt(c.charge + 1)
        end
    elseif side === :right
        aspace = U1Space(+1 => 1)
        a⁻ = TensorMap(zeros, eltype, aspace ⊗ pspace ← pspace)
        for (c, b) in blocks(a⁻)
            b .= sqrt(c.charge)
        end
    else
        throw(ArgumentError("invalid side (:$side)"))
    end
    return a⁻
end
