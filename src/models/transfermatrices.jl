#===========================================================================================
    Ising model
===========================================================================================#
"""
    classical_ising([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial];
                    beta=log(1+sqrt(2))/2)

MPO for the partition function of the two-dimensional classical Ising model, defined as

```math
\\mathcal{Z}(\\beta) = \\sum_{\\{s\\}} \\exp(-\\beta H(s)) \\text{ with } H(s) = \\sum_{\\langle i, j \\rangle} s_i s_j

```
where each classical spin can take the values ``s = \\pm 1``.
"""
function classical_ising end
function classical_ising(symmetry::Type{<:Sector}; kwargs...)
    return classical_ising(ComplexF64, symmetry; kwargs...)
end
function classical_ising(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial;
                         beta=log(1 + sqrt(2)) / 2)
    t = elt[exp(beta) exp(-beta); exp(-beta) exp(beta)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2)
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1

    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    return DenseMPO(TensorMap(o, ℂ^2 * ℂ^2, ℂ^2 * ℂ^2))
end

function classical_ising(elt::Type{<:Number}, ::Type{Z2Irrep}; beta=log(1 + sqrt(2)) / 2)
    x = cosh(beta)
    y = sinh(beta)

    sec = ℤ₂Space(0 => 1, 1 => 1)
    mpo = TensorMap(zeros, elt, sec * sec, sec * sec)
    blocks(mpo)[Irrep[ℤ₂](0)] = [2x^2 2x*y; 2x*y 2y^2]
    blocks(mpo)[Irrep[ℤ₂](1)] = [2x*y 2x*y; 2x*y 2x*y]

    return DenseMPO(mpo)
end

#===========================================================================================
    Six vertex model
===========================================================================================#

"""
    sixvertex([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial];
              a=1.0, b=1.0, c=1.0)

MPO for the partition function of the two-dimensional six vertex model.
"""
function sixvertex end
sixvertex(symmetry::Type{<:Sector}; kwargs...) = sixvertex(ComplexF64, symmetry; kwargs...)
function sixvertex(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial; a=1.0, b=1.0,
                   c=1.0)
    d = elt[a 0 0 0
            0 c b 0
            0 b c 0
            0 0 0 a]
    return DenseMPO(permute(TensorMap(d, ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), ((1, 2), (4, 3))))
end
function sixvertex(elt::Type{<:Number}, ::Type{U1Irrep}; a=1.0, b=1.0, c=1.0)
    pspace = U1Space(-1 // 2 => 1, 1 // 2 => 1)
    mpo = TensorMap(zeros, elt, pspace ⊗ pspace, pspace ⊗ pspace)
    blocks(mpo)[Irrep[U₁](0)] = [b c; c b]
    blocks(mpo)[Irrep[U₁](1)] = reshape([a], (1, 1))
    blocks(mpo)[Irrep[U₁](-1)] = reshape([a], (1, 1))
    return DenseMPO(permute(mpo, ((1, 2), (4, 3))))
end
function sixvertex(elt::Type{<:Number}, ::Type{CU1Irrep}; a=1.0, b=1.0, c=1.0)
    pspace = CU1Space(1 // 2 => 1)
    mpo = TensorMap(zeros, elt, pspace ⊗ pspace, pspace ⊗ pspace)
    blocks(mpo)[Irrep[CU₁](0, 0)] = reshape([b + c], (1, 1))
    blocks(mpo)[Irrep[CU₁](0, 1)] = reshape([-b + c], (1, 1))
    blocks(mpo)[Irrep[CU₁](1, 2)] = reshape([a], (1, 1))
    return DenseMPO(permute(mpo, ((1, 2), (4, 3))))
end

#===========================================================================================
    Hard hexagon model
===========================================================================================#

"""
    hard_hexagon([elt::Type{<:Number}=ComplexF64])

MPO for the partition function of the two-dimensional hard hexagon model.
"""
function hard_hexagon(elt::Type{<:Number}=ComplexF64)
    P = Vect[FibonacciAnyon](:τ => 1)
    O = TensorMap(ones, elt, P ⊗ P ← P ⊗ P)
    blocks(O)[FibonacciAnyon(:I)] *= 0
    return DenseMPO(O)
end

#===========================================================================================
    q-state clock model
===========================================================================================#

"""
    qstate_clock([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial]; beta::Number=1.0, q::Integer=3)

MPO for the partition function of the two-dimensional discrete clock model with ``q`` states.
"""
function qstate_clock(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial;
                      beta::Number=1.0, q::Integer=3)
    comega(d) = cos(2 * pi * d / q)
    O = zeros(elt, q, q, q, q)
    for i in 1:q, j in 1:q, k in 1:q, l in 1:q
        O[i, j, k, l] = exp(beta *
                            (comega(i - j) + comega(j - k) + comega(k - l) + comega(l - i)))
    end

    return DenseMPO(TensorMap(O, ℂ^q * ℂ^q, ℂ^q * ℂ^q))
end
