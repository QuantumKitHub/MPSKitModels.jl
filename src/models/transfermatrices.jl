#===========================================================================================
    Ising model
===========================================================================================#
"""
    classical_ising(; beta=log(1+sqrt(2))/2)

MPO for the classical Ising partition function, defined by
    
``Z(β) = ∑_s exp(-βH(s))`` with ``H(s) = ∑_{<i,j>}σ_i σ_j``
"""
function classical_ising(eltype=ComplexF64, ::Type{ℤ{1}}=ℤ{1}, lattice=InfiniteChain(1);
                         beta=log(1 + sqrt(2)) / 2)
    t = [exp(beta) exp(-beta); exp(-beta) exp(beta)]

    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2)
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1

    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    return DenseMPO(TensorMap(complex(o), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2))
end

function classical_ising(eltype, ::Type{ℤ₂}, lattice=InfiniteChain(1);
                         beta=log(1 + sqrt(2)) / 2)
    x = cosh(beta)
    y = sinh(beta)

    sec = ℤ₂Space(0 => 1, 1 => 1)
    mpo = TensorMap(zeros, ComplexF64, sec * sec, sec * sec)
    blocks(mpo)[Irrep[ℤ₂](0)] = [2x^2 2x*y; 2x*y 2y^2]
    blocks(mpo)[Irrep[ℤ₂](1)] = [2x*y 2x*y; 2x*y 2x*y]

    return DenseMPO(mpo)
end

#===========================================================================================
    Six vertex model
===========================================================================================#

"""
    sixvertex(; a=1.0, b=1.0, c=1.0)

MPO for the six vertex model.
"""
function sixvertex(eltype=ComplexF64, ::Type{ℤ{1}}=ℤ{1}, lattice=InfiniteChain(1);
                   a=1.0, b=1.0, c=1.0)
    d = [a 0 0 0
         0 c b 0
         0 b c 0
         0 0 0 a]
    return DenseMPO(permute(TensorMap(complex(d), ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), (1, 2), (4, 3)))
end

function sixvertex(eltype, ::Type{U₁}, lattice=InfiniteChain(1);
                   a=1.0, b=1.0, c=1.0)
    pspace = U1Space(-1 // 2 => 1, 1 // 2 => 1)
    mpo = TensorMap(zeros, eltype, pspace ⊗ pspace, pspace ⊗ pspace)
    blocks(mpo)[Irrep[U₁](0)] = [b c; c b]
    blocks(mpo)[Irrep[U₁](1)] = reshape([a], (1, 1))
    blocks(mpo)[Irrep[U₁](-1)] = reshape([a], (1, 1))
    return DenseMPO(permute(mpo, (1, 2), (4, 3)))
end

function sixvertex(eltype, ::Type{CU₁}, lattice=InfiniteChain(1);
                   a=1.0, b=1.0, c=1.0)
    pspace = CU1Space(1 // 2 => 1)
    mpo = TensorMap(zeros, eltype, pspace ⊗ pspace, pspace ⊗ pspace)
    blocks(mpo)[Irrep[CU₁](0, 0)] = reshape([b + c], (1, 1))
    blocks(mpo)[Irrep[CU₁](0, 1)] = reshape([-b + c], (1, 1))
    blocks(mpo)[Irrep[CU₁](1, 2)] = reshape([a], (1, 1))
    return DenseMPO(permute(mpo, (1, 2), (4, 3)))
end

#===========================================================================================
    Hard hexagon model
===========================================================================================#

"""
    hard_hexagon(eltype=ComplexF64)

MPO for the hard hexagon model.
"""
function hard_hexagon(eltype=ComplexF64)
    P = Vect[FibonacciAnyon](:τ => 1)
    O = TensorMap(ones, eltype, P ⊗ P ← P ⊗ P)
    blocks(O)[FibonacciAnyon(:I)] *= 0
    return DenseMPO(O)
end

#===========================================================================================
    q-state clock model
===========================================================================================#

"""
    qstate_clock(eltype=ComplexF64, ::Type{ℤ₁}=ℤ₁; beta::Number=1.0, q::Integer=3)

MPO for the discrete clock model with ``q`` states.
"""
function qstate_clock(eltype=ComplexF64, ::Type{ℤ₁}=ℤ₁; beta::Number=1.0, q::Integer=3)
    comega(d) = cos(2 * pi * d / q)
    O = zeros(eltype, q, q, q, q)
    for i in 1:q, j in 1:q, k in 1:q, l in 1:q
        O[i, j, k, l] = exp(beta * (comega(i - j) + comega(j - k) + comega(k - l) +
                                      comega(l - i)))
    end
    
    return DenseMPO(TensorMap(O, ℂ^q * ℂ^q, ℂ^q * ℂ^q))
end
