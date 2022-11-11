#################
## Ising model ##
#################
"""
    transverse_field_ising(; J=1.0, h=1.0)

MPO for the hamiltonian of the transverse field Ising model, defined by
    ``H = -J(∑<i,j> X{i}X{j} + ∑<i> h * Z{i} + h2 * X{i})``
"""
function transverse_field_ising(eltype = ComplexF64, symmetry = ℤ{1},
                                lattice = InfiniteChain(1);
                                J = 1.0, h = 0.5, h2 = 0.0, spin = 1 // 2)
    symmetry == ℤ{1} || @assert h2==zero(h2) "parameters and symmetry incompatible"

    XX = sigma_xx(eltype, symmetry; spin = spin)
    X = sigma_x(eltype, symmetry; spin = spin)
    Z = sigma_z(eltype, symmetry; spin = spin)
    return @mpoham sum(-J * (XX{i, j} + h * Z{i} + h2 * X{i})
                       for (i, j) in nearest_neighbours(lattice))
end

"""
    classical_ising(; beta=log(1+sqrt(2))/2)

MPO for the classical Ising partition function, defined by
    ``Z(β) = ∑ₛ exp(-βH(s))`` with ``H(s) = ∑<i,j>σᵢσⱼ``
"""
function classical_ising(eltype = ComplexF64, ::ℤ{1} = ℤ{1}, lattice = InfiniteChain(1);
                         beta = log(1 + sqrt(2)) / 2)
    t = [exp(beta) exp(-beta); exp(-beta) exp(beta)]

    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2)
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1

    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    return InfiniteMPO(TensorMap(complex(o), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2))
end

function classical_ising(eltype, ::ℤ₂, lattice = InfiniteChain(1);
                         beta = log(1 + sqrt(2)) / 2)
    x = cosh(beta)
    y = sinh(beta)

    sec = ℤ₂Space(0 => 1, 1 => 1)
    mpo = TensorMap(zeros, ComplexF64, sec * sec, sec * sec)
    blocks(mpo)[Irrep[ℤ₂](0)] = [2x^2 2x*y; 2x*y 2y^2]
    blocks(mpo)[Irrep[ℤ₂](1)] = [2x*y 2x*y; 2x*y 2x*y]

    return InfiniteMPO(mpo)
end

"""
    sixvertex(; a=1.0, b=1.0, c=1.0)

MPO for the six vertex model.
"""
function sixvertex(eltype = ComplexF64, ::ℤ{1} = ℤ{1}, lattice = InfiniteChain(1);
                   a = 1.0, b = 1.0, c = 1.0)
    d = [a 0 0 0
         0 c b 0
         0 b c 0
         0 0 0 a]
    return InfiniteMPO(permute(TensorMap(complex(d), ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2), (1, 2), (4, 3)))
end

function sixvertex(eltype, ::U₁, lattice = InfiniteChain(1);
                   a = 1.0, b = 1.0, c = 1.0)
    pspace = U1Space(-1 => 1, 1 => 1)
    mpo = TensorMap(zeros, eltype, pspace ⊗ pspace, pspace ⊗ pspace)
    blocks(mpo)[Irrep[U₁](0)] = [b c; c b]
    blocks(mpo)[Irrep[U₁](2)] = reshape([a], (1, 1))
    blocks(mpo)[Irrep[U₁](-2)] = reshape([a], (1, 1))
    return InfiniteMPO(permute(mpo, (1, 2), (4, 3)))
end

function sixvertex(eltype, ::CU₁, lattice = InfiniteChain(1);
                   a = 1.0, b = 1.0, c = 1.0)
    pspace = CU1Space(1 // 2 => 1)
    mpo = TensorMap(zeros, eltype, pspace ⊗ pspace, pspace ⊗ pspace)
    blocks(mpo)[Irrep[CU₁](0, 0)] = reshape([b + c], (1, 1))
    blocks(mpo)[Irrep[CU₁](0, 1)] = reshape([-b + c], (1, 1))
    blocks(mpo)[Irrep[CU₁](1, 2)] = reshape([a], (1, 1))
    return InfiniteMPO(permute(mpo, (1, 2), (4, 3)))
end

"""
    xxx(; J=1.0, spin=1)

MPO for the hamiltonian of the xxx Heisenberg model, defined by
    ``H = J(∑<i,j> X{i}X{j} + Y{i}Y{j} + Z{i}Z{j})``
"""
function xxx(eltype = ComplexF64, symmetry = ℤ{1}, lattice = InfiniteChain(1);
             J = 1.0, spin = 1)
    SS = sigma_exchange(eltype, symmetry; spin = spin)
    return @mpoham sum(J * SS{i, j} for (i, j) in nearest_neighbours(lattice))
end

"""
    xxz(; J=1.0, Δ=1.0, spin=1)

MPO for the hamiltonian of the xxx Heisenberg model, defined by
    ``H = J(∑<i,j> X{i}X{j} + Y{i}Y{j} + Δ * Z{i}Z{j})``
"""
function xxz(eltype = ComplexF64, symmetry = ℤ{1}, lattice = InfiniteChain(1);
             J = 1.0, Δ = 1.0, spin = 1)
    XX = sigma_xx(eltype, symmetry; spin = spin)
    YY = sigma_yy(eltype, symmetry; spin = spin)
    ZZ = sigma_zz(eltype, symmetry; spin = spin)
    return @mpoham sum(J * (XX{i, j} + YY{i, j} + Δ * ZZ{i, j})
                       for (i, j) in nearest_neighbours(lattice))
end

function xyz(eltype = ComplexF64, symmetry = ℤ{1}, lattice = InfiniteChain(1);
             Jx = 1.0, Jy = 1.0, Jz = 1.0, spin = 1)
    XX = sigma_xx(eltype, symmetry; spin = spin)
    YY = sigma_yy(eltype, symmetry; spin = spin)
    ZZ = sigma_zz(eltype, symmetry; spin = spin)
    return @mpoham sum(Jx * XX{i, j} + Jy * YY{i, j} + Jz * ZZ{i, j}
                       for (i, j) in nearest_neighbours(lattice))
end
