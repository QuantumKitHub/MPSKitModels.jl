#################
## Ising model ##
#################
"""
    transverse_field_ising(; J=1.0, h=1.0)

MPO for the hamiltonian of the transverse field Ising model, defined by
    ``H = -J(∑<i,j> X{i}X{j} + ∑<i> Z{i})``
"""
function transverse_field_ising(
    eltype=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
    J=1.0, h=0.5, spin=1 // 2
)
    XX = sigma_xx(eltype, symmetry; spin=spin)
    Z = sigma_z(eltype, symmetry; spin=spin)
    return @mpoham sum(-J * (XX{i,j} + h * Z{i}) for (i, j) in nearest_neighbours(lattice))
end

"""
    classical_ising(; beta=log(1+sqrt(2))/2)

MPO for the classical Ising partition function, defined by
    ``Z(β) = ∑ₛ exp(-βH(s))`` with ``H(s) = ∑<i,j>σᵢσⱼ``
"""
function classical_ising(
    eltype=ComplexF64, ::ℤ{1}=ℤ{1}, lattice=InfiniteChain(1);
    beta=log(1 + sqrt(2)) / 2
)
    t = [exp(beta) exp(-beta); exp(-beta) exp(beta)]

    r = eigen(t)
    nt = r.vectors * sqrt(Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2)
    O[1, 1, 1, 1] = 1
    O[2, 2, 2, 2] = 1

    @tensor o[-1 -2; -3 -4] := O[1 2; 3 4] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4]

    return InfiniteMPO(TensorMap(complex(o), ℂ^2 * ℂ^2, ℂ^2 * ℂ^2))
end

function classical_ising(
    eltype=ComplexF64, ::ℤ₂=ℤ₂, lattice=InfiniteChain(1);
    beta=log(1 + sqrt(2)) / 2
)
    x = cosh(beta)
    y = sinh(beta)

    sec = ℤ₂Space(0 => 1, 1 => 1)
    mpo = TensorMap(zeros, ComplexF64, sec * sec, sec * sec)
    blocks(mpo)[Irrep[ℤ₂](0)] = [2x^2 2x*y; 2x*y 2y^2]
    blocks(mpo)[Irrep[ℤ₂](1)] = [2x*y 2x*y; 2x*y 2x*y]

    return InfiniteMPO(mpo)
end