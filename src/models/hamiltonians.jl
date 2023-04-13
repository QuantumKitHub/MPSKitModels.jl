"""
    transverse_field_ising(; J=1.0, hx=1.0, hz=0.0, spin=1//2)

MPO for the hamiltonian of the transverse field Ising model, defined by
    ``H = -J(∑_{<i,j>} Z_i Z_j + ∑_{<i>} hx X_i + hz Z_i)``
"""
function transverse_field_ising(eltype=ComplexF64, symmetry=ℤ{1},
                                lattice=InfiniteChain(1);
                                J=1.0, hx=0.5, hz=0.0, spin=1 // 2)
    ZZ = sigma_zz(eltype, symmetry; spin=spin)
    X = sigma_x(eltype, symmetry; spin=spin)
    if symmetry != ℤ{1}
        @assert hz == zero(hz) "parameters and symmetry incompatible"
        return @mpoham sum(-J * (ZZ{i,j} + hx * X{i})
                       for (i, j) in nearest_neighbours(lattice))
    else
        Z = sigma_z(eltype, symmetry; spin=spin)
        return @mpoham sum(-J * (ZZ{i,j} + hx * X{i} + hz * Z{i})
                       for (i, j) in nearest_neighbours(lattice))
    end
end