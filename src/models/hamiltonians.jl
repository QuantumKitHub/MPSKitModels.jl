#===========================================================================================
    Ising model
===========================================================================================#

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

#===========================================================================================
    Heisenberg models
===========================================================================================#

"""
    xxx(; J=1.0, spin=1)

MPO for the hamiltonian of the xxx Heisenberg model, defined by
    ``H = J(∑_{<i,j>} X_i X_j + Y_i Y_j + Z_i Z_j)``
"""
function xxx(eltype=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
             J=1.0, spin=1)
    SS = sigma_exchange(eltype, symmetry; spin=spin)
    return @mpoham sum(J * SS{i,j} for (i, j) in nearest_neighbours(lattice))
end

"""
    xxz(; J=1.0, Δ=1.0, spin=1)

MPO for the hamiltonian of the xxz Heisenberg model, defined by
    ``H = J(∑_{<i,j>} X_i X_j + Y_i Y_j + Δ Z_i Z_j)``
"""
function xxz(eltype=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
             J=1.0, Δ=1.0, spin=1, hz=0.0)
    XX = sigma_xx(eltype, symmetry; spin=spin)
    YY = sigma_yy(eltype, symmetry; spin=spin)
    ZZ = sigma_zz(eltype, symmetry; spin=spin)
    H = @mpoham sum(J * (XX{i,j} + YY{i,j} + Δ * ZZ{i,j})
                       for (i, j) in nearest_neighbours(lattice))
    if !iszero(hz)
        @assert symmetry !== SU₂
        H += @mpoham sum(hz * sigma_z(eltype, symmetry; spin=spin){i} for i in vertices(lattice))
    end
    return H
end
function xxz(eltype=ComplexF64, ::Type{U₁}=U₁, lattice=InfiniteChain(1);
             J=1.0, Δ=1.0, spin=1, hz=0.0)
    plusmin = sigma_plusmin(eltype, U₁; spin=spin)
    minplus = sigma_minplus(eltype, U₁; spin=spin)
    ZZ = sigma_zz(eltype, U₁; spin=spin)
    H = @mpoham sum(J * (plusmin{i,j} + minplus{i,j} + Δ * ZZ{i,j})
                    for (i, j) in nearest_neighbours(lattice))
    if !iszero(hz)
        H += @mpoham sum(hz * sigma_z(eltype, U₁; spin=spin){i}
                         for i in vertices(lattice))
    end
    return H
end

"""
    xyz(; Jx=1.0, Jy=1.0, Jz=1.0, spin=1)

MPO for the hamiltonian of the xyz Heisenberg model, defined by
    ``H = J(∑_{<i,j>} J_x X_{i}X_{j} + J_y Y_{i}Y_{j} + J_z Z_{i}Z_{j})``
"""
function xyz(eltype=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
             Jx=1.0, Jy=1.0, Jz=1.0, spin=1)
    XX = sigma_xx(eltype, symmetry; spin=spin)
    YY = sigma_yy(eltype, symmetry; spin=spin)
    ZZ = sigma_zz(eltype, symmetry; spin=spin)
    return @mpoham sum(Jx * XX{i,j} + Jy * YY{i,j} + Jz * ZZ{i,j}
                       for (i, j) in nearest_neighbours(lattice))
end
