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
                                J=1.0, hx=1.0, hz=0.0, spin=1 // 2)
    ZZ = sigma_zz(eltype, symmetry; spin=spin) * 4
    X = sigma_x(eltype, symmetry; spin=spin) * 2
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

function free_fermion_ising(eltype=ComplexF64, lattice=InfiniteChain(1);
                    J=1.0, hx=1.0)
    hopping_term = c⁺c⁻(eltype) + c⁻c⁺(eltype) + c⁺c⁺(eltype) + c⁻c⁻(eltype)
    interaction_term = 2 * c_number(eltype)
    interaction_term -= id(domain(interaction_term))
    
    return @mpoham sum(-J * (hopping_term{i,j} + hx * interaction_term{i}) for (i,j) in nearest_neighbours(lattice))
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
        H += @mpoham sum(hz * sigma_z(eltype, symmetry; spin=spin){i}
                         for i in vertices(lattice))
    end
    return H
end
function xxz(eltype, ::Type{U₁}, lattice=InfiniteChain(1);
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

"""
    bilinear_biquadratic_heisenberg(elt, symmetry, lattice;
        spin=1, J=1.0, θ=0.0)

MPO for the hamiltonian of the bilinear biquadratic Heisenberg model, defined by

``H = J ∑_{<i,j>} (\\cos(θ) ⃗σᵢ ⃗σⱼ + \\sin(θ) (⃗σᵢ ⃗σⱼ)²)``
"""
function bilinear_biquadratic_heisenberg(elt=ComplexF64, symmetry=ℤ{1},
                                         lattice=InfiniteChain(1); spin=1, J=1.0, θ=0.0)
    SS = sigma_exchange(elt, symmetry; spin=spin)
    return @mpoham begin
        sum(J * (cos(θ) * SS{i,j} + sin(θ) * SS{i,j} * SS{i,j})
            for (i, j) in nearest_neighbours(lattice))
    end
end

#===========================================================================================
    Hubbard models
===========================================================================================#

"""
    hubbard_model(eltype, symmetry, lattice;
                       cutoff, t, U, mu, particle_number)

MPO for the hamiltonian of the Bose-Hubbard model, defined by a nearest-neighbour hopping
term, an on-site interaction term and a chemical potential.

``H = -t∑_{<i,j>} (c⁺_{σ,i}c⁻_{σ,j} + c⁻_{σ,i}c⁺_{σ,j}) + U ∑_i n_{i,↑}n_{i,↓} - ∑_i μnᵢ``
"""
function hubbard_model(elt=ComplexF64, particle_symmetry=ℤ₁, spin_symmetry=ℤ₁,
                       lattice=InfiniteChain(1);
                       t=1.0, U=1.0, mu=0.0, n::Integer=0)
    hopping_term = e⁺e⁻(elt, particle_symmetry, spin_symmetry) +
                   e⁻e⁺(elt, particle_symmetry, spin_symmetry)
    interaction_term = nꜛnꜜ(elt, particle_symmetry, spin_symmetry)
    N = e_number(elt, particle_symmetry, spin_symmetry)

    @mpoham begin
        H = sum(-t * hopping_term{i,j} + U * interaction_term{i} - mu * N{i}
                for (i, j) in nearest_neighbours(lattice))
    end

    return H
end

"""
    bose_hubbard_model(eltype, symmetry, lattice;
                       cutoff, t, U, mu, particle_number)

MPO for the hamiltonian of the Bose-Hubbard model, defined by a nearest-neighbour hopping
term, an on-site interaction term and a chemical potential.

``H = -t∑_{<i,j>} (a⁺_{i}a⁻_{j} + a⁻_{i}a⁺_{j}) - ∑_i μnᵢ + U / 2 ∑_i nᵢ(nᵢ - 1)``
"""
function bose_hubbard_model(eltype=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
                            cutoff=5, t=1.0, U=1.0, mu=0.0, n::Integer=0)
    hopping_term = contract_twosite(a_plus(cutoff, eltype, symmetry; side=:L),
                                    a_min(cutoff, eltype, symmetry; side=:R)) +
                   contract_twosite(a_min(cutoff, eltype, symmetry; side=:L),
                                    a_plus(cutoff, eltype, symmetry; side=:R))
    N = contract_onesite(a_plus(cutoff, eltype, symmetry; side=:L),
                         a_min(cutoff, eltype, symmetry; side=:R))
    interaction_term = contract_onesite(N, N - id(domain(N)))

    @mpoham begin
        H = sum(-t * hopping_term{i,j} + U / 2 * interaction_term{i} - mu * N{i}
                for (i, j) in nearest_neighbours(lattice))
    end

    if symmetry == ℤ{1}
        iszero(n) ||
            throw(ArgumentError("imposing particle number requires `U₁` symmetry"))
    elseif symmetry == U₁
        isinteger(2n) ||
            throw(ArgumentError("`U₁` symmetry requires halfinteger particle number"))
        H = MPSKit.add_physical_charge(H, fill(U1Irrep(n), length(H)))
    else
        throw(ArgumentError("symmetry not implemented"))
    end

    return H
end
