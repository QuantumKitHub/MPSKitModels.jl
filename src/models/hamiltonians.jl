#===========================================================================================
    Ising model
===========================================================================================#

"""
    transverse_field_ising(; J=1.0, g=1.0, spin=1//2)

MPO for the hamiltonian of the
[Transverse-field Ising model](https://en.wikipedia.org/wiki/Transverse-field_Ising_model),
as defined by
```math
    H = -J\\left(∑_{<i,j>} Z_i Z_j + g ∑_{<i>} X_i\\right)
```
"""
function transverse_field_ising end
function transverse_field_ising(lattice::AbstractLattice; kwargs...)
    return transverse_field_ising(ComplexF64, Trivial, lattice; kwargs...)
end
function transverse_field_ising(symmetry::Type{<:Sector},
                                lattice::AbstractLattice=InfiniteChain(1);
                                kwargs...)
    return transverse_field_ising(ComplexF64, symmetry, lattice; kwargs...)
end
function transverse_field_ising(elt::Type{<:Number}, lattice::AbstractLattice;
                                kwargs...)
    return transverse_field_ising(elt, Trivial, lattice; kwargs...)
end
function transverse_field_ising(elt::Type{<:Number}=ComplexF64,
                                symmetry::Type{<:Sector}=Trivial,
                                lattice::AbstractLattice=InfiniteChain(1); J=1.0, g=1.0)
    ZZ = rmul!(sigma_zz(elt, symmetry; spin=1 // 2), 4)
    X = rmul!(sigma_x(elt, symmetry; spin=1 // 2), 2)

    return -J * (@mpoham sum(ZZ{i,j} for (i, j) in nearest_neighbours(lattice)) +
                         g * sum(X{i} for i in vertices(lattice)))
end

function transverse_field_ising(elt::Type{<:Number}, ::Type{fℤ₂}, lattice::AbstractLattice;
                                J=1.0, g=1.0)
    H1 = kitaev_chain(elt, lattice; t=2J, mu=-2*g*J, Delta=2J)
    E = rmul!(id(Matrix{elt}, H1.pspaces[1]), -g*J)
    H2 = @mpoham sum(E{i} for i in vertices(lattice))
    return H1 + H2
end

#===========================================================================================
    Kitaev model
===========================================================================================#

"""
    kitaev_chain(; t=1.0, mu=1.0, Delta=1.0)

MPO for the hamiltonian of the Kitaev chain, as defined by
```math
    H = ∑_{<i,j>} \\left(-\\frac{t}{2}(c†ᵢcⱼ + c†ⱼcᵢ) + \\frac{Δ}{2}(c†ᵢc†ⱼ + cⱼcᵢ) \\right) - μ ∑_{<i>} c†ᵢcᵢ
```
"""
function kitaev_chain end
function kitaev_chain(lattice::AbstractLattice=InfiniteChain(1); kwargs...)
    return kitaev_chain(ComplexF64, lattice; kwargs...)
end
function kitaev_chain(elt::Type{<:Number}=ComplexF64,
                      lattice::AbstractLattice=InfiniteChain(1); t=1.0, mu=1.0, Delta=1.0)
    # tight-binding term
    TB = rmul!(c_plusmin(elt) + c_minplus(elt), -t / 2)
    # chemical potential term
    CP = rmul!(c_number(elt), -mu)
    # superconducting term
    SC = rmul!(c_plusplus(elt) + c_minmin(elt), Delta / 2)

    return @mpoham sum(TB{i,j} + SC{i,j} for (i, j) in nearest_neighbours(lattice)) +
                   sum(CP{i} for i in vertices(lattice))
end

#===========================================================================================
    Heisenberg models
===========================================================================================#

"""
    xxx(; J=1.0, spin=1)

MPO for the hamiltonian of the xxx Heisenberg model, defined by
    ``H = J(∑_{<i,j>} X_i X_j + Y_i Y_j + Z_i Z_j)``
"""
function xxx(elt=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
             J=1.0, spin=1)
    SS = sigma_exchange(elt, symmetry; spin=spin)
    return @mpoham sum(J * SS{i,j} for (i, j) in nearest_neighbours(lattice))
end

"""
    xxz(; J=1.0, Δ=1.0, spin=1)

MPO for the hamiltonian of the xxz Heisenberg model, defined by
    ``H = J(∑_{<i,j>} X_i X_j + Y_i Y_j + Δ Z_i Z_j)``
"""
function xxz(elt=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
             J=1.0, Δ=1.0, spin=1, hz=0.0)
    XX = sigma_xx(elt, symmetry; spin=spin)
    YY = sigma_yy(elt, symmetry; spin=spin)
    ZZ = sigma_zz(elt, symmetry; spin=spin)
    H = @mpoham sum(J * (XX{i,j} + YY{i,j} + Δ * ZZ{i,j})
                    for (i, j) in nearest_neighbours(lattice))
    if !iszero(hz)
        @assert symmetry !== SU₂
        H += @mpoham sum(hz * sigma_z(elt, symmetry; spin=spin){i}
                         for i in vertices(lattice))
    end
    return H
end
function xxz(elt, ::Type{U₁}, lattice=InfiniteChain(1);
             J=1.0, Δ=1.0, spin=1, hz=0.0)
    plusmin = sigma_plusmin(elt, U₁; spin=spin)
    minplus = sigma_minplus(elt, U₁; spin=spin)
    ZZ = sigma_zz(elt, U₁; spin=spin)
    H = @mpoham sum(J * (plusmin{i,j} + minplus{i,j} + Δ * ZZ{i,j})
                    for (i, j) in nearest_neighbours(lattice))
    if !iszero(hz)
        H += @mpoham sum(hz * sigma_z(elt, U₁; spin=spin){i}
                         for i in vertices(lattice))
    end
    return H
end

"""
    xyz(; Jx=1.0, Jy=1.0, Jz=1.0, spin=1)

MPO for the hamiltonian of the xyz Heisenberg model, defined by
    ``H = J(∑_{<i,j>} J_x X_{i}X_{j} + J_y Y_{i}Y_{j} + J_z Z_{i}Z_{j})``
"""
function xyz(elt=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
             Jx=1.0, Jy=1.0, Jz=1.0, spin=1)
    XX = sigma_xx(elt, symmetry; spin=spin)
    YY = sigma_yy(elt, symmetry; spin=spin)
    ZZ = sigma_zz(elt, symmetry; spin=spin)
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
    hubbard_model(elt, symmetry, lattice;
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
    bose_hubbard_model(elt, symmetry, lattice;
                       cutoff, t, U, mu, particle_number)

MPO for the hamiltonian of the Bose-Hubbard model, defined by a nearest-neighbour hopping
term, an on-site interaction term and a chemical potential.

``H = -t∑_{<i,j>} (a⁺_{i}a⁻_{j} + a⁻_{i}a⁺_{j}) - ∑_i μnᵢ + U / 2 ∑_i nᵢ(nᵢ - 1)``
"""
function bose_hubbard_model(elt=ComplexF64, symmetry=ℤ{1}, lattice=InfiniteChain(1);
                            cutoff=5, t=1.0, U=1.0, mu=0.0, n::Integer=0)
    hopping_term = contract_twosite(a_plus(cutoff, elt, symmetry; side=:L),
                                    a_min(cutoff, elt, symmetry; side=:R)) +
                   contract_twosite(a_min(cutoff, elt, symmetry; side=:L),
                                    a_plus(cutoff, elt, symmetry; side=:R))
    N = contract_onesite(a_plus(cutoff, elt, symmetry; side=:L),
                         a_min(cutoff, elt, symmetry; side=:R))
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
