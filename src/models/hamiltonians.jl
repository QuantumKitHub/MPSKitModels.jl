#===========================================================================================
    Ising model
===========================================================================================#

"""
    transverse_field_ising([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                           [lattice::AbstractLattice]; J=1.0, g=1.0)

MPO for the hamiltonian of the
[Transverse-field Ising model](https://en.wikipedia.org/wiki/Transverse-field_Ising_model),
as defined by
```math
H = -J\\left(∑_{<i,j>} Z_i Z_j + g ∑_{<i>} X_i\\right)
```

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors.
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
    return -J * @mpoham begin 
        sum(σᶻᶻ(elt, symmetry){i,j} for (i, j) in nearest_neighbours(lattice)) +
        g * sum(σˣ(elt, symmetry){i} for i in vertices(lattice))
    end
end

function transverse_field_ising(elt::Type{<:Number}, ::Type{fℤ₂}, lattice::AbstractLattice;
                                J=1.0, g=1.0)
    H1 = kitaev_model(elt, lattice; t=2J, mu=-2 * g * J, Delta=2J)
    E = rmul!(id(Matrix{elt}, H1.pspaces[1]), -g * J)
    H2 = @mpoham sum(E{i} for i in vertices(lattice))
    return H1 + H2
end

#===========================================================================================
    Kitaev model
===========================================================================================#

"""
    kitaev_model([elt::Type{<:Number}], [lattice::AbstractLattice];
                 t=1.0, mu=1.0, Delta=1.0)

MPO for the hamiltonian of the Kitaev model, as defined by
```math
H = ∑_{<i,j>} \\left(-\\frac{t}{2}(c⁺ᵢcⱼ + c⁺ⱼcᵢ) + \\frac{Δ}{2}(c⁺ᵢc⁺ⱼ + cⱼcᵢ) \\right) - μ ∑_{<i>} c⁺ᵢcᵢ
```

By default, the model is defined on an infinite chain with unit lattice spacing and with `ComplexF64` entries of the tensors.
"""
function kitaev_model end
function kitaev_model(lattice::AbstractLattice; kwargs...)
    return kitaev_model(ComplexF64, lattice; kwargs...)
end
function kitaev_model(elt::Type{<:Number}=ComplexF64,
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
    heisenberg_XXX([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                           [lattice::AbstractLattice]; J=1.0, spin=1)

MPO for the hamiltonian of the isotropic Heisenberg model, as defined by
```math
H = J ∑_{<i,j>} S⃗ᵢ⋅S⃗ⱼ
```

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors.

See also [`heisenberg_XXZ`](@ref) and [`heisenberg_XYZ`](@ref).
"""
function heisenberg_XXX end
function heisenberg_XXX(lattice::AbstractLattice; kwargs...)
    return heisenberg_XXX(ComplexF64, Trivial, lattice; kwargs...)
end
function heisenberg_XXX(symmetry::Type{<:Sector}, lattice::AbstractLattice=InfiniteChain(1);
                        kwargs...)
    return heisenberg_XXX(ComplexF64, symmetry, lattice; kwargs...)
end
function heisenberg_XXX(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return heisenberg_XXX(elt, Trivial, lattice; kwargs...)
end
function heisenberg_XXX(elt::Type{<:Number}=ComplexF64, symmetry::Type{<:Sector}=Trivial, lattice::AbstractLattice=InfiniteChain(1);
                        J=1.0, spin=1)
    SS = S_exchange(elt, symmetry; spin=spin)
    return @mpoham sum(J * SS{i,j} for (i, j) in nearest_neighbours(lattice))
end

"""
    heisenberg_XXZ([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                   [lattice::AbstractLattice]; J=1.0, Delta=1.0, spin=1)

MPO for the hamiltonian of the XXZ Heisenberg model, as defined by
```math
H = J(∑_{<i,j>} XᵢXⱼ + YᵢYⱼ + Δ ZᵢZⱼ)
```

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors.
"""
function heisenberg_XXZ end
function heisenberg_XXZ(lattice::AbstractLattice; kwargs...)
    return heisenberg_XXZ(ComplexF64, Trivial, lattice; kwargs...)
end
function heisenberg_XXZ(symmetry::Type{<:Sector}, lattice::AbstractLattice=InfiniteChain(1);
                        kwargs...)
    return heisenberg_XXZ(ComplexF64, symmetry, lattice; kwargs...)
end
function heisenberg_XXZ(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return heisenberg_XXZ(elt, Trivial, lattice; kwargs...)
end
function heisenberg_XXZ(elt::Type{<:Number}=ComplexF64, symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
             J=1.0, Delta=1.0, spin=1)
    XX = S_xx(elt, symmetry; spin=spin)
    YY = S_yy(elt, symmetry; spin=spin)
    ZZ = S_zz(elt, symmetry; spin=spin)
    return @mpoham sum(J * (XX{i,j} + YY{i,j} + Delta * ZZ{i,j})
                    for (i, j) in nearest_neighbours(lattice))
end

"""
    heisenberg_XYZ([elt::Type{<:Number}], [lattice::AbstractLattice];
        Jx=1.0, Jy=1.0, Jz=1.0, spin=1)

MPO for the hamiltonian of the xyz Heisenberg model, defined by
```math
H = ∑_{<i,j>} (JˣXᵢXⱼ + JʸYᵢYⱼ + JᶻZᵢZⱼ)
```

By default, the model is defined on an infinite chain with unit lattice spacing and with `ComplexF64` entries of the tensors.
"""
function heisenberg_XYZ end
function heisenberg_XYZ(lattice::AbstractLattice; kwargs...)
    return heisenberg_XYZ(ComplexF64, lattice; kwargs...)
end
function heisenberg_XYZ(elt::Type{<:Number}=ComplexF64, 
                        lattice::AbstractLattice=InfiniteChain(1);
                        Jx=1.0, Jy=1.0, Jz=1.0, spin=1)
    XX = S_xx(elt, symmetry; spin=spin)
    YY = S_yy(elt, symmetry; spin=spin)
    ZZ = S_zz(elt, symmetry; spin=spin)
    return @mpoham sum(Jx * XX{i,j} + Jy * YY{i,j} + Jz * ZZ{i,j}
                       for (i, j) in nearest_neighbours(lattice))
end

"""
    bilinear_biquadratic_model([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                               [lattice::AbstractLattice]; spin=1, J=1.0, θ=0.0)

MPO for the hamiltonian of the bilinear biquadratic Heisenberg model, as defined by
```math
H = J ∑_{<i,j>} (\\cos(θ) S⃗ᵢ⋅S⃗ⱼ + \\sin(θ) (⃗S⃗ᵢ⋅S⃗ⱼ)²)
```

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors.
"""
function bilinear_biquadratic_model end
function bilinear_biquadratic_model(lattice::AbstractLattice; kwargs...)
    return bilinear_biquadratic_model(ComplexF64, Trivial, lattice; kwargs...)
end
function bilinear_biquadratic_model(symmetry::Type{<:Sector}, lattice::AbstractLattice=InfiniteChain(1);
                        kwargs...)
    return bilinear_biquadratic_model(ComplexF64, symmetry, lattice; kwargs...)
end
function bilinear_biquadratic_model(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return bilinear_biquadratic_model(elt, Trivial, lattice; kwargs...)
end
function bilinear_biquadratic_model(elt::Type{<:Number}=ComplexF64,
                                    symmetry::Type{<:Sector}=Trivial,
                                    lattice::AbstractLattice=InfiniteChain(1);
                                    spin=1, J=1.0, θ=0.0)
    SS = S_exchange(elt, symmetry; spin=spin)
    return @mpoham begin
        sum(J * (cos(θ) * SS{i,j} + sin(θ) * SS{i,j} * SS{i,j})
            for (i, j) in nearest_neighbours(lattice))
    end
end

#===========================================================================================
    Hubbard models
===========================================================================================#

"""
    hubbard_model([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}],
                  [spin_symmetry::Type{<:Sector}], [lattice::AbstractLattice];
                  t, U, mu, n)

MPO for the hamiltonian of the Hubbard model, as defined by
```math
H = -t∑_{<i,j>} (c⁺_{σ,i}c⁻_{σ,j} + c⁻_{σ,i}c⁺_{σ,j}) + U ∑_i n_{i,↑}n_{i,↓} - ∑_i μnᵢ
```

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors. If the `particle_symmetry` is not `Trivial`, a fixed particle number density `n` can be imposed.
"""
function hubbard_model end
hubbard_model(lattice::AbstractLattice; kwargs...) =
    hubbard_model(ComplexF64, Trivial, Trivial, lattice; kwargs...)
hubbard_model(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector}; kwargs...) =
    hubbard_model(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
hubbard_model(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...) =
    hubbard_model(elt, Trivial, Trivial, lattice; kwargs...)
function hubbard_model(elt::Type{<:Number}=ComplexF64,
                       particle_symmetry::Type{<:Sector}=Trivial,
                       spin_symmetry::Type{<:Sector}=Trivial,
                       lattice::AbstractLattice=InfiniteChain(1);
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
    bose_hubbard_model([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                       [lattice::AbstractLattice];
                       cutoff, t, U, mu, n)

MPO for the hamiltonian of the Bose-Hubbard model, as defined by
```math
H = -t∑_{<i,j>} (a⁺_{i}a⁻_{j} + a⁻_{i}a⁺_{j}) - ∑_i μnᵢ + U / 2 ∑_i nᵢ(nᵢ - 1)
```

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors. The Hilbert space is truncated such that at maximum of `cutoff` bosons can be at a single site. If the `symmetry` is not `Trivial`, a fixed particle number density `n` can be imposed.
"""
function bose_hubbard_model(elt::Type{<:Number}=ComplexF64,
                            symmetry::Type{<:Sector}=Trivial,
                            lattice::AbstractLattice=InfiniteChain(1);
                            cutoff::Integer=5, t=1.0, U=1.0, mu=0.0, n::Integer=0)
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

    if symmetry === Trivial
        iszero(n) ||
            throw(ArgumentError("imposing particle number requires `U₁` symmetry"))
    elseif symmetry === U1Irrep
        isinteger(2n) ||
            throw(ArgumentError("`U₁` symmetry requires halfinteger particle number"))
        H = MPSKit.add_physical_charge(H, fill(U1Irrep(n), length(H)))
    else
        throw(ArgumentError("symmetry not implemented"))
    end

    return H
end
