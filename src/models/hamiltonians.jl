#===========================================================================================
    Ising model
===========================================================================================#

"""
    transverse_field_ising([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                           [lattice::AbstractLattice]; J=1.0, g=1.0)

MPO for the hamiltonian of the one-dimensional
[Transverse-field Ising model](https://en.wikipedia.org/wiki/Transverse-field_Ising_model),
as defined by
```math
H = -J\\left(\\sum_{\\langle i,j \\rangle} \\sigma^z_i \\sigma^z_j + g \\sum_{i} \\sigma^x_i \\right)
```
where the ``\\sigma^i`` are the spin-1/2 Pauli operators. Possible values for the `symmetry`
are `Trivial`, `Z2Irrep` or `FermionParity`.

By default, the model is defined on an infinite chain with unit lattice spacing, with
`Trivial` symmetry and with `ComplexF64` entries of the tensors.
"""
function transverse_field_ising end
function transverse_field_ising(lattice::AbstractLattice; kwargs...)
    return transverse_field_ising(ComplexF64, Trivial, lattice; kwargs...)
end
function transverse_field_ising(S::Type{<:Sector},
                                lattice::AbstractLattice=InfiniteChain(1);
                                kwargs...)
    return transverse_field_ising(ComplexF64, S, lattice; kwargs...)
end
function transverse_field_ising(T::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return transverse_field_ising(T, Trivial, lattice; kwargs...)
end
function transverse_field_ising(T::Type{<:Number},
                                S::Type{<:Sector},
                                lattice::AbstractLattice;
                                kwargs...)
    throw(ArgumentError("`symmetry` must be either `Trivial`, `Z2Irrep` or `FermionParity`"))
end
function transverse_field_ising(T::Type{<:Number}=ComplexF64,
                                S::Union{Type{Trivial},Type{Z2Irrep}}=Trivial,
                                lattice::AbstractLattice=InfiniteChain(1);
                                J=1.0, g=1.0)
    ZZ = scale!(σᶻᶻ(T, S), -J)
    X = scale!(σˣ(T, S), g * -J)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return ZZ{i,j}
        end + sum(vertices(lattice)) do i
            return X{i}
        end
    end
end
function transverse_field_ising(T::Type{<:Number}, ::Type{fℤ₂},
                                lattice::AbstractLattice=InfiniteChain(1);
                                J=1.0, g=1.0)
    hop = add!(c_plusmin(T), c_minplus(T))
    sc = add!(c_plusplus(T), c_minmin(T))
    twosite = add!(hop, sc, J, -J)

    N = c_number(T)
    E = id(storagetype(N), space(N, 1))
    onesite = add!(N, E, -g * J, 2g * J)

    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return twosite{i,j}
        end + sum(vertices(lattice)) do i
            return onesite{i}
        end
    end
end

#===========================================================================================
    Kitaev model
===========================================================================================#

"""
    kitaev_model([elt::Type{<:Number}], [lattice::AbstractLattice];
                 t=1.0, mu=1.0, Delta=1.0)

MPO for the hamiltonian of the Kitaev model, as defined by
```math
H = \\sum_{\\langle i,j \\rangle} \\left(-\\frac{t}{2}(c_i^+ c_j^- + c_j^+c_i^-) + \\frac{Δ}{2}(c_i^+c_j^+ + c_j^-c_i^-) \\right) - \\mu \\sum_{i} c_i^+ c_i^-
```

By default, the model is defined on an infinite chain with unit lattice spacing and with `ComplexF64` entries of the tensors.
"""
function kitaev_model end
function kitaev_model(lattice::AbstractLattice; kwargs...)
    return kitaev_model(ComplexF64, lattice; kwargs...)
end
function kitaev_model(elt::Type{<:Number}=ComplexF64,
                      lattice::AbstractLattice=InfiniteChain(1);
                      t=1.0, mu=1.0, Delta=1.0)
    TB = rmul!(c_plusmin(elt) + c_minplus(elt), -t / 2)     # tight-binding term
    SC = rmul!(c_plusplus(elt) + c_minmin(elt), Delta / 2)  # superconducting term
    CP = rmul!(c_number(elt), -mu)                          # chemical potential term

    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return (TB + SC){i,j}
        end + sum(vertices(lattice)) do i
            return CP{i}
        end
    end
end

#===========================================================================================
    Heisenberg models
===========================================================================================#

"""
    heisenberg_XXX([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                           [lattice::AbstractLattice]; J=1.0, spin=1)

MPO for the hamiltonian of the isotropic Heisenberg model, as defined by
```math
H = J \\sum_{\\langle i,j \\rangle} \\vec{S}_i \\cdot \\vec{S}_j
```
where ``\\vec{S} = (S^x, S^y, S^z)``.

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
function heisenberg_XXX(T::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        J::Real=1.0, spin::Real=1)
    term = rmul!(S_exchange(T, symmetry; spin=spin), J)
    return @mpoham sum(nearest_neighbours(lattice)) do (i, j)
        return term{i,j}
    end
end

"""
    heisenberg_XXZ([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                   [lattice::AbstractLattice]; J=1.0, Delta=1.0, spin=1)

MPO for the hamiltonian of the XXZ Heisenberg model, as defined by
```math
H = J \\left( \\sum_{\\langle i,j \\rangle} S_i^x S_j^x + S_i^y S_j^y + \\Delta S_i^z S_j^z \\right)
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
function heisenberg_XXZ(elt::Type{<:Number}=ComplexF64,
                        symmetry::Type{<:Sector}=Trivial,
                        lattice::AbstractLattice=InfiniteChain(1);
                        J=1.0, Delta=1.0, spin=1)
    term = rmul!(S_xx(elt, symmetry; spin=spin), J) +
           rmul!(S_yy(elt, symmetry; spin=spin), J) +
           rmul!(S_zz(elt, symmetry; spin=spin), Delta * J)
    return @mpoham sum(nearest_neighbours(lattice)) do (i, j)
        return term{i,j}
    end
end

"""
    heisenberg_XYZ([elt::Type{<:Number}], [lattice::AbstractLattice];
        Jx=1.0, Jy=1.0, Jz=1.0, spin=1)

MPO for the hamiltonian of the XYZ Heisenberg model, defined by
```math
H = \\sum_{\\langle i,j \\rangle} \\left( J^x S_i^x S_j^x + J^y S_i^y S_j^y + J^z S_i^z S_j^z \\right)
```

By default, the model is defined on an infinite chain with unit lattice spacing and with `ComplexF64` entries of the tensors.
"""
function heisenberg_XYZ end
function heisenberg_XYZ(lattice::AbstractLattice; kwargs...)
    return heisenberg_XYZ(ComplexF64, lattice; kwargs...)
end
function heisenberg_XYZ(T::Type{<:Number}=ComplexF64,
                        lattice::AbstractLattice=InfiniteChain(1);
                        Jx=1.0, Jy=1.0, Jz=1.0, spin=1)
    term = rmul!(S_xx(T, Trivial; spin=spin), Jx) +
           rmul!(S_yy(T, Trivial; spin=spin), Jy) +
           rmul!(S_zz(T, Trivial; spin=spin), Jz)
    return @mpoham sum(nearest_neighbours(lattice)) do (i, j)
        return term{i,j}
    end
end

"""
    bilinear_biquadratic_model([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                               [lattice::AbstractLattice]; spin=1, J=1.0, θ=0.0)

MPO for the hamiltonian of the bilinear biquadratic Heisenberg model, as defined by
```math
H = J \\sum_{\\langle i,j \\rangle} \\left(\\cos(\\theta) \\vec{S}_i \\cdot \\vec{S}_j + \\sin(\\theta) \\left( \\vec{S}_i \\cdot \\vec{S}_j \\right)^2 \\right)
```
where ``\\vec{S} = (S^x, S^y, S^z)``.

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors.
"""
function bilinear_biquadratic_model end
function bilinear_biquadratic_model(lattice::AbstractLattice; kwargs...)
    return bilinear_biquadratic_model(ComplexF64, Trivial, lattice; kwargs...)
end
function bilinear_biquadratic_model(symmetry::Type{<:Sector},
                                    lattice::AbstractLattice=InfiniteChain(1); kwargs...)
    return bilinear_biquadratic_model(ComplexF64, symmetry, lattice; kwargs...)
end
function bilinear_biquadratic_model(elt::Type{<:Number}, lattice::AbstractLattice;
                                    kwargs...)
    return bilinear_biquadratic_model(elt, Trivial, lattice; kwargs...)
end
function bilinear_biquadratic_model(elt::Type{<:Number}=ComplexF64,
                                    symmetry::Type{<:Sector}=Trivial,
                                    lattice::AbstractLattice=InfiniteChain(1);
                                    spin=1, J=1.0, θ=0.0)
    return @mpoham sum(nearest_neighbours(lattice)) do (i, j)
        return J * cos(θ) * S_exchange(elt, symmetry; spin=spin){i,j} +
               J * sin(θ) * (S_exchange(elt, symmetry; spin=spin)^2){i,j}
    end
end

#===========================================================================================
    Potts models
===========================================================================================#
"""
    quantum_potts([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                  [lattice::AbstractLattice]; q=3, J=1.0, g=1.0)

MPO for the hamiltonian of the quantum Potts model, as defined by
```math
H = - J \\sum_{\\langle i,j \\rangle} Z_i^\\dagger Z_j + Z_i Z_j^\\dagger
- g \\sum_i (X_i + X_i^\\dagger)
```
where the operators ``Z`` and ``X`` are the ``q``-rotation operators satisfying
``Z^q = X^q = 1`` and ``ZX = \\omega XZ`` where ``\\omega = e^{2πi/q}``.
"""
function quantum_potts end
function quantum_potts(lattice::AbstractLattice; kwargs...)
    return quantum_potts(ComplexF64, Trivial, lattice; kwargs...)
end
function quantum_potts(symmetry::Type{<:Sector},
                       lattice::AbstractLattice=InfiniteChain(1); kwargs...)
    return quantum_potts(ComplexF64, symmetry, lattice; kwargs...)
end
function quantum_potts(elt::Type{<:Number}, lattice::AbstractLattice;
                       kwargs...)
    return quantum_potts(elt, Trivial, lattice; kwargs...)
end
function quantum_potts(elt::Type{<:Number}=ComplexF64,
                       symmetry::Type{<:Sector}=Trivial,
                       lattice::AbstractLattice=InfiniteChain(1);
                       q=3, J=1.0, g=1.0)
    return @mpoham sum(sum(nearest_neighbours(lattice)) do (i, j)
                           return -J * (potts_ZZ(elt, symmetry; q)^k){i,j}
                       end - sum(vertices(lattice)) do i
                             return g * (potts_field(elt, symmetry; q)^k){i}
                             end for k in 1:(q - 1))
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
H = -t \\sum_{\\langle i,j \\rangle} \\sum_{\\sigma} \\left( e_{i,\\sigma}^+ e_{j,\\sigma}^- + c_{i,\\sigma}^- c_{j,\\sigma}^+ \\right) + U \\sum_i n_{i,\\uparrow}n_{i,\\downarrow} - \\sum_i \\mu n_i
```
where ``\\sigma`` is a spin index that can take the values ``\\uparrow`` or ``\\downarrow``
and ``n`` is the fermionic number operator [`e_number`](@ref).

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors. If the `particle_symmetry` is not `Trivial`, a fixed particle number density `n` can be imposed.
"""
function hubbard_model end
function hubbard_model(lattice::AbstractLattice; kwargs...)
    return hubbard_model(ComplexF64, Trivial, Trivial, lattice; kwargs...)
end
function hubbard_model(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                       kwargs...)
    return hubbard_model(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
end
function hubbard_model(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return hubbard_model(elt, Trivial, Trivial, lattice; kwargs...)
end
function hubbard_model(T::Type{<:Number}=ComplexF64,
                       particle_symmetry::Type{<:Sector}=Trivial,
                       spin_symmetry::Type{<:Sector}=Trivial,
                       lattice::AbstractLattice=InfiniteChain(1);
                       t=1.0, U=1.0, mu=0.0, n::Integer=0)
    hopping = e⁺e⁻(T, particle_symmetry, spin_symmetry) +
              e⁻e⁺(T, particle_symmetry, spin_symmetry)
    interaction_term = nꜛnꜜ(T, particle_symmetry, spin_symmetry)
    N = e_number(T, particle_symmetry, spin_symmetry)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t * hopping{i,j}
        end +
        sum(vertices(lattice)) do i
            return U * interaction_term{i} - mu * N{i}
        end
    end
end

"""
    bose_hubbard_model([elt::Type{<:Number}], [symmetry::Type{<:Sector}],
                       [lattice::AbstractLattice];
                       cutoff, t, U, mu, n)

MPO for the hamiltonian of the Bose-Hubbard model, as defined by
```math
H = -t \\sum_{\\langle i,j \\rangle} \\left( a_{i}^+ a_{j}^- + a_{i}^- a_{j}^+ \\right) - \\sum_i \\mu N_i + \\frac{U}{2} \\sum_i N_i(N_i - 1).
```
where ``N`` is the bosonic number operator [`a_number`](@ref).

By default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with `ComplexF64` entries of the tensors. The Hilbert space is truncated such that at maximum of `cutoff` bosons can be at a single site. If the `symmetry` is not `Trivial`, a fixed particle number density `n` can be imposed.
"""
function bose_hubbard_model end
function bose_hubbard_model(lattice::AbstractLattice; kwargs...)
    return bose_hubbard_model(ComplexF64, Trivial, lattice; kwargs...)
end
function bose_hubbard_model(symmetry::Type{<:Sector},
                            lattice::AbstractLattice=InfiniteChain(1); kwargs...)
    return bose_hubbard_model(ComplexF64, symmetry, lattice; kwargs...)
end
function bose_hubbard_model(elt::Type{<:Number}=ComplexF64,
                            symmetry::Type{<:Sector}=Trivial,
                            lattice::AbstractLattice=InfiniteChain(1);
                            cutoff::Integer=5, t=1.0, U=1.0, mu=0.0, n::Integer=0)
    hopping_term = a_plusmin(elt, symmetry; cutoff=cutoff) +
                   a_minplus(elt, symmetry; cutoff=cutoff)
    N = a_number(elt, symmetry; cutoff=cutoff)
    interaction_term = contract_onesite(N, N - id(domain(N)))

    H = @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return -t * hopping_term{i,j}
        end +
        sum(vertices(lattice)) do i
            return U / 2 * interaction_term{i} - mu * N{i}
        end
    end

    if symmetry === Trivial
        iszero(n) || throw(ArgumentError("imposing particle number requires `U₁` symmetry"))
    elseif symmetry === U1Irrep
        isinteger(2n) ||
            throw(ArgumentError("`U₁` symmetry requires halfinteger particle number"))
        H = MPSKit.add_physical_charge(H, fill(U1Irrep(n), length(H)))
    else
        throw(ArgumentError("symmetry not implemented"))
    end

    return H
end

#===========================================================================================
    t-J models
===========================================================================================#

"""
    tj_model([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}],
                  [spin_symmetry::Type{<:Sector}], [lattice::AbstractLattice];
                  t, J, mu, slave_fermion::Bool=false, sigma::Bool=false)

MPO for the hamiltonian of the t-J model or the sigma t-J model, 
```math
H = H_t + J \\sum_{\\langle i,j \\rangle}(\\mathbf{S}_i \\cdot \\mathbf{S}_j - \\frac{1}{4} n_i n_j)
    - \\mu \\sum_i n_i
```
where the hopping term is
```math
H_t = -t \\sum_{\\langle i,j \\rangle, \\sigma}
    (\\tilde{e}^\\dagger_{i,\\sigma} \\tilde{e}_{j,\\sigma} + h.c.)
```
for the t-J model (with `sigma = false`), or 
```math
H = -t \\sum_{\\langle i,j \\rangle, \\sigma} \\sigma
    (\\tilde{e}^\\dagger_{i,\\sigma} \\tilde{e}_{j,\\sigma} + h.c.)
```
for the sigma t-J model (with `sigma = true`, introduced in https://doi.org/10.1038/srep02586).
In both cases, ``\\tilde{e}_{i,\\sigma}`` is the electron operator with spin ``\\sigma`` restrict to the no-double-occupancy subspace. 
"""
function tj_model end
function tj_model(lattice::AbstractLattice; kwargs...)
    return tj_model(ComplexF64, Trivial, Trivial, lattice; kwargs...)
end
function tj_model(particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector};
                  kwargs...)
    return tj_model(ComplexF64, particle_symmetry, spin_symmetry; kwargs...)
end
function tj_model(elt::Type{<:Number}, lattice::AbstractLattice; kwargs...)
    return tj_model(elt, Trivial, Trivial, lattice; kwargs...)
end
function tj_model(T::Type{<:Number}=ComplexF64,
                  particle_symmetry::Type{<:Sector}=Trivial,
                  spin_symmetry::Type{<:Sector}=Trivial,
                  lattice::AbstractLattice=InfiniteChain(1);
                  t=2.5, J=1.0, mu=0.0, slave_fermion::Bool=false, sigma::Bool=false)
    hopping = TJOperators.e_plusmin(T, particle_symmetry, spin_symmetry; slave_fermion,
                                    sigma) +
              TJOperators.e_minplus(T, particle_symmetry, spin_symmetry; slave_fermion,
                                    sigma)
    num = TJOperators.e_number(T, particle_symmetry, spin_symmetry; slave_fermion)
    heisenberg = TJOperators.S_exchange(T, particle_symmetry, spin_symmetry;
                                        slave_fermion) -
                 (1 / 4) * (num ⊗ num)
    return @mpoham begin
        sum(nearest_neighbours(lattice)) do (i, j)
            return (-t) * hopping{i,j} + J * heisenberg{i,j}
        end + sum(vertices(lattice)) do i
            return (-mu) * num{i}
        end
    end
end

# TODO: add (hardcore) bosonic t-J model (https://arxiv.org/abs/2409.15424)
