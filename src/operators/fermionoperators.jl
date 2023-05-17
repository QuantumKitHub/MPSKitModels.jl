#===========================================================================================
    Spinless fermions
===========================================================================================#

"""
    c_plus(elt=ComplexF64, ::Type{fℤ₂}; side=:L)

fermionic creation operator.
"""
function c_plus(elt=ComplexF64; side=:L)
    if side === :L
        pspace = Vect[fℤ₂](0 => 1, 1 => 1)
        vspace = Vect[fℤ₂](1 => 1)
        c⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(c⁺)[fℤ₂(1)] .= one(elt)
    elseif side === :R
        C = c_plus(elt; side=:L)
        F = isometry(flip(space(C, 3)), space(C, 3))
        @plansor c⁺[-1 -2; -3] := C[-2; 1 2] * τ[1 2; 3 -3] * F[-1; 3]
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return c⁺
end
const c⁺ = c_plus

"""
    c_min(elt=ComplexF64, ::Type{fℤ₂}; side=:L)

fermionic creation operator.
"""
function c_min(elt=ComplexF64; side=:L)
    if side === :L
        C = c_plus(elt; side=:L)'
        F = isometry(flip(space(C, 2)), space(C, 2))
        @plansor c⁻[-1; -2 -3] := C[-1 1; -2] * F[-3; 1]
    elseif side === :R
        c⁻ = permute(c_plus(elt; side=:L)', (2, 1), (3,))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return c⁻
end
const c⁻ = c_min

c_plusmin(elt=ComplexF64) = contract_twosite(c⁺(elt; side=:L), c⁻(elt; side=:R))
const c⁺c⁻ = c_plusmin
c_minplus(elt=ComplexF64) = contract_twosite(c⁻(elt; side=:L), c⁺(elt; side=:R))
const c⁻c⁺ = c_minplus
c_plusplus(elt=ComplexF64) = contract_twosite(c⁺(elt; side=:L), c⁺(elt; side=:R))
const c⁺c⁺ = c_plusplus
c_minmin(elt=ComplexF64) = contract_twosite(c⁻(elt; side=:L), c⁻(elt; side=:R))
const c⁻c⁻ = c_minmin

"""
    c_number(elt=ComplexF64, ::Type{fℤ₂}=fℤ₂)

fermionic number operator.
"""
function c_number(elt=ComplexF64)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    n = TensorMap(zeros, elt, pspace ← pspace)
    blocks(n)[fℤ₂(1)] .= one(elt)
    return n
end

#===========================================================================================
    spin 1/2 fermions
===========================================================================================#

"""
    e_plus(elt=Complexf64, particle_symmetry, spin_symmetry; side=:L)

creation operator for electron-like fermions.
"""
function e_plus(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁; side=:L)
    if side == :L
        if spin_symmetry === ℤ₁
            if particle_symmetry === ℤ₁
                pspace = Vect[fℤ₂](0 => 2, 1 => 2)
                vspace = Vect[fℤ₂](1 => 2)
                e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
                reshape(view(blocks(e⁺)[fℤ₂(1)], :, :), 2, 2, 2)[1, 1, 1] = one(elt)
                reshape(view(blocks(e⁺)[fℤ₂(1)], :, :), 2, 2, 2)[2, 2, 1] = one(elt)
                reshape(view(blocks(e⁺)[fℤ₂(0)], :, :), 2, 2, 2)[2, 1, 2] = -one(elt)
                reshape(view(blocks(e⁺)[fℤ₂(0)], :, :), 2, 2, 2)[2, 2, 1] = one(elt)
            elseif particle_symmetry === U₁
                pspace = Vect[FermionParity ⊠ Irrep[U₁]]((0, 0) => 1, (1, 1) => 2,
                                                         (0, 2) => 1)
                vspace = Vect[FermionParity ⊠ Irrep[U₁]]((1, 1) => 2)
                e⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
                blocks(e⁺)[fℤ₂(1) ⊠ U₁(1)][1, 1] = one(elt)
                blocks(e⁺)[fℤ₂(1) ⊠ U₁(1)][2, 2] = one(elt)
                blocks(e⁺)[fℤ₂(0) ⊠ U₁(2)][1, 2] = one(elt)
                blocks(e⁺)[fℤ₂(0) ⊠ U₁(2)][1, 3] = one(elt)
                
            elseif particle_symmetry === SU₂
                error("tba")
                # pspace = Vect[fSU₂](0 => 2, 1 // 2 => 1)
                # vspace = Vect[fSU₂](1 => 2)
            else
                throw(ArgumentError("unknown particle symmetry"))
            end
        else
            throw(ArgumentError("unknown spin symmetry"))
        end
    else
        C = e_plus(elt, particle_symmetry, spin_symmetry; side=:L)
        F = isometry(flip(space(C, 3)), space(C, 3))
        @plansor e⁺[-1 -2; -3] := C[-2; 1 2] * τ[1 2; 3 -3] * F[-1; 3]
    end
    return e⁺
end
const e⁺ = e_plus

"""
    e_min(elt=Complexf64, particle_symmetry, spin_symmetry; side=:L)

annihilation operator for electron-like fermions.
"""
function e_min(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁; side=:L)
    if side === :L
        E = e_plus(elt, particle_symmetry, spin_symmetry; side=:L)'
        F = isomorphism(flip(space(E, 2)), space(E, 2))
        @plansor e⁻[-1; -2 -3] := E[-1 1; -2] * F[-3; 1]
    elseif side === :R
        e⁻ = permute(e_plus(elt, particle_symmetry, spin_symmetry; side=:L)', (2, 1), (3,))
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return e⁻
end
const e⁻ = e_min

function e_plusmin(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁)
    return contract_twosite(e⁺(elt, particle_symmetry, spin_symmetry; side=:L),
                            e⁻(elt, particle_symmetry, spin_symmetry; side=:R))
end
const e⁺e⁻ = e_plusmin
function e_minplus(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁)
    return contract_twosite(e⁻(elt, particle_symmetry, spin_symmetry; side=:L),
                            e⁺(elt, particle_symmetry, spin_symmetry; side=:R))
end
const e⁻e⁺ = e_minplus

"""
    e_number(elt=ComplexF64, particle_symmetry=fℤ₂, spin_symmetry=ℤ₁)

number operator for electron-like fermions.
"""
function e_number(elt=ComplexF64, particle_symmetry=ℤ₁, spin_symmetry=ℤ₁)
    if spin_symmetry === ℤ₁
        if particle_symmetry === ℤ₁
            pspace = Vect[fℤ₂](0 => 2, 1 => 2)
            n = TensorMap(zeros, elt, pspace ← pspace)
            blocks(n)[fℤ₂(1)][1, 1] = 1
            blocks(n)[fℤ₂(1)][2, 2] = 1
            blocks(n)[fℤ₂(0)][2, 2] = 2
        elseif particle_symmetry === U₁
            pspace = Vect[FermionParity ⊠ Irrep[U₁]]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
            n = TensorMap(zeros, elt, pspace ← pspace)
            blocks(n)[fℤ₂(1) ⊠ U₁(1)][1, 1] = 1
            blocks(n)[fℤ₂(1) ⊠ U₁(1)][2, 2] = 1
            blocks(n)[fℤ₂(0) ⊠ U₁(2)][1, 1] = 2
        elseif particle_symmetry === fSU₂
            error("tba")
            # pspace = Vect[fSU₂](0 => 2, 1 // 2 => 1)
            # vspace = Vect[fSU₂](1 => 2)
        else
            throw(ArgumentError("unknown particle symmetry"))
        end
    else
        throw(ArgumentError("unknown spin symmetry"))
    end
    
    return n
end

function e_number_updown(elt=ComplexF64, particle_symmetry=ℤ₁, spin_symmetry=ℤ₁)
    if spin_symmetry === ℤ₁
        if particle_symmetry === ℤ₁
            pspace = Vect[fℤ₂](0 => 2, 1 => 2)
            n = TensorMap(zeros, elt, pspace ← pspace)
            blocks(n)[fℤ₂(0)][2, 2] = 1
        elseif particle_symmetry === U₁
            pspace = Vect[FermionParity ⊠ Irrep[U₁]]((0, 0) => 1, (1, 1) => 2, (0, 2) => 1)
            n = TensorMap(zeros, elt, pspace ← pspace)
            blocks(n)[fℤ₂(0) ⊠ U₁(2)][1, 1] = 1
        elseif particle_symmetry === fSU₂
            error("tba")
            # pspace = Vect[fSU₂](0 => 2, 1 // 2 => 1)
            # vspace = Vect[fSU₂](1 => 2)
        else
            throw(ArgumentError("unknown particle symmetry"))
        end
    else
        throw(ArgumentError("unknown spin symmetry"))
    end

    return n
end
const nꜛnꜜ = e_number_updown