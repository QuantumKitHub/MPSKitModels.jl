_pauliterm(spin, i, j) = sqrt((spin + 1) * (i + j - 1) - i * j) / 2.0

"""
    spinmatrices(spin [, eltype])

the spinmatrices according to https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins
"""
function spinmatrices(s::Union{Rational{Int},Int}, elt=ComplexF64)
    N = Int(2 * s)

    Sx = zeros(elt, N + 1, N + 1)
    Sy = zeros(elt, N + 1, N + 1)
    Sz = zeros(elt, N + 1, N + 1)

    for row in 1:(N + 1)
        for col in 1:(N + 1)
            term = _pauliterm(s, row, col)

            if (row + 1 == col)
                Sx[row, col] += term
                Sy[row, col] -= 1im * term
            end

            if (row == col + 1)
                Sx[row, col] += term
                Sy[row, col] += 1im * term
            end

            if (row == col)
                Sz[row, col] += s + 1 - row
            end
        end
    end
    return Sx, Sy, Sz, one(Sx)
end

"""
    sigma_x([eltype [, symmetry]]; spin=S)

spin `S` operator along the x-axis.

See also [`σˣ`](@ref)
"""
function sigma_x(elt::Type{<:Number}=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; spin=1 // 2)
    sigma_x_mat, _, _ = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(sigma_x_mat, 1))
    return TensorMap(sigma_x_mat, pspace ← pspace)
end

function sigma_x(elt::Type{<:Number}, ::Type{ℤ{2}}; spin=1 // 2)
    @assert spin == 1 // 2
    pspace = Z2Space(0 => 1, 1 => 1)
    σˣ = TensorMap(zeros, elt, pspace, pspace)
    blocks(σˣ)[Z2Irrep(0)] .= one(elt) / 2
    blocks(σˣ)[Z2Irrep(1)] .= -one(elt) / 2
    return σˣ 
end

# function sigma_x(elt::Type{<:Number}, symmetry::Type{U₁}; spin=1 // 2)
#     Splus = sigma_plus(elt, symmetry; spin=spin)
#     Smin = sigma_min(elt, symmetry; spin=spin)
#     Sx = catdomain(permute(Smin, (1, 3), (2,)), permute(Splus, (1, 3), (2,)))
#     return permute(Sx, (1, 3), (2,)) / 2
# end

"""Pauli x operator"""
const σˣ = sigma_x(;)

"""
    sigma_y([eltype [, symmetry]; spin=S)

spin `S` operator along the y-axis.

See also [`σʸ`](@ref)
"""
function sigma_y(elt::Type{<:Complex}=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; spin=1 // 2)
    _, sigma_y_mat, _, _ = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(sigma_y_mat, 1))
    return TensorMap(sigma_y_mat, pspace ← pspace)
end

# function sigma_y(elt::Type{<:Number}, symmetry::Type{U₁}; spin=1 // 2)
#     Splus = sigma_plus(elt, symmetry; spin=spin)
#     Smin = sigma_min(elt, symmetry; spin=spin)
#     Sy = catdomain(permute(-Splus, (1, 3), (2,)), permute(Smin, (1, 3), (2,)))
#     return 1im * permute(Sy, (1, 3), (2,)) / 2
# end

"""Pauli y operator"""
const σʸ = sigma_y(;)

"""
    sigma_z([eltype [, symmetry]]; spin=S)

spin `S` operator along the z-axis.

See also [`σᶻ`](@ref)
"""
function sigma_z(elt::Type{<:Number}=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; spin=1 // 2)
    _, _, sigma_z_mat = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(sigma_z_mat, 1))
    return TensorMap(sigma_z_mat, pspace ← pspace)
end

function sigma_z(elt::Type{<:Number}, ::Type{ℤ₂}; spin=1 // 2)
    spin == 1 // 2 || error("Z2 symmetry only implemented for spin 1//2")
    pspace = Z2Space(0 => 1, 1 => 1)
    aspace = Z2Space(1 => 1)
    S = TensorMap(ones, elt, pspace ⊗ aspace ← pspace)
    return S / 2
end

function sigma_z(elt::Type{<:Number}, ::Type{U₁}; spin=1 // 2)
    charges = U₁.((-spin):spin)
    pspace = U1Space((v => 1 for v in charges))
    S = TensorMap(zeros, elt, pspace ← pspace)
    for (i, c) in enumerate(charges)
        blocks(S)[c] .= spin + 1 - i
    end
    return S
end

"""Pauli z operator"""
const σᶻ = sigma_z(;)

"""
    sigma_plus([eltype [, symmetry]]; spin=S)

spin `S` raising operator.

See also [`σ⁺`](@ref)
"""
function sigma_plus(elt::Type{<:Number}=ComplexF64, symm::Type{G}=ℤ{1};
                    spin=1 // 2) where {G<:Union{ℤ{1},ℤ₂}}
    return sigma_x(elt, symm; spin=spin) + 1im * sigma_y(elt, symm; spin=spin)
end

function sigma_plus(elt::Type{<:Number}, ::Type{U₁}; spin=1 // 2)
    charges = U₁.((-spin):spin)
    pspace = U1Space((v => 1 for v in charges))
    aspace = U1Space(-1 => 1)
    S = TensorMap(zeros, elt, pspace ⊗ aspace ← pspace)
    for (i, c) in enumerate(charges)
        c == U₁(spin) && continue
        blocks(S)[c] .= 2 * _pauliterm(spin, i, i + 1)
    end
    return S
end

"""Pauli raising operator"""
const σ⁺ = sigma_plus(;)

"""
    sigma_min([eltype [, symmetry]]; spin=S)

spin `S` lowering operator.

See also [`σ⁻`](@ref)
"""
function sigma_min(elt::Type{<:Number}=ComplexF64, symm::Type{G}=ℤ{1};
                   spin=1 // 2) where {G<:Union{ℤ{1},ℤ₂}}
    return sigma_x(elt, symm; spin=spin) - 1im * sigma_y(elt, symm; spin=spin)
end

function sigma_min(elt::Type{<:Number}, ::Type{U₁}; spin=1 // 2)
    charges = U₁.((-spin):spin)
    pspace = U1Space((v => 1 for v in charges))
    aspace = U1Space(+1 => 1)
    S = TensorMap(zeros, elt, pspace ⊗ aspace ← pspace)
    for (i, c) in enumerate(charges)
        c == U₁(-spin) && continue
        blocks(S)[c] .= 2 * _pauliterm(spin, i - 1, i)
    end
    return S
end

"""Pauli lowering operator"""
const σ⁻ = sigma_min(;)

function sigma_plusmin end
function sigma_minplus end
function sigma_exchange end

"""
    sigma_xx([eltype [, symmetry]]; spin=S)

spin `S` xx exchange operator.

See also [`σˣˣ`](@ref)
"""
function sigma_xx(elt::Type{<:Number}=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; spin=1 // 2)
    return sigma_x(elt; spin=spin) ⊗ sigma_x(elt; spin=spin)
end

"""Pauli xx exchange operator"""
const σˣˣ = sigma_xx(;)

"""
    sigma_yy([eltype [, symmetry]]; spin=S)

spin `S` yy exchange operator.

See also [`σʸʸ`](@ref)
"""
function sigma_yy(elt::Type{<:Number}=ComplexF64, symm::Type{ℤ{1}}=ℤ{1}; spin=1 // 2)
    return sigma_y(elt, symm; spin=spin) ⊗ sigma_y(elt, symm; spin=spin)
end

"""Pauli yy exchange operator"""
const σʸʸ = sigma_yy(;)

"""
    sigma_zz([eltype [, symmetry]]; spin=S)

spin `S` zz exchange operator.

See also [`σᶻᶻ`](@ref)
"""
function sigma_zz(elt::Type{<:Number}=ComplexF64, symmetry=ℤ{1}; spin=1 // 2)
    return sigma_z(elt, symmetry; spin=spin) ⊗ sigma_z(elt, symmetry; spin=spin)
end

function sigma_zz(elt::Type{<:Number}, symmetry::Type{ℤ{2}}; spin=1 // 2)
    Z = sigma_z(elt, symmetry; spin=spin)
    return @tensor ZZ[-1 -2; -3 -4] := Z[-1 1 -3] * conj(Z[-4 1 -2])
end

"""Pauli zz exchange operator"""
const σᶻᶻ = sigma_zz(;)

"""
    sigma_plusmin([eltype [, symmetry]]; spin=S)

spin `S` +- exchange operator.

See also [`σ⁺⁻`](@ref)
"""
function sigma_plusmin(elt::Type{<:Number}=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; spin=1 // 2)
    return sigma_plus(elt; spin=spin) ⊗ sigma_min(elt; spin=spin)
end

function sigma_plusmin(elt::Type{<:Number}, symmetry::Type{G};
                       spin=1 // 2) where {G<:Union{ℤ₂,U₁}}
    Splus = sigma_plus(elt, symmetry; spin=spin)
    return @tensor S[-1 -2; -3 -4] := Splus[-1 1; -3] * conj(Splus[-4 1; -2])
end

"""Pauli +- exchange operator"""
const σ⁺⁻ = sigma_plusmin(;)

"""
    sigma_minplus([eltype [, symmetry]]; spin=S)

spin `S` -+ exchange operator.

See also [`σ⁻⁺`](@ref)
"""
function sigma_minplus(elt::Type{<:Number}=ComplexF64, ::Type{ℤ{1}}=ℤ{1}; spin=1 // 2)
    return sigma_min(elt; spin=spin) ⊗ sigma_plus(elt; spin=spin)
end

function sigma_minplus(elt::Type{<:Number}, symmetry::Type{G};
                       spin=1 // 2) where {G<:Union{ℤ₂,U₁}}
    Smin = sigma_min(elt, symmetry; spin=spin)
    return @tensor S[-1 -2; -3 -4] := Smin[-1 1; -3] * conj(Smin[-4 1; -2])
end

"""Pauli -+ exchange operator"""
const σ⁻⁺ = sigma_minplus(;)

"""
    sigma_exchange([eltype [, symmetry]]; spin=S)

spin `S` exchange operator.

See also [`σσ`](@ref)
"""
function sigma_exchange(elt::Type{<:Number}=ComplexF64, ::Type{ℤ{1}}=ℤ{1};
                        spin=1 // 2)
    return sigma_xx(elt; spin=spin) + sigma_yy(elt; spin=spin) +
           sigma_zz(elt; spin=spin)
end

function sigma_exchange(elt::Type{<:Number}, symmetry::Type{G};
                        spin=1 // 2) where {G<:Union{ℤ₂,U₁}}
    return (sigma_plusmin(elt, symmetry; spin=spin) +
            sigma_minplus(elt, symmetry; spin=spin)) / 2 +
           sigma_zz(elt, symmetry; spin=spin)
end

function sigma_exchange(elt::Type{<:Number}, ::Type{SU₂}; spin=1 // 2)
    pspace = SU2Space(spin => 1)
    aspace = SU2Space(1 => 1)

    Sleft = TensorMap(ones, elt, pspace ← pspace ⊗ aspace)
    Sright = -TensorMap(ones, elt, aspace ⊗ pspace ← pspace)

    @tensor SS[-1 -2; -3 -4] := Sleft[-1; -3 1] * Sright[1 -2; -4] * (spin^2 + spin)
    return SS
end

"""Pauli exchange operator"""
const σσ = sigma_exchange(;)

"""
    electron_plusmin(elt::Type{<:Number} = ComplexF64) 

    creates the operator c^{up,+} ⊗ c^{up,-} + c^{down,+} ⊗ c^{down,-}
"""
function electron_plusmin(elt::Type{<:Number}=ComplexF64)
    psp = Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 0, 0) => 1,
                                                         (1, 1 // 2, 1) => 1,
                                                         (2, 0, 0) => 1)

    ap = TensorMap(ones, elt,
                   psp *
                   Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((-1, 1 // 2, 1) => 1),
                   psp)
    blocks(ap)[(U1Irrep(0) ⊠ SU2Irrep(0) ⊠ FermionParity(0))] .*= -sqrt(2)
    blocks(ap)[(U1Irrep(1) ⊠ SU2Irrep(1 // 2) ⊠ FermionParity(1))] .*= 1

    bm = TensorMap(ones, elt, psp,
                   Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((-1, 1 // 2, 1) => 1) *
                   psp)
    blocks(bm)[(U1Irrep(0) ⊠ SU2Irrep(0) ⊠ FermionParity(0))] .*= sqrt(2)
    blocks(bm)[(U1Irrep(1) ⊠ SU2Irrep(1 // 2) ⊠ FermionParity(1))] .*= -1

    @plansor nn[-1 -2; -3 -4] := ap[-1 1; -3] * bm[-2; 1 -4]
end

"""
    electron_plusmin(elt::Type{<:Number} = ComplexF64) 

    creates the operator c^{up,-} ⊗ c^{up,+} + c^{down,-} ⊗ c^{down,+}
"""
electron_minplus(elt::Type{<:Number}=ComplexF64) = -electron_plusmin(elt)'

"""
    electron_numberoperator(elt::Type{<:Number} = ComplexF64) 

    creates the operator c^{up,+} c^{up,-} + c^{down,+} c^{down,-}
"""
function electron_n(elt::Type{<:Number}=ComplexF64)
    psp = Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((0, 0, 0) => 1,
                                                         (1, 1 // 2, 1) => 1,
                                                         (2, 0, 0) => 1)
    h_pm = TensorMap(ones, elt, psp, psp)
    blocks(h_pm)[(U1Irrep(0) ⊠ SU2Irrep(0) ⊠ FermionParity(0))] .= 0
    blocks(h_pm)[(U1Irrep(1) ⊠ SU2Irrep(1 // 2) ⊠ FermionParity(1))] .= 1
    blocks(h_pm)[(U1Irrep(2) ⊠ SU2Irrep(0) ⊠ FermionParity(0))] .= 2

    return h_pm
end
