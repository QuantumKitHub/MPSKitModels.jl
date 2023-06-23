function _pauliterm(spin, i, j)
    1 <= i <= 2 * spin + 1 || return 0.0
    1 <= j <= 2 * spin + 1 || return 0.0
    return sqrt((spin + 1) * (i + j - 1) - i * j) / 2.0
end
function _pauliterm(spin, i::U1Irrep, j::U1Irrep)
    -spin <= i.charge <= spin || return 0.0
    -spin <= j.charge <= spin || return 0.0
    return sqrt((spin + 1) * (i.charge + j.charge + 2 * spin + 1) -
                (i.charge + spin + 1) * (j.charge + spin + 1)) / 2.0
end

"""
    spinmatrices(spin [, eltype])

the spinmatrices according to [Wikipedia](https://en.wikipedia.org/wiki/Spin_(physics)#Higher_spins).
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
    sigma_x([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the x-axis.

See also [`σˣ`](@ref)
"""
function sigma_x end
sigma_x(; kwargs...) = sigma_x(ComplexF64, Trivial; kwargs...)
sigma_x(elt::Type{<:Number}; kwargs...) = sigma_x(elt, Trivial; kwargs...)
sigma_x(symm::Type{<:Sector}; kwargs...) = sigma_x(ComplexF64, symm; kwargs...)

function sigma_x(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    sigma_x_mat, _, _ = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(sigma_x_mat, 1))
    return TensorMap(sigma_x_mat, pspace ← pspace)
end

function sigma_x(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2)
    spin == 1 // 2 || error("not implemented")
    pspace = Z2Space(0 => 1, 1 => 1)
    σˣ = TensorMap(zeros, elt, pspace, pspace)
    blocks(σˣ)[Z2Irrep(0)] .= one(elt) / 2
    blocks(σˣ)[Z2Irrep(1)] .= -one(elt) / 2
    return σˣ
end

function sigma_x(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    vspace = U1Space(1 => 1, -1 => 1)
    if side == :L
        σˣ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (f1, f2) in fusiontrees(σˣ)
            c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1 || c₁.charge + 1 == c₂.charge
                σˣ[f1, f2] .= _pauliterm(spin, c₁, c₂)
            end
        end
    elseif side == :R
        σˣ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (f1, f2) in fusiontrees(σˣ)
            c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1 || c₁.charge + 1 == c₂.charge
                σˣ[f1, f2] .= _pauliterm(spin, c₁, c₂)
            end
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σˣ
end

"""Pauli x operator"""
const σˣ = sigma_x(;)

"""
    sigma_y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the y-axis.

See also [`σʸ`](@ref)
"""
function sigma_y end
sigma_y(; kwargs...) = sigma_y(ComplexF64, Trivial; kwargs...)
sigma_y(elt::Type{<:Complex}; kwargs...) = sigma_y(elt, Trivial; kwargs...)
sigma_y(symm::Type{<:Sector}; kwargs...) = sigma_y(ComplexF64, symm; kwargs...)

function sigma_y(elt::Type{<:Complex}, ::Type{Trivial}; spin=1 // 2)
    _, sigma_y_mat, _, _ = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(sigma_y_mat, 1))
    return TensorMap(sigma_y_mat, pspace ← pspace)
end

function sigma_y(elt::Type{<:Complex}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("not implemented")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(1 => 1)
    if side == :L
        σʸ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(σʸ)[Z2Irrep(0)] .= one(elt)im / 2
        blocks(σʸ)[Z2Irrep(1)] .= -one(elt)im / 2
    elseif side == :R
        σʸ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(σʸ)[Z2Irrep(0)] .= -one(elt)im / 2
        blocks(σʸ)[Z2Irrep(1)] .= one(elt)im / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σʸ
end

function sigma_y(elt::Type{<:Complex}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    vspace = U1Space(1 => 1, -1 => 1)
    if side == :L
        σʸ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (f1, f2) in fusiontrees(σʸ)
            c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1
                σʸ[f1, f2] .= _pauliterm(spin, c₁, c₂)im
            elseif c₁.charge + 1 == c₂.charge
                σʸ[f1, f2] .= -_pauliterm(spin, c₁, c₂)im
            end
        end
    elseif side == :R
        σʸ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (f1, f2) in fusiontrees(σʸ)
            c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1
                σʸ[f1, f2] .= _pauliterm(spin, c₁, c₂)im
            elseif c₁.charge + 1 == c₂.charge
                σʸ[f1, f2] .= -_pauliterm(spin, c₁, c₂)im
            end
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σʸ
end

"""Pauli y operator"""
const σʸ = sigma_y(;)

"""
    sigma_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the z-axis.

See also [`σᶻ`](@ref)
"""
function sigma_z end
sigma_z(; kwargs...) = sigma_z(ComplexF64, Trivial; kwargs...)
sigma_z(elt::Type{<:Number}; kwargs...) = sigma_z(elt, Trivial; kwargs...)
sigma_z(symm::Type{<:Sector}; kwargs...) = sigma_z(ComplexF64, symm; kwargs...)

function sigma_z(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    _, _, sigma_z_mat = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(sigma_z_mat, 1))
    return TensorMap(sigma_z_mat, pspace ← pspace)
end

function sigma_z(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("Z2 symmetry only implemented for spin 1 // 2")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(1 => 1)
    if side == :L
        σᶻ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(σᶻ)[Z2Irrep(0)] .= one(elt) / 2
        blocks(σᶻ)[Z2Irrep(1)] .= one(elt) / 2
    elseif side == :R
        σᶻ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(σᶻ)[Z2Irrep(0)] .= one(elt) / 2
        blocks(σᶻ)[Z2Irrep(1)] .= one(elt) / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σᶻ
end

function sigma_z(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2)
    charges = U1Irrep.((-spin):spin)
    pspace = U1Space((v => 1 for v in charges))
    σᶻ = TensorMap(zeros, elt, pspace ← pspace)
    for (i, c) in enumerate(charges)
        blocks(σᶻ)[c] .= spin + 1 - i
    end
    return σᶻ
end

"""Pauli z operator"""
const σᶻ = sigma_z(;)

"""
    sigma_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin plus operator.

See also [`σ⁺`](@ref)
"""
function sigma_plus end
sigma_plus(; kwargs...) = sigma_plus(ComplexF64, Trivial; kwargs...)
sigma_plus(elt::Type{<:Number}; kwargs...) = sigma_plus(elt, Trivial; kwargs...)
sigma_plus(symm::Type{<:Sector}; kwargs...) = sigma_plus(ComplexF64, symm; kwargs...)

function sigma_plus(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    σ⁺ = sigma_x(elt, Trivial; spin=spin) + 1im * sigma_y(complex(elt), Trivial; spin=spin)
    return elt <: Real ? real(σ⁺) : σ⁺
end

function sigma_plus(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("Z2 symmetry only implemented for spin 1 // 2")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(0 => 1, 1 => 1)
    if side == :L
        σ⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(σ⁺)[Z2Irrep(0)] .= [1 -1] / 2
        blocks(σ⁺)[Z2Irrep(1)] .= [-1 1] / 2
    elseif side == :R
        σ⁺ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(σ⁺)[Z2Irrep(0)] .= [1 1]' / 2
        blocks(σ⁺)[Z2Irrep(1)] .= [-1 -1]' / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σ⁺
end

function sigma_plus(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    if side == :L
        vspace = U1Space(-1 => 1)
        σ⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (c, b) in blocks(σ⁺)
            b .= 2 * _pauliterm(spin, c, only(c ⊗ U1Irrep(+1)))
        end
    elseif side == :R
        vspace = U1Space(1 => 1)
        σ⁺ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (c, b) in blocks(σ⁺)
            b .= 2 * _pauliterm(spin, only(c ⊗ U1Irrep(-1)), c)
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σ⁺
end

"""Pauli plus operator"""
const σ⁺ = sigma_plus(;)

"""
    sigma_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin minus operator.

See also [`σ⁻`](@ref)
"""
function sigma_min end
sigma_min(; kwargs...) = sigma_min(ComplexF64, Trivial; kwargs...)
sigma_min(elt::Type{<:Number}; kwargs...) = sigma_min(elt, Trivial; kwargs...)
sigma_min(symm::Type{<:Sector}; kwargs...) = sigma_min(ComplexF64, symm; kwargs...)

function sigma_min(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    σ⁻ = sigma_x(elt, Trivial; spin=spin) - 1im * sigma_y(complex(elt), Trivial; spin=spin)
    return elt <: Real ? real(σ⁻) : σ⁻
end

function sigma_min(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("Z2 symmetry only implemented for spin 1 // 2")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(0 => 1, 1 => 1)
    if side == :L
        σ⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(σ⁻)[Z2Irrep(0)] .= [1 1] / 2
        blocks(σ⁻)[Z2Irrep(1)] .= [-1 -1] / 2
    elseif side == :R
        σ⁻ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(σ⁻)[Z2Irrep(0)] .= [1 -1]' / 2
        blocks(σ⁻)[Z2Irrep(1)] .= [1 -1]' / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σ⁻
end

function sigma_min(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    if side == :L
        vspace = U1Space(1 => 1)
        σ⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (c, b) in blocks(σ⁻)
            b .= 2 * _pauliterm(spin, only(c ⊗ U1Irrep(-1)), c)
        end
    elseif side == :R
        vspace = U1Space(-1 => 1)
        σ⁻ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (c, b) in blocks(σ⁻)
            b .= 2 * _pauliterm(spin, c, only(c ⊗ U1Irrep(+1)))
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return σ⁻
end

"""Pauli minus operator"""
const σ⁻ = sigma_min(;)

unicode_table = Dict(:x => :ˣ, :y => :ʸ, :z => :ᶻ, :plus => :⁺, :min => :⁻)

pauli_docstring(L::Symbol, R::Symbol) = """
    sigma_$L$R([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin $L$R exchange operator.

See also [`σ$(unicode_table[L])$(unicode_table[R])`](@ref)
"""
function pauli_unicode_docstring(L::Symbol, R::Symbol)
    return """Pauli $L$R operator"""
end

for (L, R) in ((:x, :x), (:y, :y), (:z, :z), (:plus, :min), (:min, :plus))
    f = Symbol(:sigma_, L, R)
    fₗ = Symbol(:sigma_, L)
    fᵣ = Symbol(:sigma_, R)
    f_unicode = Symbol(:σ, unicode_table[L], unicode_table[R])
    docstring = pauli_docstring(L, R)
    unicode_docstring = pauli_unicode_docstring(L, R)
    @eval MPSKitModels begin
        @doc $docstring $f
        ($f)(; kwargs...) = ($f)(ComplexF64, Trivial; kwargs...)
        ($f)(elt::Type{<:Number}; kwargs...) = ($f)(elt, Trivial; kwargs...)
        ($f)(symmetry::Type{<:Sector}; kwargs...) = ($f)(ComplexF64, symmetry; kwargs...)

        function ($f)(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
            return contract_twosite($(fₗ)(elt, Trivial; spin=spin),
                                    $(fᵣ)(elt, Trivial; spin=spin))
        end

        function ($f)(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
            return contract_twosite($(fₗ)(elt, symmetry; spin=spin, side=:L),
                                    $(fᵣ)(elt, symmetry; spin=spin, side=:R))
        end
        
        @doc $unicode_docstring
        const $f_unicode = ($f)(;)
    end
end

function sigma_xx(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2)
    return contract_twosite(sigma_x(elt, Z2Irrep; spin=spin),
                            sigma_x(elt, Z2Irrep; spin=spin))
end
function sigma_zz(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2)
    return contract_twosite(sigma_z(elt, U1Irrep; spin=spin),
                            sigma_z(elt, U1Irrep; spin=spin))
end

"""
    sigma_exchange([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin exchange operator.

See also [`σσ`](@ref)
"""
function sigma_exchange end
sigma_exchange(; kwargs...) = sigma_exchange(ComplexF64, Trivial; kwargs...)
sigma_exchange(elt::Type{<:Number}; kwargs...) = sigma_exchange(elt, Trivial; kwargs...)
function sigma_exchange(symmetry::Type{<:Sector}; kwargs...)
    return sigma_exchange(ComplexF64, symmetry; kwargs...)
end

function sigma_exchange(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    return sigma_xx(elt, Trivial; spin=spin) + sigma_yy(elt, Trivial; spin=spin) +
           sigma_zz(elt, Trivial; spin=spin)
end

function sigma_exchange(elt::Type{<:Number}, ::Type{SU2Irrep}; spin=1 // 2)
    pspace = SU2Space(spin => 1)
    aspace = SU2Space(1 => 1)

    Sleft = TensorMap(ones, elt, pspace ← pspace ⊗ aspace)
    Sright = -TensorMap(ones, elt, aspace ⊗ pspace ← pspace)

    @tensor SS[-1 -2; -3 -4] := Sleft[-1; -3 1] * Sright[1 -2; -4] * (spin^2 + spin)
    return SS
end

function sigma_exchange(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    return (sigma_plusmin(elt, symmetry; spin=spin) +
            sigma_minplus(elt, symmetry; spin=spin)) / 2 +
           sigma_zz(elt, symmetry; spin=spin)
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
