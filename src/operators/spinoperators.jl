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
    S_x([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    Sˣ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the x-axis.

See also [`σˣ`](@ref)
"""
function S_x end
S_x(; kwargs...) = S_x(ComplexF64, Trivial; kwargs...)
S_x(elt::Type{<:Number}; kwargs...) = S_x(elt, Trivial; kwargs...)
S_x(symm::Type{<:Sector}; kwargs...) = S_x(ComplexF64, symm; kwargs...)

function S_x(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    S_x_mat, _, _ = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(S_x_mat, 1))
    return TensorMap(S_x_mat, pspace ← pspace)
end

function S_x(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2)
    spin == 1 // 2 || error("not implemented")
    pspace = Z2Space(0 => 1, 1 => 1)
    X = TensorMap(zeros, elt, pspace, pspace)
    blocks(X)[Z2Irrep(0)] .= one(elt) / 2
    blocks(X)[Z2Irrep(1)] .= -one(elt) / 2
    return X
end

function S_x(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    vspace = U1Space(1 => 1, -1 => 1)
    if side == :L
        X = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (f1, f2) in fusiontrees(X)
            c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1 || c₁.charge + 1 == c₂.charge
                X[f1, f2] .= _pauliterm(spin, c₁, c₂)
            end
        end
    elseif side == :R
        X = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (f1, f2) in fusiontrees(X)
            c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1 || c₁.charge + 1 == c₂.charge
                X[f1, f2] .= _pauliterm(spin, c₁, c₂)
            end
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return X
end

const Sˣ = S_x

"""Pauli x operator."""
σˣ(args...; kwargs...) = 2 * S_x(args...; kwargs...)

"""
    S_y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    Sʸ([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the y-axis.

See also [`σʸ`](@ref)
"""
function S_y end
S_y(; kwargs...) = S_y(ComplexF64, Trivial; kwargs...)
S_y(elt::Type{<:Complex}; kwargs...) = S_y(elt, Trivial; kwargs...)
S_y(symm::Type{<:Sector}; kwargs...) = S_y(ComplexF64, symm; kwargs...)

function S_y(elt::Type{<:Complex}, ::Type{Trivial}; spin=1 // 2)
    _, Y, _, _ = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(Y, 1))
    return TensorMap(Y, pspace ← pspace)
end

function S_y(elt::Type{<:Complex}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("not implemented")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(1 => 1)
    if side == :L
        Y = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(Y)[Z2Irrep(0)] .= one(elt)im / 2
        blocks(Y)[Z2Irrep(1)] .= -one(elt)im / 2
    elseif side == :R
        Y = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(Y)[Z2Irrep(0)] .= -one(elt)im / 2
        blocks(Y)[Z2Irrep(1)] .= one(elt)im / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return Y
end

function S_y(elt::Type{<:Complex}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    vspace = U1Space(1 => 1, -1 => 1)
    if side == :L
        Y = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (f1, f2) in fusiontrees(Y)
            c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1
                Y[f1, f2] .= _pauliterm(spin, c₁, c₂)im
            elseif c₁.charge + 1 == c₂.charge
                Y[f1, f2] .= -_pauliterm(spin, c₁, c₂)im
            end
        end
    elseif side == :R
        Y = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (f1, f2) in fusiontrees(Y)
            c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1
                Y[f1, f2] .= _pauliterm(spin, c₁, c₂)im
            elseif c₁.charge + 1 == c₂.charge
                Y[f1, f2] .= -_pauliterm(spin, c₁, c₂)im
            end
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return Y
end

const Sʸ = S_y

"""Pauli y operator."""
σʸ(args...; kwargs...) = 2 * S_y(args...; kwargs...)

"""
    S_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    Sᶻ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin operator along the z-axis. Possible values for `symmetry` are `Trivial`, `Z2Irrep`,
and `U1Irrep`.

See also [`σᶻ`](@ref)
"""
function S_z end
S_z(; kwargs...) = S_z(ComplexF64, Trivial; kwargs...)
S_z(elt::Type{<:Number}; kwargs...) = S_z(elt, Trivial; kwargs...)
S_z(symm::Type{<:Sector}; kwargs...) = S_z(ComplexF64, symm; kwargs...)

function S_z(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    _, _, S_z_mat = spinmatrices(spin, elt)
    pspace = ComplexSpace(size(S_z_mat, 1))
    return TensorMap(S_z_mat, pspace ← pspace)
end

function S_z(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("Z2 symmetry only implemented for spin 1 // 2")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(1 => 1)
    if side == :L
        Z = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(Z)[Z2Irrep(0)] .= one(elt) / 2
        blocks(Z)[Z2Irrep(1)] .= one(elt) / 2
    elseif side == :R
        Z = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(Z)[Z2Irrep(0)] .= one(elt) / 2
        blocks(Z)[Z2Irrep(1)] .= one(elt) / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return Z
end

function S_z(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2)
    charges = U1Irrep.((-spin):spin)
    pspace = U1Space((v => 1 for v in charges))
    Z = TensorMap(zeros, elt, pspace ← pspace)
    for (i, c) in enumerate(charges)
        blocks(Z)[c] .= spin + 1 - i
    end
    return Z
end

const Sᶻ = S_z

"""Pauli z operator."""
σᶻ(args...; kwargs...) = 2 * S_z(args...; kwargs...)

"""
    S_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    S⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin plus operator.

See also [`σ⁺`](@ref)
"""
function S_plus end
S_plus(; kwargs...) = S_plus(ComplexF64, Trivial; kwargs...)
S_plus(elt::Type{<:Number}; kwargs...) = S_plus(elt, Trivial; kwargs...)
S_plus(symm::Type{<:Sector}; kwargs...) = S_plus(ComplexF64, symm; kwargs...)

function S_plus(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    S⁺ = S_x(elt, Trivial; spin=spin) + 1im * S_y(complex(elt), Trivial; spin=spin)
    return elt <: Real ? real(S⁺) : S⁺
end

function S_plus(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("Z2 symmetry only implemented for spin 1 // 2")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(0 => 1, 1 => 1)
    if side == :L
        S⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(S⁺)[Z2Irrep(0)] .= [1 -1] / 2
        blocks(S⁺)[Z2Irrep(1)] .= [-1 1] / 2
    elseif side == :R
        S⁺ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(S⁺)[Z2Irrep(0)] .= [1 1]' / 2
        blocks(S⁺)[Z2Irrep(1)] .= [-1 -1]' / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return S⁺
end

function S_plus(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    if side == :L
        vspace = U1Space(-1 => 1)
        S⁺ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (c, b) in blocks(S⁺)
            b .= 2 * _pauliterm(spin, c, only(c ⊗ U1Irrep(+1)))
        end
    elseif side == :R
        vspace = U1Space(1 => 1)
        S⁺ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (c, b) in blocks(S⁺)
            b .= 2 * _pauliterm(spin, only(c ⊗ U1Irrep(-1)), c)
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return S⁺
end

const S⁺ = S_plus

"""Pauli plus operator."""
σ⁺(args...; kwargs...) = 2 * S_plus(args...; kwargs...)

"""
    S_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    S⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin minus operator.

See also [`σ⁻`](@ref)
"""
function S_min end
S_min(; kwargs...) = S_min(ComplexF64, Trivial; kwargs...)
S_min(elt::Type{<:Number}; kwargs...) = S_min(elt, Trivial; kwargs...)
S_min(symm::Type{<:Sector}; kwargs...) = S_min(ComplexF64, symm; kwargs...)

function S_min(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    S⁻ = S_x(elt, Trivial; spin=spin) - 1im * S_y(complex(elt), Trivial; spin=spin)
    return elt <: Real ? real(S⁻) : S⁻
end

function S_min(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2, side=:L)
    spin == 1 // 2 || error("Z2 symmetry only implemented for spin 1 // 2")
    pspace = Z2Space(0 => 1, 1 => 1)
    vspace = Z2Space(0 => 1, 1 => 1)
    if side == :L
        S⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        blocks(S⁻)[Z2Irrep(0)] .= [1 1] / 2
        blocks(S⁻)[Z2Irrep(1)] .= [-1 -1] / 2
    elseif side == :R
        S⁻ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        blocks(S⁻)[Z2Irrep(0)] .= [1 -1]' / 2
        blocks(S⁻)[Z2Irrep(1)] .= [1 -1]' / 2
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return S⁻
end

function S_min(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2, side=:L)
    pspace = U1Space(i => 1 for i in (-spin):spin)
    if side == :L
        vspace = U1Space(1 => 1)
        S⁻ = TensorMap(zeros, elt, pspace ← pspace ⊗ vspace)
        for (c, b) in blocks(S⁻)
            b .= 2 * _pauliterm(spin, only(c ⊗ U1Irrep(-1)), c)
        end
    elseif side == :R
        vspace = U1Space(-1 => 1)
        S⁻ = TensorMap(zeros, elt, vspace ⊗ pspace ← pspace)
        for (c, b) in blocks(S⁻)
            b .= 2 * _pauliterm(spin, c, only(c ⊗ U1Irrep(+1)))
        end
    else
        throw(ArgumentError("invalid side `:$side`"))
    end
    return S⁻
end

const S⁻ = S_min

"""Pauli minus operator."""
σ⁻(args...; kwargs...) = 2 * S_min(args...; kwargs...)

unicode_table = Dict(:x => :ˣ, :y => :ʸ, :z => :ᶻ, :plus => :⁺, :min => :⁻)

function spinop_docstring(L::Symbol, R::Symbol)
    return """
    S_$L$R([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    $(Symbol(:S, unicode_table[L], unicode_table[R]))([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin $L$R exchange operator.

See also [`σ$(unicode_table[L])$(unicode_table[R])`](@ref)
"""
end
function pauli_unicode_docstring(L::Symbol, R::Symbol)
    return """Pauli $L$R operator."""
end

for (L, R) in ((:x, :x), (:y, :y), (:z, :z), (:plus, :min), (:min, :plus))
    f = Symbol(:S_, L, R)
    fₗ = Symbol(:S_, L)
    fᵣ = Symbol(:S_, R)
    f_unicode = Symbol(:S, unicode_table[L], unicode_table[R])
    f_pauli = Symbol(:σ, unicode_table[L], unicode_table[R])
    docstring = spinop_docstring(L, R)
    pauli_docstring = pauli_unicode_docstring(L, R)
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

        const $f_unicode = $f

        @doc $pauli_docstring function $f_pauli(args...; kwargs...)
            return 4 * ($f)(args...; kwargs...)
        end
    end
end

function S_xx(elt::Type{<:Number}, ::Type{Z2Irrep}; spin=1 // 2)
    return contract_twosite(S_x(elt, Z2Irrep; spin=spin), S_x(elt, Z2Irrep; spin=spin))
end
function S_zz(elt::Type{<:Number}, ::Type{U1Irrep}; spin=1 // 2)
    return contract_twosite(S_z(elt, U1Irrep; spin=spin), S_z(elt, U1Irrep; spin=spin))
end

"""
    S_exchange([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)
    SS([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)

The spin exchange operator.

See also [`σσ`](@ref)
"""
function S_exchange end
S_exchange(; kwargs...) = S_exchange(ComplexF64, Trivial; kwargs...)
S_exchange(elt::Type{<:Number}; kwargs...) = S_exchange(elt, Trivial; kwargs...)
function S_exchange(symmetry::Type{<:Sector}; kwargs...)
    return S_exchange(ComplexF64, symmetry; kwargs...)
end

function S_exchange(elt::Type{<:Number}, ::Type{Trivial}; spin=1 // 2)
    return S_xx(elt, Trivial; spin=spin) +
           S_yy(elt, Trivial; spin=spin) +
           S_zz(elt, Trivial; spin=spin)
end

function S_exchange(elt::Type{<:Number}, ::Type{SU2Irrep}; spin=1 // 2)
    pspace = SU2Space(spin => 1)
    aspace = SU2Space(1 => 1)

    Sleft = TensorMap(ones, elt, pspace ← pspace ⊗ aspace)
    Sright = -TensorMap(ones, elt, aspace ⊗ pspace ← pspace)

    @tensor SS[-1 -2; -3 -4] := Sleft[-1; -3 1] * Sright[1 -2; -4] * (spin^2 + spin)
    return SS
end

function S_exchange(elt::Type{<:Number}, symmetry::Type{<:Sector}; spin=1 // 2)
    return (S_plusmin(elt, symmetry; spin=spin) + S_minplus(elt, symmetry; spin=spin)) / 2 +
           S_zz(elt, symmetry; spin=spin)
end

const SS = S_exchange

"""Pauli exchange operator."""
σσ(args...; kwargs...) = 4 * S_exchange(args...; kwargs...)
