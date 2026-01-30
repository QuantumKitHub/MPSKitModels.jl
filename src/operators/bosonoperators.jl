"""
    a_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)
    a⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
"""
function a_plus end
a_plus(; kwargs...) = a_plus(ComplexF64, Vector{ComplexF64}, Trivial; kwargs...)
a_plus(::Type{T}, ::Type{TA}; kwargs...) where {T <: Number, TA <: AbstractArray{T}} = a_plus(T, TA, Trivial; kwargs...)
a_plus(symm::Type{<:Sector}; kwargs...) = a_plus(ComplexF64, Vector{ComplexF64}, symm; kwargs...)

function a_plus(::Type{TorA}, ::Type{Trivial}; cutoff::Integer = 5) where {TorA}
    a⁺ = zeros(TorA, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁺[n + 1, n] = sqrt(n)
    end
    return a⁺
end

function build_a_plus_left!(::Type{U1Irrep}, ::Type{TA}, a⁺) where {TA}
    for (f1, f2) in fusiontrees(a⁺)
        c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
        if c₁.charge == c₂.charge + 1
            a⁺[f1, f2] .= -sqrt(c₁.charge)
        end
    end
    return
end

function build_a_plus_right!(::Type{U1Irrep}, ::Type{TA}, a⁺) where {TA}
    for (f1, f2) in fusiontrees(a⁺)
        c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
        if c₁.charge == c₂.charge + 1
            a⁺[f1, f2] .= -sqrt(c₁.charge)
        end
    end
    return
end

function a_plus(::Type{TorA}, ::Type{U1Irrep}; cutoff::Integer = 5, side = :L) where {TorA}
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :L
        vspace = U1Space(1 => 1)
        a⁺ = zeros(TorA, pspace ← pspace ⊗ vspace)
        build_a_plus_left!(U1Irrep, TorA, a⁺)
    elseif side === :R
        vspace = U1Space(-1 => 1)
        a⁺ = zeros(TorA, vspace ⊗ pspace ← pspace)
        build_a_plus_right!(U1Irrep, TorA, a⁺)
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return a⁺
end

const a⁺ = a_plus

"""
    a_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)
    a⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)    

The truncated bosonic annihilation operator, with a maximum of `cutoff` bosons per site.
"""
function a_min end
a_min(; kwargs...) = a_min(ComplexF64, Trivial; kwargs...)
a_min(::Type{TorA}; kwargs...) where {TorA} = a_min(TorA, Trivial; kwargs...)
a_min(symm::Type{<:Sector}; kwargs...) = a_min(ComplexF64, symm; kwargs...)

function a_min(::Type{TorA}, ::Type{Trivial}; cutoff::Integer = 5) where {TorA}
    a⁻ = zeros(TorA, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁻[n, n + 1] = sqrt(n)
    end
    return a⁻
end

function build_a_min_left!(::Type{U1Irrep}, ::Type{TA}, a⁻) where {TA}
    for (f1, f2) in fusiontrees(a⁻)
        c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
        if c₁.charge + 1 == c₂.charge
            a⁻[f1, f2] .= -sqrt(c₂.charge)
        end
    end
    return
end

function build_a_min_right!(::Type{U1Irrep}, ::Type{TA}, a⁻) where {TA}
    for (f1, f2) in fusiontrees(a⁻)
        c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
        if c₁.charge + 1 == c₂.charge
            a⁻[f1, f2] .= -sqrt(c₂.charge)
        end
    end
    return
end

function a_min(::Type{TorA}, ::Type{U1Irrep}; cutoff::Integer = 5, side = :L) where {TorA}
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :L
        vspace = U1Space(-1 => 1)
        a⁻ = zeros(TorA, pspace ← pspace ⊗ vspace)
        build_a_min_left!(U1Irrep, TorA, a⁻)
    elseif side === :R
        vspace = U1Space(1 => 1)
        a⁻ = zeros(TorA, vspace ⊗ pspace ← pspace)
        build_a_min_right!(U1Irrep, TorA, a⁻)
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return a⁻
end

const a⁻ = a_min

a_plusmin(symmetry::Type{<:Sector}; kwargs...) = a_plusmin(ComplexF64, symmetry; kwargs...)
function a_plusmin(
        ::Type{TorA}, ::Type{Trivial} = Trivial;
        cutoff::Integer = 5
    ) where {TorA}
    return contract_twosite(a⁺(TorA; cutoff = cutoff), a⁻(elt; cutoff = cutoff))
end
function a_plusmin(::Type{TorA}, ::Type{U1Irrep}; cutoff::Integer = 5) where {TorA}
    return contract_twosite(
        a⁺(TorA, U1Irrep; cutoff = cutoff, side = :L),
        a⁻(TorA, U1Irrep; cutoff = cutoff, side = :R)
    )
end

a_minplus(symmetry::Type{<:Sector}; kwargs...) = a_minplus(ComplexF64, symmetry; kwargs...)
function a_minplus(
        ::Type{TorA}, ::Type{Trivial} = Trivial;
        cutoff::Integer = 5
    ) where {TorA}
    return contract_twosite(a⁻(TorA; cutoff = cutoff), a⁺(elt; cutoff = cutoff))
end
function a_minplus(::Type{TorA}, ::Type{U1Irrep}; cutoff::Integer = 5) where {TorA}
    return contract_twosite(
        a⁻(TorA, U1Irrep; cutoff = cutoff, side = :L),
        a⁺(TorA, U1Irrep; cutoff = cutoff, side = :R)
    )
end

"""
    a_number([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)

The truncated bosonic number operator, with a maximum of `cutoff` bosons per site.
"""
function a_number end
a_number(; kwargs...) = a_number(ComplexF64, Trivial; kwargs...)
a_number(::Type{TorA}; kwargs...) where {TorA} = a_number(TorA, Trivial; kwargs...)
a_number(symm::Type{<:Sector}; kwargs...) = a_number(ComplexF64, symm; kwargs...)

function a_number(::Type{TorA}, ::Type{Trivial}; cutoff::Integer = 5) where {TorA}
    N = zeros(TorA, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 0:cutoff
        N[n + 1, n + 1] = n
    end
    return N
end

function a_number(::Type{TorA}, ::Type{U1Irrep}; cutoff::Integer = 5) where {TorA}
    pspace = U1Space(n => 1 for n in 0:cutoff)
    N = zeros(TorA, pspace, pspace)
    for (c, b) in blocks(N)
        b .= c.charge
    end
    return N
end
