"""
    a_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)
    a⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)

The truncated bosonic creation operator, with a maximum of `cutoff` bosons per site.
"""
function a_plus end
a_plus(; kwargs...) = a_plus(ComplexF64, Trivial; kwargs...)
a_plus(elt::Type{<:Number}; kwargs...) = a_plus(elt, Trivial; kwargs...)
a_plus(symm::Type{<:Sector}; kwargs...) = a_plus(ComplexF64, symm; kwargs...)

function a_plus(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer=5)
    a⁺ = zeros(elt, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁺[n + 1, n] = sqrt(n)
    end
    return a⁺
end

function a_plus(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer=5, side=:L)
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :L
        vspace = U1Space(1 => 1)
        a⁺ = zeros(elt, pspace ← pspace ⊗ vspace)
        for (f1, f2) in fusiontrees(a⁺)
            c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1
                a⁺[f1, f2] .= -sqrt(c₁.charge)
            end
        end
    elseif side === :R
        vspace = U1Space(-1 => 1)
        a⁺ = zeros(elt, vspace ⊗ pspace ← pspace)
        for (f1, f2) in fusiontrees(a⁺)
            c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
            if c₁.charge == c₂.charge + 1
                a⁺[f1, f2] .= -sqrt(c₁.charge)
            end
        end
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
a_min(elt::Type{<:Number}; kwargs...) = a_min(elt, Trivial; kwargs...)
a_min(symm::Type{<:Sector}; kwargs...) = a_min(ComplexF64, symm; kwargs...)

function a_min(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer=5)
    a⁻ = zeros(elt, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 1:cutoff
        a⁻[n, n + 1] = sqrt(n)
    end
    return a⁻
end

function a_min(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer=5, side=:L)
    pspace = U1Space(n => 1 for n in 0:cutoff)
    if side === :L
        vspace = U1Space(-1 => 1)
        a⁻ = zeros(elt, pspace ← pspace ⊗ vspace)
        for (f1, f2) in fusiontrees(a⁻)
            c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
            if c₁.charge + 1 == c₂.charge
                a⁻[f1, f2] .= -sqrt(c₂.charge)
            end
        end
    elseif side === :R
        vspace = U1Space(1 => 1)
        a⁻ = zeros(elt, vspace ⊗ pspace ← pspace)
        for (f1, f2) in fusiontrees(a⁻)
            c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
            if c₁.charge + 1 == c₂.charge
                a⁻[f1, f2] .= -sqrt(c₂.charge)
            end
        end
    else
        throw(ArgumentError("invalid side `:$side`, expected `:L` or `:R`"))
    end
    return a⁻
end

const a⁻ = a_min

a_plusmin(symmetry::Type{<:Sector}; kwargs...) = a_plusmin(ComplexF64, symmetry; kwargs...)
function a_plusmin(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial;
                   cutoff::Integer=5)
    return contract_twosite(a⁺(elt; cutoff=cutoff), a⁻(elt; cutoff=cutoff))
end
function a_plusmin(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer=5)
    return contract_twosite(a⁺(elt, U1Irrep; cutoff=cutoff, side=:L),
                            a⁻(elt, U1Irrep; cutoff=cutoff, side=:R))
end

a_minplus(symmetry::Type{<:Sector}; kwargs...) = a_minplus(ComplexF64, symmetry; kwargs...)
function a_minplus(elt::Type{<:Number}=ComplexF64, ::Type{Trivial}=Trivial;
                   cutoff::Integer=5)
    return contract_twosite(a⁻(elt; cutoff=cutoff), a⁺(elt; cutoff=cutoff))
end
function a_minplus(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer=5)
    return contract_twosite(a⁻(elt, U1Irrep; cutoff=cutoff, side=:L),
                            a⁺(elt, U1Irrep; cutoff=cutoff, side=:R))
end

"""
    a_number([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)

The truncated bosonic number operator, with a maximum of `cutoff` bosons per site.
"""
function a_number end
a_number(; kwargs...) = a_number(ComplexF64, Trivial; kwargs...)
a_number(elt::Type{<:Number}; kwargs...) = a_number(elt, Trivial; kwargs...)
a_number(symm::Type{<:Sector}; kwargs...) = a_number(ComplexF64, symm; kwargs...)

function a_number(elt::Type{<:Number}, ::Type{Trivial}; cutoff::Integer=5)
    N = zeros(elt, ComplexSpace(cutoff + 1), ComplexSpace(cutoff + 1))
    for n in 0:cutoff
        N[n + 1, n + 1] = n
    end
    return N
end

function a_number(elt::Type{<:Number}, ::Type{U1Irrep}; cutoff::Integer=5)
    pspace = U1Space(n => 1 for n in 0:cutoff)
    N = zeros(elt, pspace, pspace)
    for (c, b) in blocks(N)
        b .= c.charge
    end
    return N
end
