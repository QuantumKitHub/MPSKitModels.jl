module MPSKitModelsCUDAExt

using MPSKitModels, CUDA, cuTENSOR

import MPSKitModels: build_a_plus_left!, build_a_plus_right!, build_a_min_left!, build_a_min_right!
import MPSKitModels: e_plusmin_up, e_plusmin_down, e_number_up, e_number_down, spinmatrices
using MPSKitModels: SU2Irrep, U1Irrep, Trivial, fusiontrees, sectortype, block, dual

function e_number_down(::Type{TA}, ::Type{Trivial} = Trivial, ::Type{Trivial} = Trivial) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.single_site_operator(TA, Trivial, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1), I(1))][2, 2] = 1
        t[(I(0), I(0))][2, 2] = 1
    end
    return t
end
function e_number_down(::Type{TA}, ::Type{Trivial}, ::Type{U1Irrep}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.single_site_operator(TA, Trivial, U1Irrep)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1, -1 // 2), dual(I(1, -1 // 2)))][1, 1] = 1
        t[(I(0, 0), I(0, 0))][2, 2] = 1
    end
    return t
end
function e_number_down(::Type{TA}, ::Type{U1Irrep}, ::Type{Trivial}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.single_site_operator(TA, U1Irrep, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        block(t, I(1, 1))[2, 2] = 1 # expected to be [1,2]
        block(t, I(0, 2))[1, 1] = 1
    end
    return t
end
function e_number_down(::Type{TA}, ::Type{U1Irrep}, ::Type{U1Irrep}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.single_site_operator(TA, U1Irrep, U1Irrep)
    I = sectortype(t)
    CUDA.@allowscalar begin
        block(t, I(1, 1, -1 // 2)) .= 1
        block(t, I(0, 2, 0)) .= 1
    end
    return t
end
function e_number_up(::Type{TA}, ::Type{Trivial}, ::Type{Trivial}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.single_site_operator(TA, Trivial, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1), I(1))][1, 1] = 1
        t[(I(0), I(0))][2, 2] = 1
    end
    return t
end
function e_number_up(::Type{TA}, ::Type{U1Irrep}, ::Type{Trivial}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.single_site_operator(TA, U1Irrep, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        block(t, I(1, 1))[1, 1] = 1
        block(t, I(0, 2))[1, 1] = 1
    end
    return t
end
function e_number_up(::Type{TA}, ::Type{U1Irrep}, ::Type{U1Irrep}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.single_site_operator(TA, U1Irrep, U1Irrep)
    I = sectortype(t)
    CUDA.@allowscalar begin
        block(t, I(1, 1, 1 // 2)) .= 1
        block(t, I(0, 2, 0)) .= 1
    end
    return t
end
function e_plusmin_up(::Type{TA}, ::Type{Trivial}, ::Type{Trivial}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.two_site_operator(TA, Trivial, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1), I(0), dual(I(0)), dual(I(1)))][1, 1, 1, 1] = 1
        t[(I(1), I(1), dual(I(0)), dual(I(0)))][1, 2, 1, 2] = 1
        t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 2, 1] = -1
        t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 2, 2, 2] = -1
    end
    return t
end
function e_plusmin_up(::Type{TA}, ::Type{U1Irrep}, ::Type{Trivial}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.two_site_operator(TA, U1Irrep, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1, 1), I(0, 0), dual(I(0, 0)), dual(I(1, 1)))][1, 1, 1, 1] = 1
        t[(I(1, 1), I(1, 1), dual(I(0, 0)), dual(I(0, 2)))][1, 2, 1, 1] = 1
        t[(I(0, 2), I(0, 0), dual(I(1, 1)), dual(I(1, 1)))][1, 1, 2, 1] = -1
        t[(I(0, 2), I(1, 1), dual(I(1, 1)), dual(I(0, 2)))][1, 2, 2, 1] = -1
    end
    return t
end

function e_plusmin_down(::Type{TA}, ::Type{Trivial}, ::Type{Trivial}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.two_site_operator(TA, Trivial, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1), I(0), dual(I(0)), dual(I(1)))][2, 1, 1, 2] = 1
        t[(I(1), I(1), dual(I(0)), dual(I(0)))][2, 1, 1, 2] = -1
        t[(I(0), I(0), dual(I(1)), dual(I(1)))][2, 1, 1, 2] = 1
        t[(I(0), I(1), dual(I(1)), dual(I(0)))][2, 1, 1, 2] = -1
    end
    return t
end
function e_plusmin_down(::Type{TA}, ::Type{Trivial}, ::Type{U1Irrep}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.two_site_operator(T, Trivial, U1Irrep)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1, -1 // 2), I(0, 0), dual(I(0, 0)), dual(I(1, -1 // 2)))][1, 1, 1, 1] = 1
        t[(I(1, -1 // 2), I(1, 1 // 2), dual(I(0, 0)), dual(I(0, 0)))][1, 1, 1, 2] = -1
        t[(I(0, 0), I(0, 0), dual(I(1, 1 // 2)), dual(I(1, -1 // 2)))][2, 1, 1, 1] = 1
        t[(I(0, 0), I(1, 1 // 2), dual(I(1, 1 // 2)), dual(I(0, 0)))][2, 1, 1, 2] = -1
    end
    return t
end
function e_plusmin_down(::Type{TA}, ::Type{U1Irrep}, ::Type{Trivial}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.two_site_operator(TA, U1Irrep, Trivial)
    I = sectortype(t)
    CUDA.@allowscalar begin
        t[(I(1, 1), I(0, 0), dual(I(0, 0)), dual(I(1, 1)))][2, 1, 1, 2] = 1
        t[(I(1, 1), I(1, 1), dual(I(0, 0)), dual(I(0, 2)))][2, 1, 1, 1] = -1
        t[(I(0, 2), I(0, 0), dual(I(1, 1)), dual(I(1, 1)))][1, 1, 1, 2] = 1
        t[(I(0, 2), I(1, 1), dual(I(1, 1)), dual(I(0, 2)))][1, 1, 1, 1] = -1
    end
    return t
end
function e_plusmin_down(::Type{TA}, ::Type{U1Irrep}, ::Type{U1Irrep}) where {TA <: CuArray}
    t = MPSKitModels.HubbardOperators.two_site_operator(T, U1Irrep, U1Irrep)
    I = sectortype(t)
    t[(I(1, 1, -1 // 2), I(0, 0, 0), dual(I(0, 0, 0)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(1, 1, -1 // 2), I(1, 1, 1 // 2), dual(I(0, 0, 0)), dual(I(0, 2, 0)))] .= -1
    t[(I(0, 2, 0), I(0, 0, 0), dual(I(1, 1, 1 // 2)), dual(I(1, 1, -1 // 2)))] .= 1
    t[(I(0, 2, 0), I(1, 1, 1 // 2), dual(I(1, 1, 1 // 2)), dual(I(0, 2, 0)))] .= -1
    return t
end

function build_a_plus_left!(::Type{U1Irrep}, ::Type{TA}, a⁺) where {TA <: CuArray}
    for (f1, f2) in fusiontrees(a⁺)
        c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
        if c₁.charge == c₂.charge + 1
            CUDA.@allowscalar a⁺[f1, f2] .= -sqrt(c₁.charge)
        end
    end
    return
end

function build_a_plus_right!(::Type{U1Irrep}, ::Type{TA}, a⁺) where {TA <: CuArray}
    for (f1, f2) in fusiontrees(a⁺)
        c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
        if c₁.charge == c₂.charge + 1
            CUDA.@allowscalar a⁺[f1, f2] .= -sqrt(c₁.charge)
        end
    end
    return
end

function build_a_min_left!(::Type{U1Irrep}, ::Type{TA}, a⁻) where {TA <: CuArray}
    for (f1, f2) in fusiontrees(a⁻)
        c₁, c₂ = f1.uncoupled[1], f2.uncoupled[1]
        if c₁.charge + 1 == c₂.charge
            CUDA.@allowscalar a⁻[f1, f2] .= -sqrt(c₂.charge)
        end
    end
    return
end

function build_a_min_right!(::Type{U1Irrep}, ::Type{TA}, a⁻) where {TA <: CuArray}
    for (f1, f2) in fusiontrees(a⁻)
        c₁, c₂ = f1.uncoupled[2], f2.uncoupled[1]
        if c₁.charge + 1 == c₂.charge
            CUDA.@allowscalar a⁻[f1, f2] .= -sqrt(c₂.charge)
        end
    end
    return
end

function spinmatrices(s::Union{Rational{Int}, Int}, ::Type{TorA}) where {T, TorA <: CuArray{T}}
    hSx, hSy, hSz, hI = spinmatrices(s, T)
    return CuArray(hSx), CuArray(hSy), CuArray(hSz), CuArray(hI)
end

end
