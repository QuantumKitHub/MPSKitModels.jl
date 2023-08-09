using Test
using MPSKitModels

import TensorOperations
TensorOperations.disable_cache()

@testset "lattices" verbose = true begin
    include("lattices.jl")
end

@testset "Spin operators" begin
    include("spinoperators.jl")
end

@testset "Boson operators" begin
    include("bosonoperators.jl")
end

@testset "fermion operators" begin
    include("fermionoperators.jl")
end

@testset "mpoham" begin
    include("mpoham.jl")
end

@testset "transverse field ising model" begin
    include("tfim.jl")
end

@testset "heisenberg model" begin
    include("heisenberg.jl")
end

@testset "sixvertex model" begin
    include("sixvertex.jl")
end