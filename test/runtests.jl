using Test

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

@testset "transverse field ising model" begin
    include("tfim.jl")
end

@testset "heisenberg model" begin
    include("heisenberg.jl")
end
