using Test

@testset "Spin operators" begin
    include("spinoperators.jl")
end

@testset "transverse field ising model" begin
    include("tfim.jl")
end

@testset "heisenberg model" begin
    include("heisenberg.jl")
end
