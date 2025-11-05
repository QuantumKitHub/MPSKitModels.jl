using MPSKitModels, MPSKit, TensorKit

lattice = InfiniteChain(1)
H1 = @mpoham begin
    sum(nearest_neighbours(lattice)) do (i, j)
        return (σˣˣ() + σʸʸ()){i, j}
    end
end

H2 = @mpoham begin
    sum(nearest_neighbours(lattice)) do (i, j)
        return (σˣ(){i} * σˣ(){j} + σʸ(){i} * σʸ(){j})
    end
end

@testset "InfiniteCylinder" begin
    ZZ = S_zz()

    lattice = InfiniteCylinder(5)
    H = @mpoham sum(ZZ{i, j} for (i, j) in nearest_neighbours(lattice))
    @test length(H) == length(lattice)
end

@testset "deduce_pspaces" begin
    # Not fully defining the pspaces should still work
    lattice = FiniteChain(5)
    H = @mpoham S_zz(){lattice[2], lattice[3]}

    @test unique(MPSKit.physicalspace(H))[1] == ComplexSpace(2)

    @test_throws Exception @mpoham σˣ(){lattice[1]} + σˣ(; spin = 3 // 2){lattice[2]}
end
