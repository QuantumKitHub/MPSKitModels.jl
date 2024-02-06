using MPSKitModels

lattice = InfiniteChain(1)
H1 = @mpoham begin
    sum(nearest_neighbours(lattice)) do (i, j)
        return (σˣˣ() + σʸʸ()){i,j}
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
    H = @mpoham sum(ZZ{i,j} for (i, j) in nearest_neighbours(lattice))
    @test length(H) == length(lattice)
end
