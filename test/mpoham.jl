@testset "InfiniteCylinder" begin
    ZZ = S_zz()

    lattice = InfiniteCylinder(5)
    H = @mpoham sum(ZZ{i,j} for (i, j) in nearest_neighbours(lattice))
    @test length(H) == length(lattice)
end
