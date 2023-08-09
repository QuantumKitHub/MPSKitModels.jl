@testset "InfiniteCylinder" begin
    ZZ = S_zz()
    
    lattice = InfiniteCylinder(5)
    @mpoham begin
        H = sum(ZZ{i,j} for (i,j) in nearest_neighbours(lattice))
    end
    @test length(H) == length(lattice)
end