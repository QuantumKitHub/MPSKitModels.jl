using MPSKitModels
using LinearAlgebra: norm


@testset "InfiniteChain" begin
    for L in 1:4
        lattice = InfiniteChain(L)
        V = vertices(lattice)
        @test length(lattice) == length(V) == L
        
        NN = nearest_neighbours(lattice)
        @test length(NN) == L # coordination number 2
        
        min_dist = 1
        for (i,j) in NN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)
    end
end

@testset "InfiniteCylinder" begin
    for L in 2:4
        lattice = InfiniteCylinder(L)
        V = vertices(lattice)
        @test length(lattice) == length(V) == L
        
        NN = nearest_neighbours(lattice)
        @test length(NN) == 2L # coordination number 4
        
        min_dist = 1
        for (i,j) in NN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)
        
        for N in (2L, 3L)
            lattice = InfiniteCylinder(L, N)
            V = vertices(lattice)
            @test length(lattice) == length(V) == N
        
            NN = nearest_neighbours(lattice)
            @test length(NN) == 2N # coordination number 4
            
            min_dist = 1
            for (i,j) in NN
                @test norm(i - j) ≈ min_dist
            end
            @test allunique(NN)
        end
        
        @test_throws ArgumentError InfiniteCylinder(L, 5)
    end
end

@testset "InfiniteHelix" begin
    for L in 2:4, N in 1:3
        lattice = InfiniteHelix(L, N)
        V = vertices(lattice)
        @test length(lattice) == length(V) == N
        
        NN = nearest_neighbours(lattice)
        @test length(NN) == 4 * length(V) / 2 # coordination number 4
        
        min_dist = 1
        for (i,j) in NN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)
    end
end

@testset "InfiniteHoneycombYC" begin
     for L in 4:4:12, N in 1:3
        lattice = HoneycombYC(L, N*L)
        V = vertices(lattice)
        @test length(lattice) == length(V) == N * L
        
        @show NN = nearest_neighbours(lattice)
        @test length(NN) == 3 * length(V) / 2 # coordination number 3
        
        min_dist = 1
        for (i,j) in NN
            @show norm(i - j)
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)
    end
    
    @test_throws ArgumentError HoneycombYC(3)
    @test_throws ArgumentError HoneycombYC(4,6)
end