using LinearAlgebra: norm

@testset "InfiniteChain" begin
    for L in 1:4
        lattice = InfiniteChain(L)

        V = vertices(lattice)
        @test length(lattice) == length(V) == L
        @test lattice[1] == first(V)

        NN = nearest_neighbours(lattice)
        @test length(NN) == L # coordination number 2
        min_dist = 1
        for (i, j) in NN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)

        NNN = next_nearest_neighbours(lattice)
        @test length(NNN) == L
        min_dist = 2
        for (i, j) in NNN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NNN)
    end
end

@testset "InfiniteCylinder" begin
    for L in 2:4
        lattice = InfiniteCylinder(L)
        V = vertices(lattice)
        @test length(lattice) == length(V) == L
        @test lattice[1, 1] == first(V)

        NN = nearest_neighbours(lattice)
        @test length(NN) == 2L # coordination number 4
        min_dist = 1
        for (i, j) in NN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)

        NNN = next_nearest_neighbours(lattice)
        @test length(NNN) == 2L # coordination number 4
        min_dist = sqrt(2)
        for (i, j) in NNN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NNN)

        for N in (2L, 3L)
            lattice = InfiniteCylinder(L, N)
            V = vertices(lattice)
            @test length(lattice) == length(V) == N

            NN = nearest_neighbours(lattice)
            @test length(NN) == 2N # coordination number 4
            min_dist = 1
            for (i, j) in NN
                @test norm(i - j) ≈ min_dist
            end
            @test allunique(NN)

            NNN = next_nearest_neighbours(lattice)
            @test length(NNN) == 2N # coordination number 4
            min_dist = sqrt(2)
            for (i, j) in NNN
                @test norm(i - j) ≈ min_dist
            end
            @test allunique(NNN)
        end

        @test_throws ArgumentError InfiniteCylinder(L, 5)
    end
end

@testset "InfiniteHelix" begin
    for L in 2:4, N in 1:3
        lattice = InfiniteHelix(L, N)
        V = vertices(lattice)
        @test length(lattice) == length(V) == N
        @test lattice[1, 1] == first(V)

        NN = nearest_neighbours(lattice)
        @test length(NN) == 4 * length(V) / 2 # coordination number 4
        min_dist = 1
        for (i, j) in NN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)

        NNN = next_nearest_neighbours(lattice)
        @test length(NNN) == 2 * length(V) # coordination number 4
        min_dist = sqrt(2)
        for (i, j) in NNN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NNN)
    end
end

@testset "InfiniteStrip" begin
    for L in 2:4
        lattice = InfiniteStrip(L)
        V = vertices(lattice)
        @test length(lattice) == length(V) == L
        @test lattice[1, 1] == first(V)

        NN = nearest_neighbours(lattice)
        @test length(NN) == 2L - 1 # coordination number 4 - 1 boundary
        @test allunique(NN)

        NNN = next_nearest_neighbours(lattice)
        @test length(NNN) == 2L - 2 # coordination number 4 - 2 boundary
        @test allunique(NNN)

        for N in (2L, 3L)
            lattice = InfiniteStrip(L, N)
            V = vertices(lattice)
            @test length(lattice) == length(V) == N

            NN = nearest_neighbours(lattice)
            @test length(NN) == 2N - (N ÷ L) # coordination number 4
            @test allunique(NN)

            NNN = next_nearest_neighbours(lattice)
            @test length(NNN) == 2N - 2(N ÷ L) # coordination number 4
            @test allunique(NNN)
        end

        @test_throws ArgumentError InfiniteStrip(L, 5)
    end
end

@testset "InfiniteHoneycombYC" begin
    for L in 4:4:12, N in 1:3
        lattice = HoneycombYC(L, N * L)
        V = vertices(lattice)
        @test length(lattice) == length(V) == N * L

        NN = nearest_neighbours(lattice)
        @test length(NN) == 3 * length(V) / 2 # coordination number 3

        min_dist = 1
        for (i, j) in NN
            @test norm(i - j) ≈ min_dist
        end
        @test allunique(NN)
    end

    @test_throws ArgumentError HoneycombYC(3)
    @test_throws ArgumentError HoneycombYC(4, 6)
end

@testset "FiniteCylinder" begin
    for L in 2:5
        lattice = FiniteCylinder(L)
        @test length(nearest_neighbours(lattice)) == L
        @test length(next_nearest_neighbours(lattice)) == 0
        for n in 2:4
            lattice = FiniteCylinder(L, n * L)
            V = vertices(lattice)
            @test length(lattice) == length(V) == n * L
            @test lattice[1, 1] == first(V)

            NN = nearest_neighbours(lattice)
            @test length(NN) == 2 * n * L - L
            @test allunique(NN)

            NNN = next_nearest_neighbours(lattice)
            @test length(NNN) == 2 * (n - 1) * L

            @test allunique(NNN)

            @test_throws ArgumentError FiniteCylinder(L, n * L + 1)
        end
    end
end

@testset "FiniteStrip" begin
    for L in 2:8, n in 2:4
        N = n * L
        lattice = FiniteStrip(L, N)
        V = vertices(lattice)

        # Test the number of vertices
        @test length(lattice) == length(V) == N

        # Test the first vertex
        @test lattice[1, 1] == first(V)

        # Test nearest neighbors
        NN = nearest_neighbours(lattice)
        @test length(NN) == 2N - L - n # coordination number 4 - edge effects
        @test allunique(NN)

        # Test next-nearest neighbors
        NNN = next_nearest_neighbours(lattice)
        @test length(NNN) == 2N - 2L - (2n - 2) # coordination number 4 - edge effects
        @test allunique(NNN)
    end
end
