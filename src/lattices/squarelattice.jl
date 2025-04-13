"""
    FiniteStrip(L::Int, N::Int)

A finite strip with a width of `L` and a total number of `N` sites.

This representes an `L` by `N÷L` rectangular patch.
"""
struct FiniteStrip <: AbstractLattice{2}
    L::Int
    N::Int
    function FiniteStrip(L::Integer, N::Integer=L)
        N > 0 || throw(ArgumentError("period should be positive"))
        mod(N, L) == 0 ||
            throw(ArgumentError("period should be a multiple of circumference"))
        return new(L, N)
    end
end
FiniteLadder(N::Integer) = FiniteStrip(2, N)
Base.axes(strip::FiniteStrip) = (1:(strip.L), (1:(strip.N ÷ strip.L)))
Base.isfinite(::Type{FiniteStrip}) = true

"""
    InfiniteStrip(L::Int, N::Int)

An infinite strip with `L` sites per rung and `N` sites per unit cell.
"""
struct InfiniteStrip <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteStrip(L::Integer, N::Integer=L)
        N > 0 || throw(ArgumentError("period should be positive"))
        mod(N, L) == 0 ||
            throw(ArgumentError("period should be a multiple of circumference"))
        return new(L, N)
    end
end
InfiniteLadder(N::Integer) = InfiniteStrip(2, N)
Base.axes(strip::InfiniteStrip) = (1:(strip.L), (-typemax(Int)):typemax(Int))
Base.isfinite(::Type{InfiniteStrip}) = false

"""
    FiniteCylinder(L::Int, N::Int)

A cylinder with circumference `L` and `N` sites in total.
"""
struct FiniteCylinder <: AbstractLattice{2}
    L::Int
    N::Int
    function FiniteCylinder(L::Integer, N::Integer=L)
        N > 0 || throw(ArgumentError("period should be positive"))
        mod(N, L) == 0 ||
            throw(ArgumentError("period should be a multiple of circumference"))
        return new(L, N)
    end
end

function Base.axes(cylinder::FiniteCylinder)
    return ((-typemax(Int)):typemax(Int), (1:(cylinder.N ÷ cylinder.L)))
end
Base.isfinite(::Type{FiniteCylinder}) = true

"""
    InfiniteCylinder(L::Int, N::Int)

An infinite cylinder with `L` sites per rung and `N` sites per unit cell. 
"""
struct InfiniteCylinder <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteCylinder(L::Integer, N::Integer=L)
        N > 0 || throw(ArgumentError("period should be positive"))
        mod(N, L) == 0 ||
            throw(ArgumentError("period should be a multiple of circumference"))
        return new(L, N)
    end
end

Base.axes(::InfiniteCylinder) = ((-typemax(Int)):typemax(Int), (-typemax(Int)):typemax(Int))
Base.isfinite(::Type{InfiniteCylinder}) = false

"""
    FiniteHelix(L::Integer, N::Integer)

A finite helix with `L` sites per rung and `N` sites in total.
"""
struct FiniteHelix <: AbstractLattice{2}
    L::Int
    N::Int
    function FiniteHelix(L::Integer, N::Integer=1)
        N > 0 || error("period should be positive")
        return new(L, N)
    end
end

Base.axes(helix::FiniteHelix) = ((-typemax(Int)):typemax(Int), (1:(helix.N ÷ helix.L)))
Base.isfinite(::Type{FiniteHelix}) = true

"""
    InfiniteHelix(L::Integer, N::Integer)

An infinite helix with `L` sites per rung and `N` sites per unit cell.
"""
struct InfiniteHelix <: AbstractLattice{2}
    L::Int
    N::Int
    function InfiniteHelix(L::Integer, N::Integer=1)
        N > 0 || error("period should be positive")
        return new(L, N)
    end
end

function Base.axes(::InfiniteHelix)
    return ((-typemax(Int)):typemax(Int), (-typemax(Int)):typemax(Int))
end
Base.isfinite(::Type{InfiniteHelix}) = false

############################################################################################

function linearize_index(lattice::FiniteStrip, i::Int, j::Int)
    @assert (1 <= i <= lattice.L && 1 <= j <= lattice.N ÷ lattice.L) "lattice point out of bounds"
    return i + lattice.L * (j - 1)
end
function linearize_index(lattice::InfiniteStrip, i::Int, j::Int)
    @assert 1 <= i <= lattice.L "lattice point out of bounds"
    return i + lattice.L * (j - 1)
end
function linearize_index(lattice::FiniteCylinder, i::Int, j::Int)
    @assert 1 <= j <= lattice.N ÷ lattice.L "lattice point out of bounds"
    return mod1(i, lattice.L) + lattice.L * (j - 1)
end
function linearize_index(lattice::InfiniteCylinder, i::Int, j::Int)
    return mod1(i, lattice.L) + lattice.L * (j - 1)
end
function linearize_index(helix::FiniteHelix, i::Int, j::Int)
    lin_ind = mod1(i, helix.L) + helix.L * (j + (i - 1) ÷ helix.L - 1)
    @assert (1 <= j <= lattice.N ÷ lattice.L && 1 <= lin_ind <= helix.N) "lattice point out of bounds"
    return mod1(i, helix.L) + helix.L * (j + (i - 1) ÷ helix.L - 1)
end
function linearize_index(helix::InfiniteHelix, i::Int, j::Int)
    return mod1(i, helix.L) + helix.L * (j + (i - 1) ÷ helix.L - 1)
end

function vertices(lattice::Union{FiniteStrip,InfiniteStrip,FiniteCylinder,InfiniteCylinder})
    return (LatticePoint((i, j), lattice) for i in 1:(lattice.L),
                                              j in 1:(lattice.N ÷ lattice.L))
end
function vertices(lattice::Union{FiniteHelix,InfiniteHelix})
    return (LatticePoint((i, 1), lattice) for i in 1:(lattice.N))
end

function nearest_neighbours(lattice::FiniteStrip)
    rows = lattice.L
    cols = lattice.N ÷ lattice.L
    horizontal = (LatticePoint((i, j), lattice) => LatticePoint((i, j + 1), lattice)
                  for i in 1:rows, j in 1:(cols - 1))
    vertical = (LatticePoint((i, j), lattice) => LatticePoint((i + 1, j), lattice)
                for i in 1:(rows - 1), j in 1:cols)
    return [horizontal..., vertical...]
end
function nearest_neighbours(lattice::FiniteCylinder)
    rows = lattice.L
    cols = lattice.N ÷ lattice.L
    horizontal = (LatticePoint((i, j), lattice) => LatticePoint((i, j + 1), lattice)
                  for i in 1:rows, j in 1:(cols - 1))
    vertical = (LatticePoint((i, j), lattice) => LatticePoint((i + 1, j), lattice)
                for i in 1:rows, j in 1:cols)
    return [horizontal..., vertical...]
end
function nearest_neighbours(lattice::FiniteHelix)
    rows = lattice.L
    cols = lattice.N ÷ lattice.L
    horizontal = (LatticePoint((i, j), lattice) => LatticePoint((i, j + 1), lattice)
                  for i in 1:rows, j in 1:(cols - 1))
    vertical = (LatticePoint((i, j), lattice) => LatticePoint((i + 1, j), lattice)
                for i in 1:rows, j in 1:cols if (i != rows && j != cols))
    return [horizontal..., vertical...]
end
function nearest_neighbours(lattice::Union{InfiniteStrip,InfiniteCylinder,InfiniteHelix})
    V = vertices(lattice)
    neighbours = Pair{eltype(V),eltype(V)}[]
    for v in V
        push!(neighbours, v => v + (0, 1))
        if v.coordinates[1] < lattice.L ||
           lattice isa InfiniteCylinder ||
           lattice isa InfiniteHelix
            push!(neighbours, v => v + (1, 0))
        end
    end
    return neighbours
end

function next_nearest_neighbours(lattice::AbstractLattice{2})
    diag1 = (i => i + (1, 1) for i in vertices(lattice) if checkbounds(Bool, lattice,
                                                                       (i.coordinates .+
                                                                        (1, 1))...))
    diag2 = (i => i + (1, -1) for i in vertices(lattice) if checkbounds(Bool, lattice,
                                                                        (i.coordinates .+
                                                                         (1, -1))...))
    return [diag1..., diag2...]
end
function next_nearest_neighbours(lattice::Union{InfiniteStrip,InfiniteCylinder,
                                                InfiniteHelix})
    V = vertices(lattice)
    neighbours = Pair{eltype(V),eltype(V)}[]
    for v in V
        if v.coordinates[1] < lattice.L ||
           lattice isa InfiniteCylinder ||
           lattice isa InfiniteHelix
            push!(neighbours, v => v + (1, 1))
        end
        if v.coordinates[1] > 1 || lattice isa InfiniteCylinder || lattice isa InfiniteHelix
            push!(neighbours, v => v + (-1, 1))
        end
    end
    return neighbours
end

LinearAlgebra.norm(p::LatticePoint{2,InfiniteStrip}) = LinearAlgebra.norm(p.coordinates)
function LinearAlgebra.norm(p::LatticePoint{2,InfiniteCylinder})
    return min(sqrt(mod(p.coordinates[1], p.lattice.L)^2 + p.coordinates[2]^2),
               sqrt(mod(-p.coordinates[1], p.lattice.L)^2 + p.coordinates[2]^2))
end
function LinearAlgebra.norm(p::LatticePoint{2,InfiniteHelix})
    x₁ = mod(p.coordinates[1], p.lattice.L)
    y₁ = p.coordinates[2] + (p.coordinates[1] ÷ p.lattice.L)
    x₂ = mod(-p.coordinates[1], p.lattice.L)
    y₂ = p.coordinates[2] - (p.coordinates[1] ÷ p.lattice.L)
    return min(sqrt(x₁^2 + y₁^2), sqrt(x₂^2 + y₂^2))
end
