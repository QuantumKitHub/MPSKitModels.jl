"""
    LatticePoint{N,G}
    
represents an `N`-dimensional point on a `G` lattice.
"""
struct LatticePoint{N,G<:AbstractLattice{N}}
    coordinates::NTuple{N,Int}
    lattice::G
    function LatticePoint(coordinates::NTuple{N,Int}, lattice::AbstractLattice{N}) where {N}
        checkbounds(lattice, coordinates...)
        return new{N,typeof(lattice)}(coordinates, lattice)
    end
end

function LatticePoint(ind::Int, lattice::G) where {G<:AbstractLattice{1}}
    return LatticePoint((ind,), lattice)
end

function Base.show(io::IO, p::LatticePoint)
    return print(io, p.lattice, [p.coordinates...])
end

function Base.show(io::IO, ::MIME"text/plain", p::LatticePoint)
    if get(io, :typeinfo, Any) === typeof(p)
        print(io, [p.coordinates...])
    else
        print(io, p.lattice, [p.coordinates...])
    end
end

function Base.getindex(lattice::AbstractLattice{N}, inds::Vararg{Int,N}) where {N}
    return LatticePoint(inds, lattice)
end

Base.to_index(p::LatticePoint) = linearize_index(p)
linearize_index(p::LatticePoint) = linearize_index(p.lattice, p.coordinates...)

function Base.:+(i::LatticePoint{N,G}, j::LatticePoint{N,G}) where {N,G}
    i.lattice == j.lattice || throw(ArgumentError("lattices should be equal"))
    return LatticePoint(i.coordinates .+ j.coordinates, i.lattice)
end
function Base.:-(i::LatticePoint{N,G}, j::LatticePoint{N,G}) where {N,G}
    i.lattice == j.lattice || throw(ArgumentError("lattices should be equal"))
    return LatticePoint(i.coordinates .- j.coordinates, i.lattice)
end

function Base.:+(i::LatticePoint{N}, j::NTuple{N,Int}) where {N}
    return LatticePoint(i.coordinates .+ j, i.lattice)
end
Base.:+(i::LatticePoint{1}, j::Int) = i + LatticePoint(i.coordinates .+ j, i.lattice)
Base.:+(i::NTuple{N,Int}, j::LatticePoint{N}) where {N} = j + i
Base.:+(i::Int, j::LatticePoint{1}) = j + i

function Base.:-(i::LatticePoint{N}, j::NTuple{N,Int}) where {N}
    return LatticePoint(i.coordinates .+ j, i.lattice)
end
Base.:-(i::LatticePoint{1}, j::Int) = LatticePoint(i.coordinates .+ j, i.lattice)
function Base.:-(i::NTuple{N,Int}, j::LatticePoint{N}) where {N}
    return LatticePoint(i .- j.coordinates, j.lattice)
end
Base.:-(i::Int, j::LatticePoint{1}) = LatticePoint(i .- j, j.lattice)

Base.isless(i::L, j::L) where {L<:LatticePoint} = linearize_index(i) < linearize_index(j)

latticetype(::Union{LatticePoint{N,G},Type{<:LatticePoint{N,G}}}) where {N,G} = G
