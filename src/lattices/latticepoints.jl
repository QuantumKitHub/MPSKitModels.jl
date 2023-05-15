"""
    LatticePoint{N,G}
    
represents an `N`-dimensional point on a `G` lattice.
"""
struct LatticePoint{N,G<:AbstractLattice{N}}
    coordinates::NTuple{N,Int}
    lattice::G
end

LatticePoint(ind::Int, lattice::G) where {G<:AbstractLattice{1}} =
    LatticePoint{1,G}((ind,), lattice)

function Base.show(io::IO, p::LatticePoint)
    print(io, p.lattice, [p.coordinates...])
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

Base.:+(i::LatticePoint{N}, j::NTuple{N,Int}) where {N} = i + LatticePoint(j, i.lattice)
Base.:+(i::LatticePoint{1}, j::Int) = i + LatticePoint(j, i.lattice)
Base.:+(i::NTuple{N,Int}, j::LatticePoint{N}) where {N} = LatticePoint(i, j.lattice) + j
Base.:+(i::Int, j::LatticePoint{1}) = LatticePoint(i, j.lattice) + j

Base.:-(i::LatticePoint{N}, j::NTuple{N,Int}) where {N} = i - LatticePoint(j, i.lattice)
Base.:-(i::LatticePoint{1}, j::Int) = i - LatticePoint(j, i.lattice)
Base.:-(i::NTuple{N,Int}, j::LatticePoint{N}) where {N} = LatticePoint(i, j.lattice) - j
Base.:-(i::Int, j::LatticePoint{1}) = LatticePoint(i, j.lattice) - j

Base.isless(i::L, j::L) where {L <: LatticePoint} = linearize_index(i) < linearize_index(j)