# Lattices

Models can be defined on different lattices, and several lattices lend themselves to a description in terms of a (quasi-)one-dimensional operator.
In order to facilitate this mapping, the combination of the `@mpoham` macro and the lattices in this package provides an easy interface.

```@docs
AbstractLattice
InfiniteChain
InfiniteCylinder
```

Having defined a lattice, it is possible to iterate over several points or combinations of points that can be of interest, using the following methods:

```@docs
vertices
nearest_neighbours
```