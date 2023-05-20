# Lattices

```@meta
CurrentModule = TensorKit
```

Models can be defined on different lattices, and several lattices lend themselves to a description in terms of a (quasi-)one-dimensional operator.
In order to facilitate this mapping, the combination of the `@mpoham` macro and the lattices in this package provides an easy interface.

```@docs
AbstractLattice
FiniteChain
InfiniteChain
InfiniteCylinder
InfiniteHelix
InfiniteStrip
HoneycombYC
```

Having defined a lattice, it is possible to iterate over several points or combinations of points that can be of interest.
Such a point is represented as a `LatticePoint`, which is defined in terms of an integer N-dimensional coordinate system representation, and supports addition and subtraction, both with other points or with tuples.
These structures also handle the logic of being mapped to a one-dimensional system.

```@docs
LatticePoint
linearize_index
vertices
nearest_neighbours
bipartition
```

Sometimes it might be useful to change the order of the linear indices of certain lattices.
In this case a wrapper around a lattice can be defined through the following:
```@docs
SnakePattern
```

Any mapping of linear indices can be used, but the following patterns can be helpful:
```@docs
backandforth_pattern
frontandback_pattern
```