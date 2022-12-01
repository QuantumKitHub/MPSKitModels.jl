# The `@mpoham` macro

```@meta
CurrentModule = TensorKit
```

When dealing with (quasi-) one-dimensional systems that are defined by a sum of local operators, a convenient representation exists in terms of a sparse matrix product operator with an upper diagonal structure (`MPOHamiltonian`).
The generation of such an object starting from a sum of local operators is facilitated by the macro `@mpoham`, which provides several syntactic sugar features.
 
```@docs
@mpoham
```
