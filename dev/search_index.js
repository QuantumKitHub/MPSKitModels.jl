var documenterSearchIndex = {"docs":
[{"location":"package_index/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"package_index/","page":"Index","title":"Index","text":"","category":"page"},{"location":"man/mpoham/#The-@mpoham-macro","page":"The @mpoham macro","title":"The @mpoham macro","text":"","category":"section"},{"location":"man/mpoham/","page":"The @mpoham macro","title":"The @mpoham macro","text":"CurrentModule = TensorKit","category":"page"},{"location":"man/mpoham/","page":"The @mpoham macro","title":"The @mpoham macro","text":"When dealing with (quasi-) one-dimensional systems that are defined by a sum of local operators, a convenient representation exists in terms of a sparse matrix product operator with an upper diagonal structure (MPOHamiltonian). The generation of such an object starting from a sum of local operators is facilitated by the macro @mpoham, which provides several syntactic sugar features.","category":"page"},{"location":"man/mpoham/","page":"The @mpoham macro","title":"The @mpoham macro","text":"@mpoham","category":"page"},{"location":"man/mpoham/#MPSKitModels.@mpoham","page":"The @mpoham macro","title":"MPSKitModels.@mpoham","text":"@mpoham(block)\n\nSpecify a Matrix Product Operator that is represented by a sum of local operators.\n\nThis macro converts expressions of the form O{i...} to an operator acting on sites i... where O is an operator, and i can be an integer or a lattice point. The macro recognizes expressions of the following forms:\n\nO{i...} to indicate local operators O acting on sites i...\n-Inf:Inf, -∞:∞, -Inf:step:Inf, -∞:step:∞ to indicate infinite chains.\n1:L to indicate finite chains.\n∑ to represent sums.\n\nExamples\n\nH_ising = @mpoham sum(σᶻᶻ{i, i+1} + h * σˣ{i} for i in -Inf:Inf)\nH_heisenberg = @mpoham ∑(sigma_exchange(){i,j} for (i,j) in nearest_neighbours(-∞:∞))\n\n\n\n\n\n","category":"macro"},{"location":"man/models/#Models","page":"Models","title":"Models","text":"","category":"section"},{"location":"man/models/","page":"Models","title":"Models","text":"CurrentModule = MPSKitModels","category":"page"},{"location":"man/models/#(11)-dimensional-Quantum-Hamiltonians","page":"Models","title":"(1+1)-dimensional Quantum Hamiltonians","text":"","category":"section"},{"location":"man/models/","page":"Models","title":"Models","text":"transverse_field_ising\nkitaev_model\nheisenberg_XXX\nheisenberg_XXZ\nheisenberg_XYZ\nbilinear_biquadratic_model\nhubbard_model\nbose_hubbard_model\nquantum_chemistry_hamiltonian","category":"page"},{"location":"man/models/#MPSKitModels.transverse_field_ising","page":"Models","title":"MPSKitModels.transverse_field_ising","text":"transverse_field_ising([elt::Type{<:Number}], [symmetry::Type{<:Sector}],\n                       [lattice::AbstractLattice]; J=1.0, g=1.0)\n\nMPO for the hamiltonian of the one-dimensional Transverse-field Ising model, as defined by\n\nH = -Jleft(sum_langle ij rangle sigma^z_i sigma^z_j + g sum_i sigma^x_i right)\n\nwhere the sigma^i are the spin-1/2 Pauli operators. Possible values for the symmetry are Trivial, Z2Irrep or FermionParity.\n\nBy default, the model is defined on an infinite chain with unit lattice spacing, with Trivial symmetry and with ComplexF64 entries of the tensors.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.kitaev_model","page":"Models","title":"MPSKitModels.kitaev_model","text":"kitaev_model([elt::Type{<:Number}], [lattice::AbstractLattice];\n             t=1.0, mu=1.0, Delta=1.0)\n\nMPO for the hamiltonian of the Kitaev model, as defined by\n\nH = sum_langle ij rangle left(-fract2(c_i^+ c_j^- + c_j^+c_i^-) + fracΔ2(c_i^+c_j^+ + c_j^-c_i^-) right) - mu sum_i c_i^+ c_i^-\n\nBy default, the model is defined on an infinite chain with unit lattice spacing and with ComplexF64 entries of the tensors.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.heisenberg_XXX","page":"Models","title":"MPSKitModels.heisenberg_XXX","text":"heisenberg_XXX([elt::Type{<:Number}], [symmetry::Type{<:Sector}],\n                       [lattice::AbstractLattice]; J=1.0, spin=1)\n\nMPO for the hamiltonian of the isotropic Heisenberg model, as defined by\n\nH = J sum_langle ij rangle vecS_i cdot vecS_j\n\nwhere vecS = (S^x S^y S^z).\n\nBy default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with ComplexF64 entries of the tensors.\n\nSee also heisenberg_XXZ and heisenberg_XYZ.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.heisenberg_XXZ","page":"Models","title":"MPSKitModels.heisenberg_XXZ","text":"heisenberg_XXZ([elt::Type{<:Number}], [symmetry::Type{<:Sector}],\n               [lattice::AbstractLattice]; J=1.0, Delta=1.0, spin=1)\n\nMPO for the hamiltonian of the XXZ Heisenberg model, as defined by\n\nH = J left( sum_langle ij rangle S_i^x S_j^x + S_i^y S_j^y + Delta S_i^z S_j^z right)\n\nBy default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with ComplexF64 entries of the tensors.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.heisenberg_XYZ","page":"Models","title":"MPSKitModels.heisenberg_XYZ","text":"heisenberg_XYZ([elt::Type{<:Number}], [lattice::AbstractLattice];\n    Jx=1.0, Jy=1.0, Jz=1.0, spin=1)\n\nMPO for the hamiltonian of the XYZ Heisenberg model, defined by\n\nH = sum_langle ij rangle left( J^x S_i^x S_j^x + J^y S_i^y S_j^y + J^z S_i^z S_j^z right)\n\nBy default, the model is defined on an infinite chain with unit lattice spacing and with ComplexF64 entries of the tensors.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.bilinear_biquadratic_model","page":"Models","title":"MPSKitModels.bilinear_biquadratic_model","text":"bilinear_biquadratic_model([elt::Type{<:Number}], [symmetry::Type{<:Sector}],\n                           [lattice::AbstractLattice]; spin=1, J=1.0, θ=0.0)\n\nMPO for the hamiltonian of the bilinear biquadratic Heisenberg model, as defined by\n\nH = J sum_langle ij rangle left(cos(theta) vecS_i cdot vecS_j + sin(theta) left( vecS_i cdot vecS_j right)^2 right)\n\nwhere vecS = (S^x S^y S^z).\n\nBy default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with ComplexF64 entries of the tensors.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.hubbard_model","page":"Models","title":"MPSKitModels.hubbard_model","text":"hubbard_model([elt::Type{<:Number}], [particle_symmetry::Type{<:Sector}],\n              [spin_symmetry::Type{<:Sector}], [lattice::AbstractLattice];\n              t, U, mu, n)\n\nMPO for the hamiltonian of the Hubbard model, as defined by\n\nH = -t sum_langle ij rangle sum_sigma left( e_isigma^+ e_jsigma^- + c_isigma^- c_jsigma^+ right) + U sum_i n_iuparrown_idownarrow - sum_i mu n_i\n\nwhere sigma is a spin index that can take the values uparrow or downarrow and n is the fermionic number operator e_number.\n\nBy default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with ComplexF64 entries of the tensors. If the particle_symmetry is not Trivial, a fixed particle number density n can be imposed.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.bose_hubbard_model","page":"Models","title":"MPSKitModels.bose_hubbard_model","text":"bose_hubbard_model([elt::Type{<:Number}], [symmetry::Type{<:Sector}],\n                   [lattice::AbstractLattice];\n                   cutoff, t, U, mu, n)\n\nMPO for the hamiltonian of the Bose-Hubbard model, as defined by\n\nH = -t sum_langle ij rangle left( a_i^+ a_j^- + a_i^- a_j^+ right) - mu sum_i N_i + fracU2 sum_i N_i(N_i - 1)\n\nwhere N is the bosonic number operator a_number.\n\nBy default, the model is defined on an infinite chain with unit lattice spacing, without any symmetries and with ComplexF64 entries of the tensors. The Hilbert space is truncated such that at maximum of cutoff bosons can be at a single site. If the symmetry is not Trivial, a fixed (halfinteger) particle number density n can be imposed.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.quantum_chemistry_hamiltonian","page":"Models","title":"MPSKitModels.quantum_chemistry_hamiltonian","text":"quantum_chemistry_hamiltonian(E0::Number, K::AbstractMatrix{<:Number}, V::AbstractArray{<:Number,4}, [elt::Type{<:Number}=ComplexF64])\n\nMPO for the quantum chemistry Hamiltonian, with kinetic energy K and potential energy V. The Hamiltonian is given by\n\nH = E0 + sum_ij sum_s Kij e_is^+ e_js^- + sum_ijkl sum_st Vijkl e_is^+ e_jt^+ e_kt^- e_ls^-\n\nwhere s and t are spin indices, which can be uparrow or downarrow. The full hamiltonian has U₁ ⊠ SU₂ ⊠ FermionParity symmetry.\n\nnote: Note\nThis should not be regarded as state-of-the-art quantum chemistry DMRG code. It is only meant to demonstrate the use of MPSKit for quantum chemistry. In particular:No attempt was made to incorporate spacegroup symmetries\nMPSKit does not contain many required algorithms in quantum chemistry (orbital ordering/optimization)\nMPOHamiltonian is not well suited for quantum chemistry\n\n\n\n\n\n","category":"function"},{"location":"man/models/#(20)-dimensional-Statistical-Mechanics","page":"Models","title":"(2+0)-dimensional Statistical Mechanics","text":"","category":"section"},{"location":"man/models/","page":"Models","title":"Models","text":"classical_ising\nsixvertex\nhard_hexagon\nqstate_clock","category":"page"},{"location":"man/models/#MPSKitModels.classical_ising","page":"Models","title":"MPSKitModels.classical_ising","text":"classical_ising([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial];\n                beta=log(1+sqrt(2))/2)\n\nMPO for the partition function of the two-dimensional classical Ising model, defined as\n\nmathcalZ(beta) = sum_s exp(-beta H(s)) text with  H(s) = -sum_langle i j rangle s_i s_j\n\n\nwhere each classical spin can take the values s = pm 1.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.sixvertex","page":"Models","title":"MPSKitModels.sixvertex","text":"sixvertex([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial];\n          a=1.0, b=1.0, c=1.0)\n\nMPO for the partition function of the two-dimensional six vertex model.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.hard_hexagon","page":"Models","title":"MPSKitModels.hard_hexagon","text":"hard_hexagon([elt::Type{<:Number}=ComplexF64])\n\nMPO for the partition function of the two-dimensional hard hexagon model.\n\n\n\n\n\n","category":"function"},{"location":"man/models/#MPSKitModels.qstate_clock","page":"Models","title":"MPSKitModels.qstate_clock","text":"qstate_clock([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial]; beta::Number=1.0, q::Integer=3)\n\nMPO for the partition function of the two-dimensional discrete clock model with q states.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#Operators","page":"Operators","title":"Operators","text":"","category":"section"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"CurrentModule = MPSKitModels","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"There are several different operators defined, which all follow an interface similar to the following:","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"operator([elt::Type{<:Number}=ComplexF64], [symmetry::Type{<:Sector}=Trivial]; kwargs...)","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"Here, the scalar type of the operator is defined by elt, while the symmetry can be chosen through the symmetry argument. Other parameters are supplied as keywords. The special keyword argument side can be used for operators that require an additional virtual space to satisfy the symmetry constraints, in which case it determines where this auxiliary space is located, either to the left :L (default) or to the right :R.","category":"page"},{"location":"man/operators/#Spin-operators","page":"Operators","title":"Spin operators","text":"","category":"section"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"The spin operators S_x, S_y and S_z are defined such that they obey the spin commutation relations S^j S^k = i varepsilon_jkl S^l. Additionally, the ladder operators are defined as S^pm = S^x pm i S^y. Several combinations are defined that act on two spins.","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"Supported values of symmetry for spin operators are Trivial, Z2Irrep and U1Irrep.  When imposing symmetries, by convention we choose S_z as the diagonal operator for mathrmU(1), and S_x as the diagonal operator for mathbbZ_2.","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"S_x\nS_y\nS_z\nS_plus\nS_min\nS_xx\nS_yy\nS_zz\nS_plusmin\nS_minplus\nS_exchange","category":"page"},{"location":"man/operators/#MPSKitModels.S_x","page":"Operators","title":"MPSKitModels.S_x","text":"S_x([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nSˣ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin operator along the x-axis.\n\nSee also σˣ\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_y","page":"Operators","title":"MPSKitModels.S_y","text":"S_y([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nSʸ([eltype::Type{<:Complex}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin operator along the y-axis.\n\nSee also σʸ\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_z","page":"Operators","title":"MPSKitModels.S_z","text":"S_z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nSᶻ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin operator along the z-axis. Possible values for symmetry are Trivial, Z2Irrep, and U1Irrep.\n\nSee also σᶻ\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_plus","page":"Operators","title":"MPSKitModels.S_plus","text":"S_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nS⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin plus operator.\n\nSee also σ⁺\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_min","page":"Operators","title":"MPSKitModels.S_min","text":"S_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nS⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin minus operator.\n\nSee also σ⁻\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_xx","page":"Operators","title":"MPSKitModels.S_xx","text":"S_xx([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nSˣˣ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin xx exchange operator.\n\nSee also σˣˣ\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_yy","page":"Operators","title":"MPSKitModels.S_yy","text":"S_yy([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nSʸʸ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin yy exchange operator.\n\nSee also σʸʸ\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_zz","page":"Operators","title":"MPSKitModels.S_zz","text":"S_zz([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nSᶻᶻ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin zz exchange operator.\n\nSee also σᶻᶻ\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_plusmin","page":"Operators","title":"MPSKitModels.S_plusmin","text":"S_plusmin([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nS⁺⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin plusmin exchange operator.\n\nSee also σ⁺⁻\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_minplus","page":"Operators","title":"MPSKitModels.S_minplus","text":"S_minplus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nS⁻⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin minplus exchange operator.\n\nSee also σ⁻⁺\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.S_exchange","page":"Operators","title":"MPSKitModels.S_exchange","text":"S_exchange([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\nSS([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; spin=1 // 2)\n\nThe spin exchange operator.\n\nSee also σσ\n\n\n\n\n\n","category":"function"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"For convenience, the Pauli matrices can also be recovered as σⁱ = 2 Sⁱ.","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"σˣ\nσʸ\nσᶻ\nσ⁺\nσ⁻\nσˣˣ\nσʸʸ\nσᶻᶻ\nσ⁺⁻\nσ⁻⁺\nσσ","category":"page"},{"location":"man/operators/#MPSKitModels.σˣ","page":"Operators","title":"MPSKitModels.σˣ","text":"Pauli x operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σʸ","page":"Operators","title":"MPSKitModels.σʸ","text":"Pauli y operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σᶻ","page":"Operators","title":"MPSKitModels.σᶻ","text":"Pauli z operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σ⁺","page":"Operators","title":"MPSKitModels.σ⁺","text":"Pauli plus operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σ⁻","page":"Operators","title":"MPSKitModels.σ⁻","text":"Pauli minus operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σˣˣ","page":"Operators","title":"MPSKitModels.σˣˣ","text":"Pauli xx operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σʸʸ","page":"Operators","title":"MPSKitModels.σʸʸ","text":"Pauli yy operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σᶻᶻ","page":"Operators","title":"MPSKitModels.σᶻᶻ","text":"Pauli zz operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σ⁺⁻","page":"Operators","title":"MPSKitModels.σ⁺⁻","text":"Pauli plusmin operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σ⁻⁺","page":"Operators","title":"MPSKitModels.σ⁻⁺","text":"Pauli minplus operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.σσ","page":"Operators","title":"MPSKitModels.σσ","text":"Pauli exchange operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#Q-state-Potts-operators","page":"Operators","title":"Q-state Potts operators","text":"","category":"section"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"The Q-state Potts operators potts_X and potts_Z are defined to fulfill the braiding relation ZX = omega XZ with omega = e^2pi iQ. ","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"Supported values of symmetry for the X operator are Trivial and ZNIrrep{Q}, while for the Z operator only Trivial is supported.","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"potts_X\npotts_Z\npotts_ZZ","category":"page"},{"location":"man/operators/#MPSKitModels.potts_X","page":"Operators","title":"MPSKitModels.potts_X","text":"potts_X([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; Q=3)\n\nThe Potts X operator, also known as the shift operator, where X^q=1.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.potts_Z","page":"Operators","title":"MPSKitModels.potts_Z","text":"potts_Z([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; Q=3)\n\nThe Potts Z operator, also known as the clock operator, where Z^q=1.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.potts_ZZ","page":"Operators","title":"MPSKitModels.potts_ZZ","text":"potts_ZZ([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; q=3)\n\nThe Potts operator Z  Z, where Z^q = 1.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#Bosonic-operators","page":"Operators","title":"Bosonic operators","text":"","category":"section"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"The bosonic creation and annihilation operators a_plus (a^+) and a_min (a^-) are defined such that the following holds:","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"a^+ leftnright = sqrtn + 1 leftn+1right","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"a^- leftnright = sqrtn leftn-1right","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"From these, a number operator a_number (N) can be defined:","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"N = a^+ a^-","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"Nleftnright = n leftnright","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"With these, the following commutators can be obtained:","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"lefta^- a^+right = 1","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"leftN a^+right = a^+","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"leftN a^-right = -a^-","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"Supported values of symmetry for bosonic operators are Trivial and U1Irrep.","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"a_plus\na_min\na_number","category":"page"},{"location":"man/operators/#MPSKitModels.a_plus","page":"Operators","title":"MPSKitModels.a_plus","text":"a_plus([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)\na⁺([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)\n\nThe truncated bosonic creation operator, with a maximum of cutoff bosons per site.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.a_min","page":"Operators","title":"MPSKitModels.a_min","text":"a_min([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)\na⁻([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)\n\nThe truncated bosonic annihilation operator, with a maximum of cutoff bosons per site.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.a_number","page":"Operators","title":"MPSKitModels.a_number","text":"a_number([eltype::Type{<:Number}], [symmetry::Type{<:Sector}]; cutoff=5)\n\nThe truncated bosonic number operator, with a maximum of cutoff bosons per site.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#Fermionic-operators","page":"Operators","title":"Fermionic operators","text":"","category":"section"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"Spinless fermions.","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"c_plus\nc_min\nc_number","category":"page"},{"location":"man/operators/#MPSKitModels.c_plus","page":"Operators","title":"MPSKitModels.c_plus","text":"c_plus([elt::Type{<:Number}=ComplexF64]; side=:L)\nc⁺([elt::Type{<:Number}=ComplexF64]; side=:L)\n\nFermionic creation operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.c_min","page":"Operators","title":"MPSKitModels.c_min","text":"c_min([elt::Type{<:Number}=ComplexF64]; side=:L)\nc⁻([elt::Type{<:Number}=ComplexF64]; side=:L)\n\nFermionic annihilation operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/#MPSKitModels.c_number","page":"Operators","title":"MPSKitModels.c_number","text":"c_number([elt::Type{<:Number}=ComplexF64])\n\nFermionic number operator.\n\n\n\n\n\n","category":"function"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"Spinful fermions.","category":"page"},{"location":"man/operators/","page":"Operators","title":"Operators","text":"e_plus\ne_min\ne_number","category":"page"},{"location":"man/operators/#MPSKitModels.HubbardOperators.e_number","page":"Operators","title":"MPSKitModels.HubbardOperators.e_number","text":"e_number(T, particle_symmetry::Type{<:Sector}, spin_symmetry::Type{<:Sector})\n\nReturn the one-body operator that counts the number of particles.\n\n\n\n\n\n","category":"function"},{"location":"#MPSKitModels.jl","page":"Home","title":"MPSKitModels.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Operators, models and QOL for working with MPSKit.jl","category":"page"},{"location":"#Table-of-contents","page":"Home","title":"Table of contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Pages = [\"home.md\", \"man/operators.md\", \"man/mpoham.md\", \"man/models.md\", \"index.md\"]\nDepth = 4","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Install with the package manager:","category":"page"},{"location":"","page":"Home","title":"Home","text":"import Pkg\nPkg.add(\"MPSKitModels\")","category":"page"},{"location":"#Package-features","page":"Home","title":"Package features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A macro @mpoham for conveniently specifying (quasi-) 1D hamiltonians.\nA list of predefined operators, optionally with enforced symmetry.\nA list of predefined models","category":"page"},{"location":"","page":"Home","title":"Home","text":"MPSKitModels.jl is centered around specifying MPOs through the combination of local operators that act on a finite number of sites, along with a specification of allowed sites. The former are implemented using AbstractTensorMaps from TensorKit.jl, while the latter are defined through some geometry, such as a chain, strip or cylinder, and some notion of neighbours on this geometry. Additionally, several commonly used models are provided.","category":"page"},{"location":"#To-do-list","page":"Home","title":"To do list","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Add support for finite systems\nAdd support for non-local operators and partition functions","category":"page"},{"location":"man/lattices/#Lattices","page":"Lattices","title":"Lattices","text":"","category":"section"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"CurrentModule = MPSKitModels","category":"page"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"Models can be defined on different lattices, and several lattices lend themselves to a description in terms of a (quasi-)one-dimensional operator. In order to facilitate this mapping, the combination of the @mpoham macro and the lattices in this package provides an easy interface.","category":"page"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"AbstractLattice\nFiniteChain\nInfiniteChain\nInfiniteCylinder\nInfiniteHelix\nInfiniteStrip\nHoneycombYC","category":"page"},{"location":"man/lattices/#MPSKitModels.AbstractLattice","page":"Lattices","title":"MPSKitModels.AbstractLattice","text":"AbstractLattice{N}\n\nAbstract supertype of all lattices, which are mapped to N-dimensional integer grids.\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/#MPSKitModels.FiniteChain","page":"Lattices","title":"MPSKitModels.FiniteChain","text":"FiniteChain(length::Integer=1)\n\nA one-dimensional lattice of length L\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/#MPSKitModels.InfiniteChain","page":"Lattices","title":"MPSKitModels.InfiniteChain","text":"InfiniteChain(L::Integer=1)\n\nA one dimensional infinite lattice with a unit cell containing L sites.\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/#MPSKitModels.InfiniteCylinder","page":"Lattices","title":"MPSKitModels.InfiniteCylinder","text":"InfiniteCylinder(L::Int, N::Int)\n\nAn infinite cylinder with L sites per rung and N sites per unit cell. \n\n\n\n\n\n","category":"type"},{"location":"man/lattices/#MPSKitModels.InfiniteHelix","page":"Lattices","title":"MPSKitModels.InfiniteHelix","text":"InfiniteHelix(L::Integer, N::Integer)\n\nAn infinite helix with L sites per rung and N sites per unit cell.\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/#MPSKitModels.InfiniteStrip","page":"Lattices","title":"MPSKitModels.InfiniteStrip","text":"InfiniteStrip(L::Int, N::Int)\n\nAn infinite strip with L sites per rung and N sites per unit cell.\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/#MPSKitModels.HoneycombYC","page":"Lattices","title":"MPSKitModels.HoneycombYC","text":"HoneycombYC(L::Integer, N::Integer=L)\n\nA honeycomb lattice on an infinite cylinder with L sites per rung and N sites per unit cell. The y-axis is aligned along an edge of the hexagons, and the circumference is 3L4.\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"Having defined a lattice, it is possible to iterate over several points or combinations of points that can be of interest. Such a point is represented as a LatticePoint, which is defined in terms of an integer N-dimensional coordinate system representation, and supports addition and subtraction, both with other points or with tuples. These structures also handle the logic of being mapped to a one-dimensional system.","category":"page"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"LatticePoint\nlinearize_index\nvertices\nnearest_neighbours\nbipartition","category":"page"},{"location":"man/lattices/#MPSKitModels.LatticePoint","page":"Lattices","title":"MPSKitModels.LatticePoint","text":"LatticePoint{N,G}\n\nrepresents an N-dimensional point on a G lattice.\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/#MPSKitModels.linearize_index","page":"Lattices","title":"MPSKitModels.linearize_index","text":"linearize_index(lattice, indices...)\n\nconvert a given set of indices into a linear index.\n\n\n\n\n\n","category":"function"},{"location":"man/lattices/#MPSKitModels.vertices","page":"Lattices","title":"MPSKitModels.vertices","text":"vertices(lattice::AbstractLattice)\n\nconstruct an iterator over all lattice points.\n\n\n\n\n\n","category":"function"},{"location":"man/lattices/#MPSKitModels.nearest_neighbours","page":"Lattices","title":"MPSKitModels.nearest_neighbours","text":"nearest_neighbours(lattice::AbstractLattice)\n\nconstruct an iterator over all pairs of nearest neighbours.\n\n\n\n\n\n","category":"function"},{"location":"man/lattices/#MPSKitModels.bipartition","page":"Lattices","title":"MPSKitModels.bipartition","text":"bipartition(lattice)\n\nconstruct two iterators over the vertices of the bipartition of a given lattice.\n\n\n\n\n\n","category":"function"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"Sometimes it might be useful to change the order of the linear indices of certain lattices. In this case a wrapper around a lattice can be defined through the following:","category":"page"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"SnakePattern","category":"page"},{"location":"man/lattices/#MPSKitModels.SnakePattern","page":"Lattices","title":"MPSKitModels.SnakePattern","text":"SnakePattern(lattice, pattern)\n\nRepresents a given lattice with a linear order that is provided by pattern.\n\n\n\n\n\n","category":"type"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"Any mapping of linear indices can be used, but the following patterns can be helpful:","category":"page"},{"location":"man/lattices/","page":"Lattices","title":"Lattices","text":"backandforth_pattern\nfrontandback_pattern","category":"page"},{"location":"man/lattices/#MPSKitModels.backandforth_pattern","page":"Lattices","title":"MPSKitModels.backandforth_pattern","text":"backandforth_pattern(cylinder)\n\npattern that alternates directions between different rungs of a cylinder\n\n\n\n\n\n","category":"function"},{"location":"man/lattices/#MPSKitModels.frontandback_pattern","page":"Lattices","title":"MPSKitModels.frontandback_pattern","text":"frontandback_pattern(cylinder)\n\npattern that alternates between a site on the first half of a rung and a site on the second half of a rung.\n\n\n\n\n\n","category":"function"}]
}
