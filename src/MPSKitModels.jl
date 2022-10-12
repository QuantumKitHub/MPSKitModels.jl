module MPSKitModels
    using TensorKit,MPSKit
    using LinearAlgebra:Diagonal,diag
    using MPSKit:@plansor;
    
    export spinmatrices,nonsym_spintensors,nonsym_bosonictensors
    include("utility.jl")

    export LocalOperator
    include("simpleham.jl")

    export nonsym_ising_ham,nonsym_ising_mpo,z2_ising_mpo
    include("ising.jl");

    export nonsym_xxz_ham,su2_xxx_ham,u1_xxz_ham
    include("xxz.jl");

    export su2u1_grossneveu,su2u1_orderpars,su2su2_grossneveu,su2su2_orderpars
    include("grossneveu.jl");

    export nonsym_qstateclock_mpo
    include("qstateclock.jl");

    export nonsym_qed_qlm_ham,qed_qlm_G2,u1_qed_ham
    include("qed_qlm.jl")

    export nonsym_sixvertex_mpo,u1_sixvertex_mpo,cu1_sixvertex_mpo
    include("sixvertex.jl")

    export U1_strip_harper_hofstadter
    include("hofstadter.jl")
end
