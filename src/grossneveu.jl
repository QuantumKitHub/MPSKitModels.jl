function su2u1_grossneveu(;g2SPT=0,g2AFM=0)
    ph       = Rep[SU₂×U₁]( (1//2,0)=>1, (0,-1)=>1, (0,1)=>1 )
    bigonleg = Rep[SU₂×U₁]( (0,0)=>1, (1//2,-1)=>1, (1//2,1)=>1 )
    unit     = oneunit(ph)

    LK = TensorMap(ones, ComplexF64, unit*ph, bigonleg*ph)
    blocks(LK)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]    =  [im*2/sqrt(2) 1]
    blocks(LK)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)] =  [1. im -im]
    blocks(LK)[Irrep[SU₂](0)⊠Irrep[U₁](1)]    =  [im*2/sqrt(2) 1]
    LK = permute(LK,(1,2),(4,3));

    RK = TensorMap(ones, ComplexF64, bigonleg*ph, unit*ph)
    blocks(RK)[Irrep[SU₂](0)⊠Irrep[U₁](-1)][:]    =  [2/sqrt(2) 1][:]
    blocks(RK)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)][:] =  [1 -1 1][:]
    blocks(RK)[Irrep[SU₂](0)⊠Irrep[U₁](1)][:]    =  [2/sqrt(2) 1][:]
    RK = permute(RK,(1,2),(4,3));

    Cplus = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)]  = [0 im*0.5 -im*0.5; -0.5 0 0; 0.5 0 0]
    blocks(Cplus)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]     = [0 0.5*sqrt(2); im*0.5*sqrt(2) 0]
    blocks(Cplus)[Irrep[SU₂](1)⊠Irrep[U₁](-1)]     = zeros(1,1)
    blocks(Cplus)[Irrep[SU₂](0)⊠Irrep[U₁](1)]     = [0 0.5*sqrt(2); im*0.5*sqrt(2) 0]
    blocks(Cplus)[Irrep[SU₂](1)⊠Irrep[U₁](1)]     = zeros(1,1)
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[U₁](-2)] = zeros(1,1)
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[U₁](2)]  = zeros(1,1)
    Cplus = permute(Cplus,(1,2),(4,3));

    Cmin = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
    blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)]  = conj([0 im*0.5 -im*0.5; -0.5 0 0; 0.5 0 0])
    blocks(Cmin)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]     = conj([0 0.5*sqrt(2); im*0.5*sqrt(2) 0])
    blocks(Cmin)[Irrep[SU₂](1)⊠Irrep[U₁](-1)]     = zeros(1,1)
    blocks(Cmin)[Irrep[SU₂](0)⊠Irrep[U₁](1)]     = conj([0 0.5*sqrt(2); im*0.5*sqrt(2) 0])
    blocks(Cmin)[Irrep[SU₂](1)⊠Irrep[U₁](1)]     = zeros(1,1)
    blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[U₁](-2)] = zeros(1,1)
    blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[U₁](2)]  = zeros(1,1)
    Cmin = permute(Cmin,(1,2),(4,3));

    f1 = isomorphism(fuse(unit, unit), unit*unit)
    f2 = isomorphism(bigonleg*bigonleg, fuse(bigonleg, bigonleg))
    f3 = isomorphism(fuse(bigonleg, bigonleg), bigonleg*bigonleg)
    f4 = isomorphism(unit*unit, fuse(unit, unit))

    @tensor Ldiffsq[-1 -2;-3 -4] := f1[-1,1,2]*LK[1,3,-3,5]*LK[2,-2,3,4]*f2[5,4,-4]
    @tensor Cdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*Cmin[1,3,-3,5]*Cmin[2,-2,3,4]*f2[5,4,-4]
    @tensor Rdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*RK[1,3,-3,5]*RK[2,-2,3,4]*f4[5,4,-4]

    #and now with the extra O(4) breaking part ie the O operator
    O_op = TensorMap(zeros, ComplexF64, unit*ph, ph*unit)
    blocks(O_op)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)] = -zeros(1,1)
    blocks(O_op)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]    =  -1*ones(1,1)
    blocks(O_op)[Irrep[SU₂](0)⊠Irrep[U₁](1)]    =  1*ones(1,1)

    MPOHamiltonian([LK, Cplus, RK]) +
    MPOHamiltonian([-0.25*g2SPT^2*Ldiffsq, Cdiffsq, Rdiffsq]) +
    MPOHamiltonian([-0.5*g2AFM^2*O_op*O_op]) +
    MPOHamiltonian([+0.5*g2AFM^2*O_op, O_op])
end

function su2u1_orderpars()
    ph       = Rep[SU₂×U₁]( (1//2,0)=>1, (0,-1)=>1, (0,1)=>1 )
    onleg    = Rep[SU₂×U₁]( (1//2,-1)=>1, (1//2,1)=>1 )
    unit     = oneunit(ph)

    #the K operator
    LK = TensorMap(ones, ComplexF64, unit*ph, onleg*ph)
    blocks(LK)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]   =  ones(1,1)*im*2/sqrt(2)
    blocks(LK)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)] =  [im -im]
    blocks(LK)[Irrep[SU₂](0)⊠Irrep[U₁](1)]    =  ones(1,1)*im*2/sqrt(2)
    LK = permute(LK,(1,2),(4,3));

    RK = TensorMap(ones, ComplexF64, onleg*ph, unit*ph)
    blocks(RK)[Irrep[SU₂](0)⊠Irrep[U₁](-1)][:]    =  ones(1,1)*2/sqrt(2)
    blocks(RK)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)][:] =  [-1, 1][:]
    blocks(RK)[Irrep[SU₂](0)⊠Irrep[U₁](1)][:]    =  ones(1,1)*2/sqrt(2)
    RK = permute(RK,(1,2),(4,3));

    #the Q operator
    LQ = TensorMap(ones, ComplexF64, unit*ph, onleg*ph)
    blocks(LQ)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]   =  1*ones(1,1)*2/sqrt(2)
    blocks(LQ)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)] =  [1 -1]
    blocks(LQ)[Irrep[SU₂](0)⊠Irrep[U₁](1)]    =  1*ones(1,1)*2/sqrt(2)
    LQ = permute(LQ,(1,2),(4,3));

    RQ = TensorMap(ones, ComplexF64, onleg*ph, unit*ph)
    blocks(RQ)[Irrep[SU₂](0)⊠Irrep[U₁](-1)][:]    =  -1*ones(1,1)*2/sqrt(2)
    blocks(RQ)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)][:] =  [1, 1][:]
    blocks(RQ)[Irrep[SU₂](0)⊠Irrep[U₁](1)][:]    =  1*ones(1,1)*2/sqrt(2)
    RQ = permute(RQ,(1,2),(4,3));

    #and now with the extra O(4) breaking part ie the O operator
    O_op = TensorMap(zeros, ComplexF64, unit*ph, ph*unit)
    blocks(O_op)[Irrep[SU₂](1//2)⊠Irrep[U₁](0)] = -zeros(1,1)
    blocks(O_op)[Irrep[SU₂](0)⊠Irrep[U₁](-1)]    =  -1*ones(1,1)
    blocks(O_op)[Irrep[SU₂](0)⊠Irrep[U₁](1)]    =  1*ones(1,1)


    return [LK, RK] , [O_op], [LQ, RQ]
end

function su2su2_grossneveu(;g=0.,v=0.)
    ph       = Rep[SU₂×SU₂]( (1//2,0)=>1, (0,1//2)=>1 )
    bigonleg = Rep[SU₂×SU₂]( (0,0)=>1, (1//2,1//2)=>1)
    smallonleg = Rep[SU₂×SU₂]( (1//2,1//2)=>1)
    unit     = oneunit(ph)

    LK = TensorMap(ones, ComplexF64, unit*ph, bigonleg*ph)
    blocks(LK)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)] =  [im*sqrt(2) 1]
    blocks(LK)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)] =  [1 im*sqrt(2)]
    LK = permute(LK,(1,2),(4,3));

    RK = TensorMap(ones, ComplexF64, bigonleg*ph, unit*ph)
    blocks(RK)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)][:] = [sqrt(2) 1]
    blocks(RK)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)][:] = [1 -sqrt(2)]
    RK = permute(RK,(1,2),(4,3));

    Cplus = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
    blocks(Cplus)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)] =  -sqrt(2)/2*[0 1; im 0]
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)] =  -sqrt(2)/2*[0 im; -1 0]
    blocks(Cplus)[Irrep[SU₂](1)⊠Irrep[SU₂](1//2)] =  zeros(1,1)
    blocks(Cplus)[Irrep[SU₂](1//2)⊠Irrep[SU₂](1)] =  zeros(1,1)
    Cplus = permute(Cplus,(1,2),(4,3));

    ham = MPOHamiltonian([LK, Cplus, RK])

    if !iszero(v)

        LK2 = TensorMap(ones, ComplexF64, unit*ph, smallonleg*ph)
        blocks(LK2)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)] =  [im*sqrt(2) 1]
        blocks(LK2)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)] =  [im*sqrt(2) 1]
        LK2 = permute(LK2,(1,2),(4,3));

        RK2 = TensorMap(ones, ComplexF64, smallonleg*ph, unit*ph)
        blocks(RK2)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)][:] = -[sqrt(2)]
        blocks(RK2)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)][:] = [sqrt(2)]
        RK2 = permute(RK2,(1,2),(4,3));

        Z = TensorMap(ones, ComplexF64, smallonleg*ph, smallonleg*ph)
        blocks(Z)[Irrep[SU₂](1)⊠Irrep[SU₂](1//2)] = -ones(1,1)
        blocks(Z)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)] = -ones(1,1)
        Z = permute(Z,(1,2),(4,3));

        ham += MPOHamiltonian([-0.5*v*LK2, Z, RK2])

    end
    if !iszero(g)

        Cmin = TensorMap(ones, ComplexF64, bigonleg*ph, bigonleg*ph)
        blocks(Cmin)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)] =  -conj(sqrt(2)/2*[0 1; im 0])
        blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)] =  -conj(sqrt(2)/2*[0 im; -1 0])
        blocks(Cmin)[Irrep[SU₂](1)⊠Irrep[SU₂](1//2)] =  zeros(1,1)
        blocks(Cmin)[Irrep[SU₂](1//2)⊠Irrep[SU₂](1)] =  zeros(1,1)
        Cmin = permute(Cmin,(1,2),(4,3));

        f1 = isomorphism(fuse(unit, unit), unit*unit)
        f2 = isomorphism(bigonleg*bigonleg, fuse(bigonleg, bigonleg))
        f3 = isomorphism(fuse(bigonleg, bigonleg), bigonleg*bigonleg)
        f4 = isomorphism(unit*unit, fuse(unit, unit))

        @tensor Ldiffsq[-1 -2;-3 -4] := f1[-1,1,2]*LK[1,3,-3,5]*LK[2,-2,3,4]*f2[5,4,-4]
        @tensor Cdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*Cmin[1,3,-3,5]*Cmin[2,-2,3,4]*f2[5,4,-4]
        @tensor Rdiffsq[-1 -2;-3 -4] := f3[-1,1,2]*RK[1,3,-3,5]*RK[2,-2,3,4]*f4[5,4,-4]

        ham += MPOHamiltonian([-0.25*g^2*Ldiffsq, Cdiffsq, Rdiffsq])

    end

    return ham
end

function su2su2_orderpars()
    ph       = Rep[SU₂×SU₂]( (1//2,0)=>1, (0,1//2)=>1 )
    onleg    = Rep[SU₂×SU₂]( (1//2,1//2)=>1)
    unit     = oneunit(ph)

    #the K operator
    LK2 = TensorMap(ones, ComplexF64, unit*ph, onleg*ph)
    blocks(LK2)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)] =  [im*sqrt(2) 1]
    blocks(LK2)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)] =  [im*sqrt(2) 1]
    LK2 = permute(LK2,(1,2),(4,3));

    RK2 = TensorMap(ones, ComplexF64, onleg*ph, unit*ph)
    blocks(RK2)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)][:] = -[sqrt(2)]
    blocks(RK2)[Irrep[SU₂](1//2)⊠Irrep[SU₂](0)][:] = [sqrt(2)]
    RK2 = permute(RK2,(1,2),(4,3));

    Z = TensorMap(ones, ComplexF64, onleg*ph, onleg*ph)
    blocks(Z)[Irrep[SU₂](1)⊠Irrep[SU₂](1//2)] = -ones(1,1)
    blocks(Z)[Irrep[SU₂](0)⊠Irrep[SU₂](1//2)] = -ones(1,1)
    Z = permute(Z,(1,2),(4,3));

    #=
    #the Q operator
    LQ = TensorMap(ones, ComplexF64, unit*ph, onleg*ph)
    blocks(LQ)[SU₂(0)×U₁(-1)]   =  1*ones(1,1)*2/sqrt(2)
    blocks(LQ)[SU₂(1//2)×U₁(0)] =  [1 -1]
    blocks(LQ)[SU₂(0)×U₁(1)]    =  1*ones(1,1)*2/sqrt(2)
    LQ = permute(LQ,(1,2),(4,3));

    RQ = TensorMap(ones, ComplexF64, onleg*ph, unit*ph)
    blocks(RQ)[SU₂(0)×U₁(-1)][:]    =  -1*ones(1,1)*2/sqrt(2)
    blocks(RQ)[SU₂(1//2)×U₁(0)][:] =  [1, 1][:]
    blocks(RQ)[SU₂(0)×U₁(1)][:]    =  1*ones(1,1)*2/sqrt(2)
    RQ = permute(RQ,(1,2),(4,3));

    #and now with the extra O(4) breaking part ie the O operator
    O_op = TensorMap(zeros, ComplexF64, unit*ph, ph*unit)
    blocks(O_op)[SU₂(1//2)×U₁(0)] = -zeros(1,1)
    blocks(O_op)[SU₂(0)×U₁(-1)]    =  -1*ones(1,1)
    blocks(O_op)[SU₂(0)×U₁(1)]    =  1*ones(1,1)
    =#

    return MPOHamiltonian([LK2, RK2]), MPOHamiltonian([LK2, Z, RK2])#, permute(RK2,(1,2),(4,3)) #, [O_op], [LQ, RQ]
end
