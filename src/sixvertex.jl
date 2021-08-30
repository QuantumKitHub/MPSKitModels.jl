function nonsym_sixvertex_mpo(a=1., b=1., c=1.)
  d = [a 0 0 0;
       0 c b 0;
       0 b c 0;
       0 0 0 a]
  return InfiniteMPO(permute(TensorMap(complex(d), ℂ^2 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^2),(1,2),(4,3)))
end

function u1_sixvertex_mpo(a=1., b=1., c=1.)
  sym = Rep[U₁](-1 => 1, 1 => 1)
  mpo =  TensorMap(zeros, ComplexF64, sym * sym, sym * sym)
  blocks(mpo)[Irrep[U₁](0)] = [b c; c b]
  blocks(mpo)[Irrep[U₁](2)] = reshape([a], (1, 1))
  blocks(mpo)[Irrep[U₁](-2)] = reshape([a], (1, 1))

  return InfiniteMPO(permute(mpo,(1,2),(4,3)))
end

function cu1_sixvertex_mpo(a=1., b=1., c=1.)
  sym = Rep[CU₁](1//2 => 1)
  mpo =  TensorMap(zeros, ComplexF64, sym * sym, sym * sym)
  blocks(mpo)[Irrep[CU₁](0, 0)] = reshape([b + c], (1, 1))
  blocks(mpo)[Irrep[CU₁](0, 1)] = reshape([-b + c], (1, 1))
  blocks(mpo)[Irrep[CU₁](1, 2)] = reshape([a], (1, 1))

  return InfiniteMPO(permute(mpo,(1,2),(4,3)))
end
