function nonsym_ising_ham(;J = -1,spin = 1//2,lambda = 0.5,longit=0.0)
    (sx,sy,sz)=nonsym_spintensors(spin);
    id = one(sx);

    hamdat = Array{Union{Missing,typeof(sx)},3}(missing,1,3,3)
    hamdat[1,1,1] = id;
    hamdat[1,end,end] = id;
    hamdat[1,1,2] = J*sz;
    hamdat[1,2,end] = sz;
    hamdat[1,1,end] = lambda*sx+longit*sz;
    
    ham = MPOHamiltonian(hamdat);

    return ham
end

function nonsym_ising_mpo(;beta = log(1+sqrt(2))/2)
    t = [exp(beta) exp(-beta); exp(-beta) exp(beta)];

    r = eigen(t);
    nt = r.vectors*sqrt(Diagonal(r.values))*r.vectors;

    O = zeros(2,2,2,2);
    O[1,1,1,1]=1; O[2,2,2,2]=1;

    @tensor toret[-1 -2;-3 -4] := O[1,2,3,4]*nt[-1,1]*nt[-2,2]*nt[-3,3]*nt[-4,4];

    torett = TensorMap(complex(toret),ComplexSpace(2)*ComplexSpace(2),ComplexSpace(2)*ComplexSpace(2));
    return InfiniteMPO(torett);
end

function z2_ising_mpo(; beta = log(1+sqrt(2))/2)
  x = cosh(beta)
  y = sinh(beta)

  sec = ℤ₂Space(0 => 1, 1=> 1)
  mpo =  TensorMap(zeros, ComplexF64, sec * sec, sec * sec)
  blocks(mpo)[Irrep[ℤ₂](0)] = [2x^2 2x*y; 2x*y 2y^2]
  blocks(mpo)[Irrep[ℤ₂](1)] = [2x*y 2x*y; 2x*y 2x*y]

  return InfiniteMPO(mpo);
end
