
function nonsym_spintensors(s)
    (Sxd, Syd, Szd) = spinmatrices(s)
    sp = ComplexSpace(size(Sxd, 1))

    Sx = TensorMap(Sxd, sp, sp)
    Sy = TensorMap(Syd, sp, sp)
    Sz = TensorMap(Szd, sp, sp)

    return Sx, Sy, Sz, one(Sx)
end

"""
bosonic creation anihilation operators with a cutoff
cutoff = maximal number of bosons at one location
"""
function nonsym_bosonictensors(cutoff::Int, elt = ComplexF64)
    creadat = zeros(elt, cutoff + 1, cutoff + 1)

    for i in 1:cutoff
        creadat[i + 1, i] = sqrt(i)
    end

    a⁺ = TensorMap(creadat, ℂ^(cutoff + 1), ℂ^(cutoff + 1))
    a⁻ = TensorMap(collect(creadat'), ℂ^(cutoff + 1), ℂ^(cutoff + 1))
    return (a⁺, a⁻)
end
