using Pkg
Pkg.activate(".")

using ADIPoisson, DelimitedFiles, BlockArrays, FillArrays, LinearAlgebra
using AppleAccelerate # Use on Macs to get fastest solves


adi_buildtm = Dict()
adi_solvtm = Dict()
ns = 2 .^ (1:4)
# pps = n -> 2 .^ (2:round(Int, log2(20_000 / n)))
pps = n -> 2 .^ (2:round(Int, log2(20_000 / n)))
f = (x,y) -> -2 .*sin.(π*x) .* (2π*y .*cos.(π*y) .+ (1-π^2*y^2) .*sin.(π*y))

# compile
let n = 2, p = 4
    Δ⁻¹ = plan_poissonsolve!(n, p)
    # (x,y),pl_P,pl_Q = plan_poisson_transform(n, p)
    # vals = f.(x,reshape(y,1,1,size(y)...))
    # cfs = pl_P * vals

    cfs = PseudoBlockArray(randn(n*(p+1),n*(p+1)), Fill(n,p+1), Fill(n,p+1))
    Δ⁻¹ * cfs
end

for n in ns, p in pps(n) # truncation degree on each cell
    @show (n,p)
    GC.gc()
    GC.enable(false)
    adi_buildtm[(n,p)] = @elapsed(Δ⁻¹ = plan_poissonsolve!(n, p))
    GC.enable(true)
    cfs = PseudoBlockArray(randn(n*(p+1),n*(p+1)), Fill(n,p+1), Fill(n,p+1))
    GC.gc()
    GC.enable(false)
    adi_solvtm[(n,p)] = @elapsed(Δ⁻¹ * cfs)
    GC.enable(true)
    GC.gc()
end

for n in ns
    ps = pps(n)
    data = [ps getindex.(Ref(adi_buildtm),tuple.(n,ps)) getindex.(Ref(adi_solvtm),tuple.(n,ps))]
    writedlm("data/2dtims_n_$n.csv", data)
end