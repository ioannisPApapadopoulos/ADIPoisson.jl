using Pkg
Pkg.activate(".")

println("nthreads = $(Threads.nthreads())")

using PiecewiseOrthogonalPolynomials, MatrixFactorizations, DelimitedFiles, LinearAlgebra
# using MKL
using AppleAccelerate
using Base: oneto
using MatrixFactorizations: reversecholcopy

buildtm = Dict()
choltm = Dict()
solvtm = Dict()
ns = 2 .^ (1:4:15)

pps = n -> 2 .^ (1:round(Int, log2(300_000_000 / n)))

# compile
let n = ns[1], p = 4
    r = range(-1,1; length=n+1)
    Q = DirichletPolynomial(r)
    Δ = weaklaplacian(Q)
    M = grammatrix(Q)
    KR = Block.(oneto(p))
    @elapsed((Δₙ = -parent(Δ)[KR,KR]; Mₙ = parent(M)[KR,KR];  L = reversecholcopy(Symmetric(Δₙ + Mₙ))))
    @elapsed((F = reversecholesky!(L)))
    x = randn(size(L,1))
    @elapsed(ldiv!(F, x))
end

for n in ns
    ps = pps(n)
    r = range(-1,1; length=n+1)
    Q = DirichletPolynomial(r)
    Δ = weaklaplacian(Q)
    M = grammatrix(Q)

    for p = ps
        @show n,p
        KR = Block.(oneto(p))
        GC.gc()
        GC.enable(false)
        buildtm[(n,p)] = @elapsed((Δₙ = -parent(Δ)[KR,KR]; Mₙ = parent(M)[KR,KR];  L = reversecholcopy(Symmetric(Δₙ + Mₙ))))
        GC.enable(true)
        GC.gc()
        GC.enable(false)
        choltm[(n,p)] = @elapsed((F = reversecholesky!(L)))
        GC.enable(true)
        x = randn(size(L,1))
        GC.gc()
        GC.enable(false)
        solvtm[(n,p)] = @elapsed(ldiv!(F, x))
        GC.enable(true)
    end
end



for n in ns
    ps = pps(n)
    data = [ps getindex.(Ref(buildtm),tuple.(n,ps)) getindex.(Ref(choltm),tuple.(n,ps)) getindex.(Ref(solvtm),tuple.(n,ps))]
    writedlm("data/1dtims_n_$n.csv", data)
end
