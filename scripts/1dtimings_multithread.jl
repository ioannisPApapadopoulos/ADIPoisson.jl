# run via
# julia -t 4 scripts/1dtimings_multithread.jl

using PiecewiseOrthogonalPolynomials, MatrixFactorizations, DelimitedFiles
# using MKL
using AppleAccelerate
using Base: oneto
using MatrixFactorizations: reversecholcopy

# buildtm = Dict()
choltm = Dict()
solvtm = Dict()
n = 8

pps = n -> 2 .^ (1:round(Int, log2(300_000_000 / n)))
ps = pps(n)

r = range(-1,1; length=n+1)
Q = DirichletPolynomial(r)
Δ = weaklaplacian(Q)
M = grammatrix(Q)

# compile
let p = ps[1]
    KR = Block.(oneto(p))
    (Δₙ = -parent(Δ)[KR,KR]; Mₙ = parent(M)[KR,KR];  L = reversecholcopy(Symmetric(Δₙ + Mₙ)))
    F = reversecholesky!(L)
    x = randn(size(L,1))
    ldiv!(F, x)
end

for p = ps
    @show n,p
    KR = Block.(oneto(p))
    (Δₙ = -parent(Δ)[KR,KR]; Mₙ = parent(M)[KR,KR];  L = reversecholcopy(Symmetric(Δₙ + Mₙ)))
    choltm[p] = @elapsed((F = reversecholesky!(L)))
    x = randn(size(L,1))
    solvtm[p] = @elapsed(ldiv!(F, x))
end


data = [ps getindex.(Ref(choltm),ps) getindex.(Ref(solvtm),ps)]
writedlm("data/1dtims_thread_$(Threads.nthreads()).csv", data)

