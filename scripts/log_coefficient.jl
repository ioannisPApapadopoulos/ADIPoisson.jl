using MKL
using ADIPoisson, BlockArrays
using LinearMaps, IterativeSolvers
using Plots, LaTeXStrings

"""
Solve 
    (-Δ - 10 log√(x^2+y^2))u = 1 on Ω=(-1,1)^2 with zero b.c.

We discretize with a graded mesh towards (0,0) to capture the singularity
and use a matrix-free CG Krylov method preconditioned with the
Laplacian solved via ADI.

We converge in ~7 CG iterations independent of h and p.

"""

coeff(x,y) = -1e1*log(sqrt(x^2+y^2)) # NCC coefficient
f(x,y) = 1.0 # right-hand side


function solve_variable_coefficient(p::Int, m::Int)
    r = sort([-reverse(10.0.^(0:-1:-m)); 0.0; reverse(10.0.^(0:-1:-m))]) # graded mesh towards (0,0)

    P = ADILaplacePreconditioner(coeff, r, p, adi_tolerance=1e-4); # plan the ADI preconditioner

    Ff(x) = apply_ncc(P, x)
    F = LinearMap(Ff, (P.n*(P.p-1) + (P.n-1))^2; ismutating=false) # Matrix-free residual


    (x,y) = P.xy
    b = vec( (P.QP * (P.pl_P * f.(x,reshape(y,1,1,size(y)...)))) * P.QP') # assemble load vector
    u, info = IterativeSolvers.cg(F, b, Pl=P, abstol=1e-15, reltol=1e-8, log=true) # Solve via CG preconditioned with ADI
    return u, info
end

ms = [1,2,3]
ps = 2 .^(3:7)
iter_table = zeros(Int, length(ms), length(ps))

for (m, i) in zip(ms, axes(ms,1)), (p, j) in zip(ps, axes(ps,1))
    print("Considering m=$m, p=$p.\n")
    u, info = solve_variable_coefficient(p, m)
    iter_table[i, j] = info.iters
end


## Plotting
m, p = 3, 2^7
r = sort([-reverse(10.0.^(0:-1:-m)); 0.0; reverse(10.0.^(0:-1:-m))])
P = ADILaplacePreconditioner(coeff, r, p, adi_tolerance=1e-4);
u, info = solve_variable_coefficient(p, m)
n = P.n
(x,y) = P.xy
Uc =  BlockMatrix(reshape(u, p*n-1, p*n-1), [n-1;repeat([n], p-1)], [n-1;repeat([n], p-1)]);
Ux = P.pl_Q \ Uc;
Ux = reshape(Ux[end:-1:1,:,end:-1:1,:], length(x), length(y))'
Ux = hcat(zeros(size(Ux,1)+2), vcat(zeros(1,size(Ux,2)), Ux, zeros(1,size(Ux,2))), zeros(size(Ux,1)+2))

Plots.gr_cbar_offsets[] = (-0.05,-0.01)
Plots.gr_cbar_width[] = 0.03
surface([-1;vec(x[end:-1:1,:]);1], [-1;vec(y[end:-1:1,:]);1], 
    Ux, 
    xlabel=L"x", ylabel=L"y",zlabel=L"u(x,y)",
    xlabelfontsize=15,ylabelfontsize=15,zlabelfontsize=15,
    color=:thermal,
    cbarfontsize=15)
Plots.savefig("log-u.pdf")

surface(vec(x[end:-1:1,:]), vec(y[end:-1:1,:]), 
    reshape(P.coeffx[end:-1:1,:,end:-1:1,:], length(x), length(y))', 
    xlabel=L"x", ylabel=L"y",zlabel=L"-10 \cdot \mathrm{log}\sqrt{x^2+y^2}",
    xlabelfontsize=15,ylabelfontsize=15,zlabelfontsize=10,
    color=:thermal,
    cbarfontsize=15,
    xlim=[-1,1], ylim=[-1,1])
Plots.savefig("log-c.pdf")

m=3; r = sort([-reverse(10.0.^(0:-1:-m)); 0.0; reverse(10.0.^(0:-1:-m))])
Plots.plot(xlim=[-1-1e-3,1+1e-3],ylim=[-1-1e-3,1+1e-3],legend=:none,grid=false, xlabel=L"x", ylabel=L"y",
    xlabelfontsize=15,ylabelfontsize=15, aspect_ratio=:equal)
vline!(r,color=:black, linewidth=1)
hline!(r,color=:black, linewidth=1)
Plots.savefig("graded-mesh.pdf")