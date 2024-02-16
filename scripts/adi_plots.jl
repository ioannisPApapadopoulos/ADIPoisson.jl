using ADIPoisson, DelimitedFiles, PiecewiseOrthogonalPolynomials, Plots, LaTeXStrings
using AppleAccelerate # Use on Macs to get fastest solves



ns = 2 .^ (1:4)

# adi_p_build = plot(;title="ADI build+factorise", legend=:bottomright, xlabel="p", ylabel="seconds", xticks=10 .^ (1:8), yticks = 10.0 .^ (-(0:5)), ylims=[1E-4,10])
# for n in ns
#     @show n
#     ps = pps(n)
#     plot!(ps, getindex.(Ref(adi_buildtm), tuple.(n,ps)); xscale=:log10, yscale=:log10, label="n = $n", linewidth=2)
# end; adi_p_build
# savefig("adi_buildtms.pdf")


# adi_p_solve = plot(;title="ADI solve", legend=:bottomright, xlabel="p", ylabel="seconds", xticks=10 .^ (1:8), yticks = 10.0 .^ (-(-2:5)), ylims=[1E-4,10])
# for n in ns
#     @show n
#     ps = pps(n)
#     plot!(ps, getindex.(Ref(adi_solvtm), tuple.(n,ps)); xscale=:log10, yscale=:log10, label="n = $n", linewidth=2)
# end; adi_p_solve
# savefig("adi_solvetms.pdf")

pps = n -> 2 .^ (2:round(Int, log2(20_000 / n)))

adi_p_build_dof = plot(;title="ADI build+factorise", legend=:bottomright, xscale=:log10, yscale=:log10, xlabel=L"N^2" * " (Total Degrees of Freedom)", ylabel="time (seconds)", xticks=10 .^ (1:8), yticks = 10.0 .^ (-5:4), ylims=[1E-4,1E4])
markers = [:circle, :rect, :diamond, :uptriangle]
m = 0
for n in ns
    m += 1
    @show n
    r = range(-1, 1; length=n+1)
    Q = DirichletPolynomial(r)
    dat = readdlm("data/2dtims_n_$n.csv")
    scatter!([size(Q[:,Block.(Base.oneto(p))],2)^2 for p in pps(n)], dat[:,2]; label="n = $n", markershape=markers[m])
end
Ns = 2*pps(2) .- 1; plot!(Ns .^ 2, 5E-10*Ns.^2 .* log.(Ns).^2; label=false, style=:dash)
annotate!(1E7, 15, L"O(N^2 log^2 N)")
savefig("adi_buildtms_dof.pdf")


adi_p_solve_dof = plot(;title="ADI solve", legend=:bottomright, xscale=:log10, yscale=:log10, xlabel=L"N^2" * " (Total Degrees of Freedom)", ylabel="time (seconds)", xticks=10 .^ (1:8), yticks = 10.0 .^ (-5:4), ylims=[1E-4,1E4])
m = 0
for n in ns
    m += 1
    @show n
    r = range(-1, 1; length=n+1)
    Q = DirichletPolynomial(r)
    dat = readdlm("data/2dtims_n_$n.csv")
    scatter!([size(Q[:,Block.(Base.oneto(p))],2)^2 for p in pps(n)], dat[:,3]; label="n = $n", markershape=markers[m])
end
Ns = 2*pps(2) .- 1; plot!(Ns.^2, 4E-7*Ns.^2 .* log.(Ns); label=false, style=:dash)
annotate!(1E7, 8E-1, L"O(N^2 log N)")
savefig("adi_solvetms_dof.pdf")

###
# Neumann
###

pps = n -> 2 .^ (2:round(Int, log2(20_000 / n)))
n = 2

markers = [:circle, :rect, :diamond, :uptriangle]
adi_p_solve_dof = plot(;title="ADI Neumann solve", legend=:bottomright, xscale=:log10, yscale=:log10, xlabel=L"N^2" * " (Total Degrees of Freedom)", ylabel="time (seconds)", xticks=10 .^ (1:8), yticks = 10.0 .^ (-5:4), ylims=[1E-4,1E4])
m = 0
ωs = 10 .^ (0:3) # 2 .^ (1:4)
for ω in ωs
    m += 1
    @show ω
    r = range(-1, 1; length=n+1)
    Q = ContinuousPolynomial{1}(r)
    dat = readdlm("data/2dtims_neumann_omega_$ω.csv")
    scatter!([size(Q[:,Block.(Base.oneto(p))],2)^2 for p in pps(n)], dat[:,3]; label="ω = $ω", markershape=markers[m])
end
Ns = 2*pps(2) .+ 1; plot!(Ns.^2, 4E-7*Ns.^2 .* log.(Ns); label=false, style=:dash)
annotate!(1E7, 8E-1, L"O(N^2 log N)")
savefig("adi_neumann_solvetms_dof.pdf")



####
# testing
###
p = 250; n = 4;
@time (x,y),pl = plan_poissonsolve!(n, p);
f = (x,y) -> -2 .*sin.(π*x) .* (2π*y .*cos.(π*y) .+ (1-π^2*y^2) .*sin.(π*y))
@time Y = pl * f.(x, y);

r = range(-1, 1, n+1)
Q = DirichletPolynomial(r)

Ua = Q[0.1, Block.(1:p)]' * Y  * Q[0.2, Block.(1:p)]
u_exact = (x,y) -> sin.(π*x)*sin.(π*y)*y^2

@test u_exact.(0.1,0.2) ≈ Ua # ℓ^∞ error



####
# OLD
####


@time vals = f.(x, y);
@time pl * vals;







fp = Pl * f.(x, reshape(y,1,1,p,:))



@time X = adi(F, A, B, Δₙ, a, b, c, d; factorize = rc, tolerance=0.0001);
z = SVector.(-1:0.01:1, (-1:0.01:1)')
Fa = P[first.(z)[:,1], KR] * fp  * P[first.(z)[:,1], KR]'
@test Fa ≈ splat(f).(z)



F = (Q'*P)[KR, KR]*fp*((Q'*P)[KR, KR])'  # RHS <f,v>


rc = reversecholesky ∘ Symmetric

A, B, a, b, c, d = Mₙ, -Mₙ, e1s, e2s, -e2s, -e1s
@time X = adi(F, A, B, Δₙ, a, b, c, d; factorize = rc, tolerance=0.0001);

@time adip = plan_adi!(A, B, Δₙ, a, b, c, d; factorize = rc, tolerance=0.001);

@time X = adip*F;

# X = UΔ
Y = (U' \ (U \ X'))'

u_exact = z -> ((x,y)= z; sin.(π*x)*sin.(π*y)*y^2)
Ua = Q[first.(z)[:,1], Block.(1:p)] * Y  * Q[first.(z)[:,1], Block.(1:p)]'
@test u_exact.(z) ≈ Ua # ℓ^∞ error



#### timing


ns = 2 .^ (1:4)
pps = n -> 2 .^ (2:round(Int, log2(1_000 / n)))
aditm = Dict()

for n in ns
    r = range(-1, 1; length=n+1)

    P = ContinuousPolynomial{0}(r)
    Q = DirichletPolynomial(r)
    Δ = -weaklaplacian(Q)
    M = grammatrix(Q)

    for p in pps(n) # truncation degree on each cell
        @show (n,p)
        KR = Block.(oneto(p))
        Δₙ = Δ[KR,KR]
        Mₙ = M[KR,KR]

        U = reversecholesky(Symmetric(Δₙ)).U
        A = (U \ (U \ Mₙ)') # = L⁻¹ pΔ L⁻ᵀ
        @time e1s, e2s = eigmin(A), eigmax(A);


        A, B, a, b, c, d = Mₙ, -Mₙ, e1s, e2s, -e2s, -e1s
        F = randn(size(A))
        aditm[(n,p)] = @elapsed(X = ADI(A, B, Δₙ, F, a, b, c, d; factorize = rc));
    end
end

p = plot(;legend=:bottomright, xlabel="p", ylabel="seconds")
for n in ns
    plot!(pps(n), [aditm[(n,p)] for p in pps(n)]; xscale=:log10, yscale=:log10, xticks=10 .^ (0:0.5:3), yticks=10.0 .^ (-3:1), label="h = $(2/n)")
end
p
savefig("figures/adi.pdf")

p = plot(;legend=:bottomright, xlabel="DOF", ylabel="seconds")
for n in ns
    r = range(-1, 1; length=n+1)
    P = ContinuousPolynomial{0}(r)
    plot!([size(P[:,Block.(oneto(p))],2)^2 for p in pps(n)], [aditm[(n,p)] for p in pps(n)]; xscale=:log10, yscale=:log10, xticks=10 .^ (1:10), yticks=10.0 .^ (-3:1), label="h = $(2/n)", linewidth=2)
end
p; 
savefig("figures/adi_DOF.pdf")