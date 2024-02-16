# ADIPoisson.jl

This repository implements the numerical examples found in:

(1) "Quasi-optimal complexity hp-FEM for Poisson on a rectangle", Kars Knook, Sheehan Olver, and Ioannis P. A. Papadopoulos (2024).

In one dimension the hierarchical basis of Szabó and Babuška can be  solved in optimal complexity because the discretisation has a special sparsity structure that ensures that the reverse Cholesky factorisation---Cholesky starting from the bottom right instead of the top left---remains sparse. By incorporating this approach into an Alternating Direction Implicit (ADI) method à la Fortunato and Townsend (2020) one can solve, within a prescribed tolerance, an $hp$-FEM discretisation of the (screened) Poisson equation on a rectangle, in parallel, with quasi-optimal complexity: $O(N^2 \log^2 N)$ operations where N is the maximal total degrees of freedom in each dimension. When combined with fast Legendre transforms we can also solve nonlinear time-evolution partial differential equations in a quasi-optimal complexity of $O(N^2 \log^2 N)$ operations, which is demonstrated on the (viscid) Burgers' equation.

This package heavily utilises [PiecewiseOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/PiecewiseOrthogonalPolynomials.jl) for its implementation of a sparse p-FEM basis on the interval.

|Figure|File: examples/|
|:-:|:-:|
|1|[1dtimings.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/1dtimings.jl)|
|2|[1dtimings_multithread.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/1dtimings_multithread.jl)|
|3|[pixels.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/pixels.jl)|
|4|[adi_plots.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/adi_plots.jl)|
|5a|[pixels.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/pixels.jl)|
|5b|[adi_plots.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/adi_plots.jl)|
|6a|[timeevolution_plots.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/timeevolution_plots.jl)|
|6b|[timeevolution.jl](https://github.com/ioannisPApapadopoulos/QuasiOptimalhpFEM.jl/blob/main/scripts/timeevolution.jl)|


## Contact
Ioannis Papadopoulos: papadopoulos@wias-berlin.de