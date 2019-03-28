using LinearAlgebra
using Random

##### main scripts #####
include("solvers.jl")
using Distributions
using Random
using LinearAlgebra
using PyCall
plt = pyimport("matplotlib.pyplot")

##### control panel #####
Random.seed!(12345678)
n = 500
p = 100 
σ = 0.75 
λ = 0.1/n
γ = 1.0/n

##### data generation & true parameter with 3 blocks #####
X = randn(n, p)
θᵀ = zeros(p)
θᵀ[16:20] = randn() * ones(5)
θᵀ[51:60] = randn() * ones(10)
θᵀ[90:96] = randn() * ones(7)
y = X*θᵀ + σ*randn(n)

#### learning parameter & edge-vertex matrix #####
θ = randn(p)
C = zeros(p-1, p)

for j = 2:p
    C[j-1, j-1] = γ
    C[j-1, j] = -γ
end

solve_fused_lasso(X, y, θ, λ, γ, C, θᵀ)

x = collect(1:1:p)
plt.rc("text", usetex=true)
plt.rc("font", family="Times New Roman", size=12)
plt.figure()
plt.xlabel("Predictor")
plt.ylabel("Coefficient")
plt.plot(x, θᵀ, label="True")
plt.plot(x, θ, label="Estimation")
plt.legend(loc="best")
plt.savefig("simulation.png", bbox_inches="tight", transparent=true, dpi=600)
