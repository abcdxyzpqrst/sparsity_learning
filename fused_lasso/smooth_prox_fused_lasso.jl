using LinearAlgebra
using Random

# main scripts 
include("solvers.jl")
using Distributions
using Random
using LinearAlgebra
using PyCall
plt = pyimport("matplotlib.pyplot")

# control panel 
Random.seed!(1234)
n = 80          # number of observations
p = 100         # problem dimension
σ = 0.5         # noise level
λ = 1.0/n       # regularization parameter for ℓ₁
γ = 10.0/n      # regularization parameter for fusion penalty 

# data generation with 3 different blocks 
X = randn(n, p)
θᵀ = zeros(p)
θᵀ[16:20] = randn() * ones(5)
θᵀ[51:60] = randn() * ones(10)
θᵀ[90:96] = randn() * ones(7)
y = X*θᵀ + σ*randn(n)

# learning parameter & edge-vertex matrix 
θ = randn(p)
C = zeros(p-1, p)
for j = 2:p
    C[j-1, j-1] = γ
    C[j-1, j] = -γ
end

# solve a fused lasso with smooth proximal gradient descent 
solve_fused_lasso(X, y, θ, λ, γ, C, θᵀ)

# plotting results 
x = collect(1:1:p)
plt.rc("text", usetex=true)
plt.rc("font", family="Times New Roman", size=16)
plt.figure()
plt.title("Fused Lasso")
plt.xlabel("Predictor")
plt.ylabel("Coefficient")
plt.plot(x, θᵀ, "ro", ms=3.0, label="True")
plt.plot(x, θ, "bo", ms=3.0, label="Estimation")
plt.legend(loc="best")
plt.savefig("fused_lasso_simulation.pdf", bbox_inches="tight", transparent=false, dpi=600)
