##### main scripts #####
include("solvers.jl")
using Distributions
using Random
using LinearAlgebra
using PyCall
plt = pyimport("matplotlib.pyplot")

##### control panel #####
n = 20
p = 100 
σ = 0.75 
s₁ = 35.6 
s₂ = Inf

X = randn(n, p)
# three consecutive blocks in the true parameter
