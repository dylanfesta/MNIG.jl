using Revise

using MNIG
using LinearAlgebra
using Distributions
const M = MNIG

##

n = 1
gamma = let g = M.make_rand_cov_mat(n,5)
  scal = det(g)^inv(n)
  g ./ scal |> Symmetric
end
mu=fill(10.0,n)
beta = [0.01] # rand(n)
delta = 1.4
alph = 0.1
distr = M.MNIG_distr(gamma,mu,delta,beta,alph)
##
using Plots
test1 = rand(distr,10_000)
histogram(test1[:],bins=100,opacity=0.3)
test2 = vcat( [rand(distr) for _ in 1:10_000]...)
histogram!(test2,bins=100,opacity=0.3)
test3 = M.rand_detailed(distr,10_000).X[:]
histogram!(test3,bins=100,opacity=0.3)


##
# let's test the pdf (at 1 dimension!)

test1 = rand(distr,10_000)[:]
x_lim = (-20,20.)
x_vals  = LinRange(x_lim...,1000) |> collect  |>  permutedims
y_vals  = pdf(distr,x_vals)

using Plots
histogram(test1,nbins=80, normed=true)
plot!(x_vals[:],y_vals,linewidth=3)


using QuadGK

_ = let
  fun(x) = pdf(distr,[x])
  quadgk(fun,-Inf,Inf)
end
