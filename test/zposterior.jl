# testing the posterior on z using Stanmodel

using Revise

using MNIG
using LinearAlgebra
using Distributions
const M = MNIG

##

M.set_stan_folder("/home/dfesta/.cmdstan-2.18.0")

mydir= @__DIR__()

standir = Base.Filesystem.mktempdir(mydir)

## create a test system
n = 20
gamma = let g = M.make_rand_cov_mat(n,5)
  scal = det(g)^inv(n)
  g ./ scal |> Symmetric
end
mu=fill(5.0,n)
beta = 0.0.*rand(n)
delta = 1.4
alph = 0.1

distr = M.MNIG_distr(gamma,mu,delta,beta,alph)
##
test_truth = M.rand_detailed(distr,50)

##

Xbis =let Γsqrt= sqrt(distr.Γ) ,
  gY =  Γsqrt*test_truth.Y
  out = broadcast((z,x)->sqrt(z)*x , transpose(test_truth.Z),gY)
  broadcast!(+,out,out,distr.μ)
 end

test_truth.X - Xbis

##

whatevs = M.sample_z_posterior_mnig(distr,test_truth.X, 1_000 ; pdir=standir)

##

Z_guess = median(whatevs ; dims=2)[:]

using Plots; plot()

scatter(test_truth.Z,Z_guess)
plot!(x->x, linestyle=:dash, # xlim=(0,10),ylim=(0,10),
    xlabel="true", ylabel="guess", leg=false,ratio=1)

histogram(test_truth.X[:],nbins=50)

##
# now z whole distribution ...
idx_test = 18
X_test = test_truth.X[:,idx_test]
histogram(whatevs[idx_test,:][:]; nbins=100,normed=true)
xplot= LinRange(1E-4,10,500)
yplot = M.mnig_z_posterior.(xplot,Ref(distr),Ref(X_test))
plot!(xplot,yplot,linewidth=6)



##

Base.Filesystem.rm(standir;recursive=true)




##
x_test = randn(10)
gammach = cholesky(gamma).U
gammasqr = sqrt(gamma)

gammach*x_test



gammasqr*x_test
