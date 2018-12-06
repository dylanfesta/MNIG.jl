using Revise

using MNIG ;  const M = MNIG
using LinearAlgebra
using Distributions


##

## create a test system
n = 18
mygamma = let g = M.make_rand_cov_mat(n,5)
  scal = det(g)^inv(n)
  g ./ scal |> Symmetric
end
mu=fill(5.0,n)
mybeta = 0.0.*rand(n)
delta = 1.0
alph = 1.0

distr = M.MNIG_distr(mygamma,mu,alph,mybeta,delta)

##
data_test = rand(distr,500)

distr_fit = M.fit_ml_symmetric_mnig(data_test)


##
using Plots ; plot()
test1 = rand(distr,10_000)
test2 = rand(distr_fit,10_000)
histogram(test1[:]; nbins=100,normed=true,opacity=0.3)
histogram!(test2[:]; nbins=100,normed=true,opacity=0.3)


scatter(mygamma[:],distr_fit.Γ[:],ratio=1)
plot!(x->x ; linestyle=:dash,leg=false)
