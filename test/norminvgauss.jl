
using Revise

using MNIG ; const M = MNIG
using LinearAlgebra
using Distributions

##
μ = -55.0
α=8.0
β = 7.0
δ = 30.3
γ = sqrt( α^2 - β^2 )

dtest = NormalInverseGaussian(μ,α,β,δ)


n_sampl=1000
xtest = rand(dtest,n_sampl)
pinit = M._normalIG_fit_approx(xtest)

dfit = fit_mle(NormalInverseGaussian,xtest)
dfitless= M.fit_mle_less(NormalInverseGaussian,xtest)


n_sampl_test=10_000
xtest2 = rand(dfit,n_sampl_test)

using Plots ; plot()
histogram(xtest ; normed=true, nbins=30,opacity=0.2)
xplot = LinRange(xlims()...,200)
plot!(xplot,pdf.(dtest,xplot) , linewidth=2)
plot!(xplot,pdf.(dfit,xplot) , linewidth=2)
plot!(xplot,pdf.(dfitless,xplot) , linewidth=2)

##
loglikelihood(NormalInverseGaussian(newpar[1:4]...),xtest ) > loglikelihood(dtest,xtest)

##

M._normalinvgauss_fit_approx(xtest)
dfit = M.fit_approx(NormalInverseGaussian,xtest)




mean(xtest)
μ+δ*β/γ

mean(xtest2)

var(xtest)
δ*α*α/γ^3
var(dtest)
var(xtest2)


skewness(xtest)
3β/(α*sqrt(γ*δ))
skewness(xtest2)

kurtosis(xtest)
3/(γ*δ)*(1+4*β*β / α^2)
kurtosis(xtest2)
