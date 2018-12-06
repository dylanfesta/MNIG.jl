#=
Univariate Normal inverse gaussian
=#

function _normalIG_fit_approx(x::AbstractVector{T}) where T<: Real
  s = std(x)
  γ1 = skewness(x)
  γ2 = kurtosis(x)
  γhat = if 3.0γ2 > 5.0γ1^2
     3. / (s*sqrt(3.0γ2 - 5.0 * γ1^2))
   else
     3. / (s*sqrt( abs(3.0γ2)) ) # not sure what is best here
  end
  βhat = γ1 * s *γhat^2 / 3.0
  δhat =  ( s^2 * γhat^3 )/( βhat^2 + γhat^2 )
  μhat = mean(x) - βhat * δhat / γhat
  αhat = sqrt( γhat^2 + βhat^2 )
  (μhat, αhat,βhat,δhat,γhat)
end

function fit_approx(::Type{NormalInverseGaussian},x::AbstractVector{T}) where T<: Real
  (μ, α,β,δ,γ) = _normalinvgauss_fit_approx(x)
  NormalInverseGaussian(μ, α,β,δ)
end


function Distributions.rand(d::NormalInverseGaussian)
  μ,α,β,δ = params(d)
  _inversegaussian_reparam(γ,δ) = δ/γ, δ^2
  γ = sqrt(α^2 - β^2)
  z = rand(InverseGaussian(_inversegaussian_reparam(γ,δ)...))
  μ + β*z + sqrt(z)*randn()
end

# follwing equations and notations in D. Karlis (2002)
function _NIG_EM_step(x::AbstractVector{T},
            μ::T, α::T , β::T , δ::T , γ::T ) where T <: Real

  n = length(x)
  xbar = mean(x)
  # ϕ , s and w
  dphi = @. δ*sqrt(1. + ((x-μ)/δ)^2) # equivalent to δ*ϕ^1/2 in ref paper
  _b = dphi .* α
  b1 = besselk.(1,_b)
  s = @. dphi*besselk(0,_b)/(b1*α)
  w = @. α*besselk(-2,_b)/(b1*dphi)
  # avoid NaN elements if both bessels are 0.0
  s[b1 .== 0.0 ] .= 1.0
  w[b1 .== 0.0 ] .= 1.0
  # M and Λ
  Mhat = mean(s)
  Mhinv = inv(Mhat)
  Λhat = inv(mean( w .- Mhinv ))
  # upgrade the params
  sbar = Mhat  # not sure why the do not use Mhat only in the paper
  δp = sqrt(Λhat)
  γp = δp / Mhat
  sumw=sum(w)
  βp = (dot(x,w) - xbar*sumw)/(n-sbar*sumw)
  μp = xbar - βp*sbar
  αp = sqrt(  γp^2 + βp^2 )
  (μp, αp,βp,δp,γp)
end

function Distributions.fit_mle(::Type{NormalInverseGaussian},
                x::AbstractVector{T}) where T<: Real
  # initial parameters
  p_init = _normalIG_fit_approx(x)
  loglik(p) = loglikelihood( NormalInverseGaussian(p[1],p[2],p[3],p[4]), x )

  max_iter = 500
  atol = 1E-10
  p_old = p_init
  p_new = p_old
  iter = 0
  cost = 1E10
  ll_old = loglik(p_old)
  while cost > atol && iter < max_iter
    p_new = _NIG_EM_step(x,p_old...)
    ll_new = loglik(p_new)
    cost = abs((ll_new-ll_old)/ll_old )
    iter +=1
    ll_old=ll_new
   end
   if iter==max_iter
     @warn "the EM fitting procedure for the normal inverse Gaussian did not converge"
   end
   NormalInverseGaussian(p_new[1:4]...)
end

# now the case of zero mean and symmetric (beta = 0 )

# follwing equations and notations in D. Karlis (2002)
function _NIG_EM_step_less(x::AbstractVector{T},
           α::T , δ::T , γ::T ) where T <: Real

  n = length(x)
  xbar = 0.0
  # ϕ , s and w
  dphi = @. δ*sqrt(1. + (x/δ)^2) # equivalent to δ*ϕ^1/2 in ref paper
  _b = dphi .* α
  b1 = besselk.(1,_b)
  s = @. dphi*besselk(0,_b)/(b1*α)
  w = @. α*besselk(-2,_b)/(b1*dphi)
  # avoid NaN elements if both bessels are 0.0
  s[b1 .== 0.0 ] .= 1.0
  w[b1 .== 0.0 ] .= 1.0
  # M and Λ
  Mhat = mean(s)
  Mhinv = inv(Mhat)
  Λhat = inv(mean( w .- Mhinv ))
  # upgrade the params
  sbar = Mhat  # not sure why the do not use Mhat only in the paper
  δp = sqrt(Λhat)
  γp = δp / Mhat
  sumw=sum(w)
  βp = dot(x,w)/(n-sbar*sumw)
  αp = sqrt(γp^2)
  (αp,δp,γp)
end

function fit_mle_less(::Type{NormalInverseGaussian},
    x::AbstractVector{T}) where T<: Real
    # initial parameters , mean 0 , kurtosis 0
    s = std(x)
    γ1 = 0.0
    γ2 = kurtosis(x)
    γhat = if 3.0γ2 > 5.0γ1^2
       3. / (s*sqrt(3.0γ2))
     else
       3. / (s*sqrt( abs(3.0γ2)) ) # not sure what is best here
    end
    βhat = 0.0
    δhat =  s^2 * γhat
    μhat = 0.0
    αhat = sqrt( γhat^2  )
    p_init = (αhat,δhat,γhat)
    loglik(p) = loglikelihood( NormalInverseGaussian(0.0,p[1],0.0,p[2]), x )

    max_iter = 500
    atol = 1E-10
    p_old = p_init
    p_new = p_old
    iter = 0
    cost = 1E10
    ll_old = loglik(p_old)
    while cost > atol && iter < max_iter
      p_new = _NIG_EM_step_less(x,p_old...)
      ll_new = try
        loglik(p_new)
      catch
        @show p_old p_new
        error("Nooo!")
      end
      cost = abs((ll_new-ll_old)/ll_old )
      iter +=1
      ll_old=ll_new
     end
     if iter==max_iter
       @warn "the EM fitting procedure for the normal inverse Gaussian did not converge"
     end
     NormalInverseGaussian(0.0,p_new[1],0.0,p_new[2])
end
