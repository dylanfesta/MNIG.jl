#=
Univariate Normal inverse gaussian
=#



function Distributions.fit_mle(::Type{NormalInverseGaussian},data::AbstractVector{T}) where T<: Real
  ϕ(x,μ,δ) = δ*sqrt(1. + ((x-μ)/δ)^2) # equivalent to δ*ϕ^1/2 in ref paper
  function s_and_w(x,α,δ,μ) =
    dphi = ϕ(x,μ,δ)
    _b = dphi*α
    s = dphi*besselk(0,_b)/(besselk(1,_b)*α)
    w = α*besselk(-2,_b)/(besselk(-1,_b)*dphi)
    (s,w)
  end
  Mhat(s::AbstractVector) = mean(s)
  function Λhat(w::AbstractVector,_Mhat)
    Mhinv = inv(_Mhat)
    inv(mean( w .- Mhinv ))
  end

  function update_params( x, w , Mhat, Λ  )
    n = length(x)
    xbar = mean(x)
    sbar = Mhat  # not sure why the do not use Mhat only in the paper
    δp = sqrt(Λ)
    γp = δp / Mhat
    sumw=sum(w)
    βp = (dot(x,w) - xbar*sumw)/(n-sbar*sumw)
    μp = xbar - βp*sbar
    αp = sqrt(  γp^2 + βp^2 )
    (αp,βp,δp,γp)
  end

  function initial_params(x)
    xbar = mean(x)
    s = std(x)
    γ1 = skewness(x)
    γ2 = kurtosis(x)
    γhat = 3. / ( s*sqrt(3.0γ2 - 5.0*(γ1^2) ) )
    βhat = γ1 * s^2 *γhat^2 / 3.0
    δhat =  ( s^2 + γhat^3 )/( βhat^2 + γhat^2 )
    μhat = mean(x) - βhat * δhat / γhat
    αhat = sqrt( γhat^2 + βhat^2 )
    (αhat,βhat,δhat,γhat)
  end

end
