#=
Define the basic types, the sampling

=#

struct MNIG_distr{T} <: ContinuousMultivariateDistribution
  Γ::AbstractMatrix{T}
  μ::AbstractVector{T}
  α::T
  β::AbstractVector{T}
  δ::T
  function MNIG_distr(Γ,μ,α,β,δ)
    alphsq = α*α
    bgb = dot(β,Γ*β)
    @assert isapprox(det(Γ),1.0;atol=1E-5 ) "determinant of Γ must be 1 !"
    @assert alphsq > bgb "wrong parameters! Please check β and α  !"
    new{eltype(μ)}(Γ,μ,α,β,δ)
  end
end

struct MNIG_sampler{T} <: Sampleable{Multivariate,Continuous}
  Γ::AbstractMatrix{T}
  μ::AbstractVector{T}
  α::T
  β::AbstractVector{T}
  δ::T
  function MNIG_sampler(Γ,μ,α,β,δ)
    alphsq = α*α
    bgb = transpose(β)*Γ*β
    @assert alphsq > bgb "wrong parameters! Please β and α  !"
    new{eltype(μ)}(Γ,μ,α,β,δ)
  end
end

function MNIG_sampler(mn::MNIG_distr)
  MNIG_sampler([ getfield(mn,i) for i in 1:fieldcount(MNIG_distr) ]... )
end

Distributions.sampler(mn::MNIG_distr) = MNIG_sampler(mn)


function  Base.length(mm::MNIG_distr)
  length(mm.β)
end
function  Base.length(mm::MNIG_sampler)
  length(mm.β)
end

# convert parameters of inverse gaussian
function invgauss_convert(χ::Real,ψ::Real)
  _shape = χ
  _mean = sqrt(χ/ψ)
  (_mean,_shape)
end
function mnig_mixer_pars(mn)
  n=length(mn.β)
  χ = mn.δ * mn.δ
  ψ = mn.α*mn.α -  dot(mn.β,  mn.Γ * mn.β)
  invgauss_convert(χ,ψ)
end

function Distributions._rand!(mn::MNIG_sampler, x::AbstractVector{T}) where T<: Real
  n=length(mn.β)
  χ = mn.δ * mn.δ
  gammabeta = mn.Γ * mn.β
  ψ = mn.α*mn.α -  dot(mn.β,gammabeta)
  z = rand(InverseGaussian(invgauss_convert(χ,ψ)...))
  rand!(MvNormal(fill(0.0,n), mn.Γ),x)
  @. x = sqrt(z)*x + z*gammabeta + mn.μ
end
function Distributions._rand!(mn::MNIG_distr, x::AbstractVector{T}) where T<: Real
  Distributions._rand!(MNIG_sampler(mn),x)
end


function Distributions._rand!(mn::MNIG_sampler, x::DenseMatrix{T}) where T<: Real
  n=length(mn.β)
  n_sampl=size(x,2)
  χ = mn.δ * mn.δ
  gammabeta = mn.Γ * mn.β
  ψ = mn.α*mn.α -  dot(mn.β,gammabeta)
  z = rand(InverseGaussian(invgauss_convert(χ,ψ)...),n_sampl)
  zetagb = broadcast(*,transpose(z),gammabeta)
  rand!(MvNormal(fill(0.0,n),mn.Γ),x)
  broadcast!((z,x)-> sqrt(z)*x, x,transpose(z),x)
  @. x = x + zetagb + mn.μ
end
function Distributions._rand!(mn::MNIG_distr, x::DenseMatrix{T}) where T<: Real
  Distributions._rand!(MNIG_sampler(mn),x)
end

function _f2_aux_mm(xmm,mn)
  sqrt(mn.δ*mn.δ+dot(xmm,mn.Γ\xmm))
end
function _f2_aux(x,mn)
  xmm = x - mn.μ
  _f2_aux_mm(xmm,mn)
end

function Distributions._logpdf(mn::MNIG_distr,x::AbstractVector)
  n=length(mn.μ)
  dp = 0.5(n+1)
  dm = 0.5(n-1)
  xmm = x - mn.μ
  f2 = _f2_aux_mm(xmm,mn)
  f1 = mn.δ * sqrt(mn.α*mn.α - transpose(mn.β)*mn.Γ*mn.β) + transpose(mn.β)*xmm
  logbess =log(besselk(dp,mn.α*f2))
  log(mn.δ/2^dm) + dp*log(mn.α/(pi*f2)) + f1 + logbess
end

# posterior of z given x
# function mnig_z_posterior(mn::MNIG_distr,x::AbstractVector)
#   χ = _f2_aux(x,mn)
#   ψ = mn.α
#   InverseGaussian( invgauss_convert(χ,ψ)... )
# end
#
function gig_pdf_ugly(x,p,a,b)
      (a/b)^(0.5p) *
      0.5(x^(p-1.0)/besselk(p,sqrt(a*b))) *
       exp(-0.5*(a*x+b/x))
end

function mnig_z_posterior(z::Real,mn::MNIG_distr,x::AbstractVector)
  n=length(mn.β)
  qx = _f2_aux(x,mn)
  gig_pdf_ugly(z,-0.5(1+n),mn.α*mn.α,qx*qx)
end


# sample, but return all the intermediate variables!
function rand_detailed(mn::MNIG_distr, n_sampl::Integer)
  n=length(mn.β)
  y = Matrix{Float64}(undef,n,n_sampl)
  χ = mn.δ * mn.δ
  gammabeta = mn.Γ * mn.β
  ψ = mn.α*mn.α -  dot(mn.β,gammabeta)
  z = rand(InverseGaussian(invgauss_convert(χ,ψ)...),n_sampl)
  zetagb = broadcast(*,transpose(z),gammabeta)
  rand!(Normal(),y)
  # Γch = cholesky(mn.Γ).U
  # x = Γch * y
  Γsqrt = sqrt(mn.Γ)
  x = Γsqrt * y
  broadcast!((z,x)-> sqrt(z)*x, x,transpose(z),x)
  @. x = x + zetagb + mn.μ
  (X=x,Y=y,Z=z)
end


# each column of data is an independent sample
function fit_ml_symmetric_mnig(data::AbstractMatrix{T}) where T<:Float64
  n,n_data= size(data)
  μ  = mean(data;dims=2) |> vec
  C = cov(data;dims=2)
  Γhat = C ./ (det(C)^inv(n)) |> Symmetric
  data_mmu = broadcast(-,data,μ)
  zwhite = sqrt(Γhat)\data_mmu
  # let's just fit all zs together!
   _d = fit_mle_less(NormalInverseGaussian,zwhite[:])
   _,αhat,_,δhat = params(_d)
  βhat = fill(0.0,n)
  MNIG_distr( Γhat, μ , αhat , βhat , δhat )
end
