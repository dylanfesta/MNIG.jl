module MNIG
using Distributions, SpecialFunctions, Random, LinearAlgebra

# stan just for testing purposes... to be moved elsewhere
using CmdStan


"""
    make_rand_cov_mat( dims , diag_val::Float64 ; k_dims=5)

Returns a random covariance matrix that is positive definite
and has off-diagonal elements.
# Arguments
- `d`: dimensions
- `diag_val`: scaling of the diagonal
- `k-dims`: tuning of off diagonal elements
"""
function make_rand_cov_mat( dims::Integer , diag_val::Real , (k_dims::Integer)=5)
  W = randn(dims,k_dims)
  S = W*W'+ Diagonal(rand(dims))
  temp_diag = Diagonal(inv.(sqrt.(diag(S))))
  S = temp_diag * S * temp_diag
  S .*= diag_val
  Symmetric(S)
end


include("base_sample.jl")
include("stan_inference.jl")
include("normalinversegaussian.jl")


end # module
