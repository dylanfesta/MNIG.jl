
#=
Here I use Stan to find the posteriors etc
=#

# N are the number of filters*2 , i.e. the filter output dimension
# M are the datapoints (image patches)
# warning: this code will get progressively worse as noise becomes smaller!

const stan_mnig_z_posterior = """
  functions {
    real inverse_gaussian_lpdf(vector x , real mu, real lambda){
    int n = num_elements(x);
    real out=0.0;
    for (i in 1:n)
      out += 0.5*log(lambda/(2.0*pi())) - 1.5*log(x[i]) - (0.5*lambda*(x[i]-mu)^2 / (mu*mu*x[i]));
    return out;
    }
  }
  data {
     int n;
     int m;
     matrix[n,m] X;
     matrix[n,n] Gamma;
     vector<lower=0>[n] beta;
     vector[n] mu;
     real<lower=0> mixer_mean;
     real<lower=0> mixer_shape;
  }
  transformed data {
    vector[n] Gammabeta;
    Gammabeta=Gamma*beta;
  }
  parameters {
     vector<lower=0>[m] z;
  }
  model {
    vector[n] mv_mean;
    matrix[n,n] mv_cov;
    target += inverse_gaussian_lpdf(z|mixer_mean,mixer_shape);
     for (j in 1:m){
      mv_mean  = mu + z[j]*Gammabeta;
      mv_cov = z[j]*Gamma;
      target += multi_normal_lpdf(X[:,j] | mv_mean,mv_cov );
     }
  }
"""
global stan_folder = ""

function set_stan_folder(folder)
    global stan_folder=folder
    set_cmdstan_home!(folder)
end

"""
        function sample_z_posterior_mnig()

# Inputs
  - `X` : each column represents a filter output vector
  - `n_samples` : samples taken from each chain (after thinning)
  - `thin_val` : thinning of the sampling
  - `pdir` : temporary director where Stan writes its files
  - `nchains` : number of parallel chains, the number of samples is
        n_sampl*n_chains
# Outputs
 The output are the z samples.
 """
function sample_z_posterior_mnig(mnig::MNIG_distr,X, n_samples ;
            thin_val=2,
            pdir=pwd(),
            nchains=4,
            nwarmup=1000 )
    @assert !isempty(stan_folder) "Please set the folder of cmdstan using set_stan_folder"
    println("the temporary directory will be created in parent folder: " * pdir)
    dim = size(X,1)
    num_data = size(X,2)
    n_samples=thin_val*n_samples
    Γ = convert(Matrix{Float64},mnig.Γ)
    β = mnig.β
    μ = mnig.μ
    mix_mean,mix_shape = mnig_mixer_pars(mnig)
    Data=[  Dict("n"=>dim,"m"=>num_data,
                    "X"=>X,"Gamma"=>Γ,
                    "beta"=>β, "mu"=>μ ,
                    "mixer_mean" => mix_mean,
                    "mixer_shape" => mix_shape) for _ in 1:nchains]
    stanmodel = Stanmodel(num_samples=n_samples,
            thin=thin_val, name="MNIG_z_posterior",
            model= stan_mnig_z_posterior, pdir=pdir,
            nchains=nchains,
            num_warmup=nwarmup)
    sim = stan(stanmodel, Data)
    get_data = get_data_all(sim[2],sim[3])
    get_data("z",num_data)
end

# general functions to read data of any dimesion from Stan
function get_stan_data_fun(data::Array{Float64},field_names::Vector{String})
  function out(str::String)
    idx=findfirst(s->s==str,field_names)
    idx ==0 && error("cannot read data, the name $str is not in the database")
    vec(data[:,idx,:])
  end
end

"""
Reads all elements or a matrix or a vector with indexes specified
by dims
"""
function get_data_all(data,datanames)
    get_data=get_stan_data_fun(data,datanames)
    function f_out(data_name::String,dims::Integer...)
        nd=length(dims)
        n_sampl=get_data(data_name * ".1"^nd) |> length
        out=Array{Float64}(undef,dims...,n_sampl)
        to_iter = Iterators.product([(1:d) for d in dims]...)
        for ijk in to_iter
            _str=data_name
            for i in ijk
                _str*=".$i"
            end
            out[ijk...,:] = get_data(_str)
        end
        out
    end
end
