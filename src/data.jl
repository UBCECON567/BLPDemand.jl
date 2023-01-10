"""
Data from a single market for a BLP demand model
"""
struct MarketData{VType <: AbstractVector, MType <: AbstractMatrix, IDType <: AbstractVector}
  """
  `J` vector of shares
  """
  s::VType
  """
  `K × J` matrix of good characteristics. If using `estimateBLP`, `x[1,:]`, should be prices
  """
  x::MType
  """
  `L × J` matrix of cost shifters
  """
  w::MType
  """
  `J` vector of firm identifies
  """
  firmid::IDType
  """
  `M × J` matrix of demand instruments
  """
  zd::MType
  """
  `M × J` matrix of supply instruments
  """
  zs::MType
  """
  `K × S` matrix of draws of ν for Monte-Carlo integration
  """
  ν::MType
end

"""
Data set for BLP demand model.
"""
const BLPData = Array{MarketData,1}

"""
Constructs data for BLP demand model from arrays. Compared to [`MarketData`](@ref), each argument should have one more dimension with length `T`=number of markets
"""
function blpdata(s::AbstractMatrix, 
                 x::AbstractArray{T, 3} where T,
                 ν::AbstractArray{T,3} where T,
                 ivdemand::AbstractArray{T,3} where T,
                 w::AbstractArray{T, 3} where T,
                 ivsupply::AbstractArray{T, 3} where T;
                 firmid=1:size(s,1) )
  J,T = size(s)
  dat = Array{MarketData,1}(undef, T)
  for t in 1:T
    dat[t] = MarketData(s[:,t],x[:,:,t],w[:,:,t],firmid,ivdemand[:,:,t], ivsupply[:,:,t], ν[:,:,t])
  end
  return(dat)
end

"""
Constructs data for IV random coefficients logit model from arrays. Compared to [`MarketData`](@ref), each argument should have one more dimension with length `T`=number of markets
"""
function blpdata(s::AbstractMatrix, 
                 x::AbstractArray{T, 3} where T,
                 ν::AbstractArray{T,3} where T,
                 ivdemand::AbstractArray{T,3} where T;
                 firmid=1:size(s,1) )
  J,T = size(s)
  dat = Array{MarketData,1}(undef, T)
  for t in 1:T
    dat[t] = MarketData(s[:,t],x[:,:,t],zeros(0,0),firmid,ivdemand[:,:,t],zeros(0,0), ν[:,:,t])
  end
  return(dat)
end


"""
    function blpdata(df::AbstractDataFrame,
                 mid::Symbol,
                 firmid::Symbol,
                 s::Symbol,                 
                 x::Vector{Symbol},
                 w::Vector{Symbol},
                 zd::Vector{Symbol},
                 zs::Vector{Symbol},
                 ν::AbstractArray{T,3} where T)

Construct BLPData from a DataFrame.

# Arguments
- `df::AbstractDataFrame`
- `mid::Symbol` market identifier in `df`
- `s::Symbol` market shares 
- `x::Vector{Symbol}` columns of `df` of product characteristics. `x[1]` must be price
- `w::Vector{Symbol}` cost shifters
- `zd::Vector{Symbol}` demand instruments
- `zs::Vector{Symbol}` supply instruments
- `ν::Array{T, 3} where T` `K × S × T` array of draws for Monte Carlo integration

See also [`MarketData`](@ref)
"""
function blpdata(df::DataFrame,
                 mid::Symbol,
                 firmid::Symbol,
                 s::Symbol,                 
                 x::Vector{Symbol},
                 w::Vector{Symbol},
                 zd::Vector{Symbol},
                 zs::Vector{Symbol},
                 ν::AbstractArray{T,3} where T)

  tvals = unique(df[!,mid])
  T = length(tvals)
  dat = Array{MarketData,1}(undef, T)
  for t in 1:T
    i = findall(df[!,mid] .== tvals[t])
    dat[t] = MarketData(df[i,s],convert(Matrix,df[i,x])',convert(Matrix,df[i,w])',
                        df[i,firmid], convert(Matrix,df[i, zd])', convert(Matrix,df[i, zs])',
                        ν[:,:,t])
  end
  return(dat)
  
end


"""
    function loadblpdata()
  
Loads data from Berry, Levinsohn, and Pakes (1999).

Returns a DataFrame.
"""
function data_blp1999()
  csvfile=normpath(joinpath(dirname(Base.pathof(BLPDemand)),"..","data","blp_1999_data.csv"))
  dt = CSV.read(csvfile)
  return(dt)
end

