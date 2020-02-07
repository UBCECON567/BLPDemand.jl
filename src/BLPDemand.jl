module BLPDemand

using DataFrames: DataFrame
import CSV
import ForwardDiff
import NLsolve
using LinearAlgebra: dot, norm, Diagonal, inv, I, pinv
using Optim
using JuMP, Ipopt
using Statistics: cov
#import StatsBase
#import StatsModels

export 
  data_blp1999,
  share,
  sharep, dsharedp,
  delta,
  demandmoments,
  supplymoments,
  eqprices,
  simulateIVRClogit,
  simulateBLP,
  pack,
  estimateRCIVlogit,
  makeivblp,
  estimateBLP,
  varianceBLP,
  optimalIV,
  MarketData,
  BLPData


include("data.jl")
include("share.jl")
include("estimation.jl")
include("simulate.jl")

end # module
