module BLPDemand

using DataFrames: DataFrame
import CSV
import ForwardDiff
import NLsolve
using LinearAlgebra: dot, norm, Diagonal, inv, I, pinv
using Optim
import LineSearches
using JuMP
import Ipopt
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
  varianceRCIVlogit,
  makeivblp,
  estimateBLP,
  varianceBLP,
  optimalIV,
  MarketData,
  BLPData,
  fracRCIVlogit


include("data.jl")
include("share.jl")
include("estimation.jl")
include("simulate.jl")

end # module
