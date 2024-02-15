module TensorTrainApprox

using ChebyshevApprox
using GaussQuadrature
using Maxvol
using LinearAlgebra
using PiecewiseLinearApprox

include("TTtoolbox.jl")

export chebyshev_nodes,
       piecewise_linear_nodes

export TensorTrain,
       ChebyshevTensorTrain,
       FunctionalTensorTrain

export constantTT,
       randomTT,
       TTsvd,
       DMRGcross,
       DMRGcross_threaded,
       decompress,
       decompress_threaded,
       TTevaluate,
       TTderivative,
       TTgradient,
       createCTT,
       createFTT,
       TTinterp,
       TTintegrate_GC,
       TTintegrate_GL,
       TTintegrate_GH

end