module TensorTrainApprox

using GaussQuadrature
using Maxvol
using LinearAlgebra
using PiecewiseLinearApprox

include("TTtoolbox.jl")

export chebyshev_nodes,
       legendre_nodes,
       piecewise_linear_nodes

export TensorTrain,
       ChebyshevTensorTrain,
       LegendreTensorTrain,
       PiecewiseTensorTrain,
       FunctionalTensorTrain

export constantTT,
       randomTT,
       TTsvd,
       DMRGcross,
       DMRGcross_generic,
       DMRGcross_threaded,
       decompress,
       decompress_threaded,
       TTevaluate,
       TTderivative,
       TTgradient,
       createCTT,
       createLTT,
       createPTT,
       createFTT,
       TTinterp,
       TTintegrate_GC,
       TTintegrate_GL,
       TTintegrate_GH

end