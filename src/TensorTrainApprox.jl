module TensorTrainApprox

using GaussQuadrature
using Maxvol
using LinearAlgebra
using GenericLinearAlgebra

include("TTtoolbox.jl")

export chebyshev_nodes,
       legendre_nodes,
       piecewise_linear_nodes

export BaseTensorTrain,
       ExtendedTensorTrain,
       ChebyshevTensorTrain,
       LegendreTensorTrain,
       PiecewiseTensorTrain,
       FunctionalTensorTrain

export TTconstant,
       TTrandom,
       TTChebyshev,
       TTLegendre,
       TTsvd,
       DMRGcross,
       DMRGcross_generic,
       DMRGcross_threaded,
       DMRGcross_generic_threaded,
       createCTT,
       createLTT,
       createPTT,
       createFTT,
       TTinterp,
       decompress,
       decompress_threaded,
       TTevaluate,
       TTderivative,
       TTgradient,
       TTintegrate_GC,
       TTintegrate_GL,
       TTintegrate_GH,
       TTintegrate_PL,
       compute_marginal_GH,
       compute_marginal_GC,
       compute_marginal_GL,
       compute_marginal_PL,
       TTrounding,
       TTmult,
       TTadd,
       TTsubtract,
       TTorthright,
       TTorthleft,
       TTorthleftright,
       TTnorm,
       TTCD_GH,
       TTCD_GC,
       TTCD_GL,
       TTCD_PL,
       TTextremize,
       TToptimize

end