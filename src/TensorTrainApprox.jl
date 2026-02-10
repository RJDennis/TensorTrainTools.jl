module TensorTrainApprox

using GaussQuadrature
using Maxvol
using LinearAlgebra
using GenericLinearAlgebra

include("TTtoolbox.jl")

export cheb_nodes,
       legendre_nodes,
       piecewise_linear_nodes

export BaseTensorTrain,
       ExtendedTensorTrain,
       ChebyshevTensorTrain,
       LegendreTensorTrain,
       PiecewiseTensorTrain,
       FunctionalTensorTrain

export TTconstant,
       TTrand,
       TTrandn,
       TTChebyshev,
       TTLegendre,
       TTsvd,
       DMRGcross,
       DMRGcross_generic,
       DMRGcross_threaded,
       DMRGcross_generic_threaded,
       TTcreateCTT,
       TTcreateLTT,
       TTcreatePTT,
       TTcreateFTT,
       TTinterp,
       TTdecompress,
       TTdecompress_threaded,
       TTevaluate,
       TTderivative,
       TTgradient,
       TTintegrate_GC,
       TTintegrate_GL,
       TTintegrate_GH,
       TTintegrate_PL,
       TTcompute_marginal_GH,
       TTcompute_marginal_GC,
       TTcompute_marginal_GL,
       TTcompute_marginal_PL,
       TTrounding,
       TTmult,
       TTadd,
       TTsubtract,
       TTsquared,
       TTpower,
       TTHadamard,
       TTorthright,
       TTorthleft,
       TTorthleftright,
       TTnorm,
       TTsize,
       TTinner_prod,
       TTCD_GH,
       TTCD_GC,
       TTCD_GL,
       TTCD_PL,
       TTreverse_indices,
       TTextremize,
       TToptimize,
       TTOpt,
       TTnewton,
       TTnewton_step,
       TTdescent,
       TTdescent_step

end