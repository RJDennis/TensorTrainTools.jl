module TensorTrainTools

using GaussQuadrature
using Maxvol
using LinearAlgebra
using GenericLinearAlgebra

include("TTtoolbox.jl")

# Types
export TensorTrain, DiscreteTensorTrain, ContinuousTensorTrain, MixedTensorTrain,
       BaseTensorTrain, LeftOrthBaseTensorTrain, RightOrthBaseTensorTrain, ExtendedTensorTrain,
       LeftOrthExtendedTensorTrain, RightOrthExtendedTensorTrain, FunctionalTensorTrain,
       ChebyshevTensorTrain, LegendreTensorTrain, PiecewiseTensorTrain

# Building a train directly
export TTrand, TTrandn, TTones, TTzeros, TTconstant, TTChebyshev, TTLegendre

# Compressing an array or a sampled function
export TTsvd, TTsvd_threaded, TTals, DMRGcross, DMRGcross_generic, DMRGcross_threaded,
       DMRGcross_generic_threaded

# Continuous and functional trains
export TTcreateCTT, TTcreateLTT, TTcreatePTT, TTcreateFTT, TTinterp

# Evaluation and size
export TTevaluate, TTdecompress, TTdecompress_threaded, TTsize, TTnorm

# Algebra
export TTadd, TTsubtract, TTmult, TTHadamard, TTsquared, TTpower, TTexp, TTrounding,
       TTinner_prod, TTreverse_indices

# Orthogonalisation
export TTorthleft, TTorthright, TTorthleftright

# Derivatives
export TTderivative, TTgradient, TThessian

# Integration and marginals
export TTintegrate_GC, TTintegrate_GL, TTintegrate_GH, TTintegrate_PL, TTcompute_marginal_GC,
       TTcompute_marginal_GL, TTcompute_marginal_GH, TTcompute_marginal_PL

# Sampling
export TTCD_GC, TTCD_GL, TTCD_GH, TTCD_PL, TTSIRT_GC, TTSIRT_GL, TTSIRT_GH, TTSIRT_PL, TTMH,
       TTMHlog

# Optimisation over a train
export TToptimize, TTextremize, TTOpt, TTnewton, TTnewton_step, TTdescent, TTdescent_step

# Nodes, polynomials and interpolation
export cheb_nodes, legendre_nodes, piecewise_linear_nodes, normalize_node, cheb_polynomial,
       cheb_polynomial_deriv, cheb_polynomial_sec_deriv, legendre_polynomial,
       legendre_polynomial_deriv, legendre_polynomial_sec_deriv, cheb_weights, legendre_weights,
       cheb_evaluate, legendre_evaluate, piecewise_linear_evaluate, chebyshev_interp,
       legendre_interp

end
