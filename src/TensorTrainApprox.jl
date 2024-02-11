module TensorTrainApprox

using ChebyshevApprox
using GuassQuadrature
using Maxvol

include("TTtoolbox.jl")

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
       TTintegrate_GC,
       TTintegrate_GL,
       TTintegrate_GH,

end
