# TensorTrainApprox

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RJDENNIS.github.io/TensorTrainApprox.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RJDENNIS.github.io/TensorTrainApprox.jl/dev/)
[![Build Status](https://github.com/RJDENNIS/TensorTrainApprox.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/RJDENNIS/TensorTrainApprox.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/RJDENNIS/TensorTrainApprox.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RJDENNIS/TensorTrainApprox.jl)

Introduction
============

TensorTrains.jl is a Julia package for computing and working with arrays stored in the tensor train format.  Tensor trains---also known as the Matrix Product State (MPS)---have their origins in quantum physics where they are used to ??????.  Rediscovered in the field of computational linear algebra, tensor trains are an important tool for array compression and for working with numerical problems that have high spacial dimension.  My interest in tensor trains comes from solving dynamic stochastic general equilibrium models with many state variables and from Bayesian estimation, but their breadth of application is far wider that that.

A tensor train is a system of connected 3d-arrays where the number of cores or carriages in the train reflects the number of dimensions in the system.  Think of a `D`-dimensional array that has `N` nodes along each dimension.  The number of elements in this array is `N^D`, which increases exponentially in `D`.  The number of nodes, `N`, and the number of dimensions, `D`, do not have to be that large before your compute will run out of memory trying to store the array.  The idea of the tenor train is to represent such an array in a compressed structure that contains far fewer elements than the original array, but that can be used to recover the elements in the original array to a controlable accuracy.  The size of the `k`'th core in a tensor train is governed by two rank indices, `r$_k$` and `r$_{k+1}$` and by the number of nodes in that spacial dimension, `N$_k$`, i.e. in the form of an `r$_k$ $\times$ N $\times$ r$_{k+1}$` array.  If we suppose that the ranks and the node-sizes are the same for all cores, then the number of elements in the tensor train in `$r^2$$\times$ND`, which increases quadratically in `r` and linearly in `D`.  If `r` is sufficiently small (and this is key), then the number of elemenets in the tensor train can be considerably smaller than the number in the original array. 

Initializing tensor trains
--------------------------

??????

Algebraic operations
--------------------

??????

Array compression
-----------------

??????

Function approximation
----------------------

??????

Functional tensor trains
------------------------

??????

Integration
-----------

??????

Sampling
--------

??????

Optimization
------------

??????

References
----------

Bigoni, D., Engsig-Karup, A., and Y. Marzouk, (2016), "Spectral tensor-train decomposition," *SIAM Journal on Scientific Computing*, 38, 4, pp. A2405--A2439.

Chertkov, A.., Ryzhakov, G., Novikov, G., and I. Oseledets, (2022), "Optimization of Functions given in the tensor train format," arXiv:2209.14808v1.

Cui, T., Dolgov, S., and O Zahm, (2023), "Scalable conditional deep inverse Rosenblatt transports using tensor trains and gradient-based dimension reduction," *Journal of Computational Physics*, 485, 112103.

Dolgov, S., Anaya-Izquierdo, K., Fox, C., and R. Scheichi, (2020), "Approximation and sampling of multivariate probability distributions in the tensor train decomposition," *Statistics and Computing*, 30, pp. 603--625.

Dolgov. S., and D. Savostyanov, (2020), "Parallel cross interpolation for high-precision calculation of high-dimensional integrals," *Computer Physics Communication*, 246, 106869.

Gorodetsky, A., Karaman, S., and Y. Marzouk, (2018), "A continuous analogue of the tensor train decomposition," *Computer Methods in Applied Mechanics and Engineering*, 347, pp. 59--84.

Oseledets, I., (2009), "Tensor train decomposotion," *SIAM Journal on Scientific Computing*, 33, 5, pp. 2295--2317.

Oseledets, I., and E. Tyrtyshnikov, (2010), "TT-cross approximation for multidimensional arrays," *Linear Algebra and its Applications*, 432, pp. 70--88.