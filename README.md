# TensorTrainApprox

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://RJDENNIS.github.io/TensorTrainApprox.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://RJDENNIS.github.io/TensorTrainApprox.jl/dev/)
[![Build Status](https://github.com/RJDENNIS/TensorTrainApprox.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/RJDENNIS/TensorTrainApprox.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/RJDENNIS/TensorTrainApprox.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RJDENNIS/TensorTrainApprox.jl)

# Introduction

TensorTrainApprox.jl is a Julia package for computing and working with arrays stored in the tensor train format.  Tensor trains---also known as the Matrix Product State (MPS)---have their origins in computational physics where they are used to describe and analyze many-body quantum systems.  Rediscovered in the field of computational linear algebra, tensor trains are an important tool for array compression and for working with numerical problems that have high spacial dimension.  My interest in tensor trains comes from solving dynamic stochastic general equilibrium models with many state variables and from Bayesian estimation, but their breadth of application is far wider that that.

A tensor train is a system of connected 3d-arrays where the number of cores or carriages in the train reflects the number of dimensions in the system.  Think of a `D`-dimensional array that has `N` nodes along each dimension.  The number of elements in this array is $N^D$, which increases exponentially in `D`.  The number of nodes, `N`, and the number of dimensions, `D`, do not have to be that large before your computer will run out of memory trying to store the array.  The idea of the tenor train is to represent such an array in a compressed structure that contains far fewer elements than the original array, but that can be used to recover the elements in the original array to a controlable accuracy.  The size of the `k`'th core in a tensor train is governed by two rank indices, $r_{k}$ and $r_{k+1}$ and by the number of nodes in that spacial dimension, $N_{k}$, i.e. in the form of an $r_{k}$ $\times$ $N_{k}$ $\times$ $r_{k+1}$ array.  If we suppose that the ranks and the node-sizes are the same for all cores, then the number of elements in the tensor train is $r^2$ $\times$ N $\times$ D, which increases quadratically in `r` and linearly in `D`.  If `r` is sufficiently small (and this is key), then the number of elements in the tensor train can be considerably smaller than the number in the original array.  Reducing the number of elements in the tensor train further is the fact that $r_{1} = r_{D} = 1$.

## Initializing tensor trains

To initialize a discrete tensor train with `d` cores with rank `r`, and nodes `n` to a constant value, use:
```julia
train = TTconstant(value,n,r)
```
where `n` is a tuple of `d` integers specifying the number of nodes along each dimension and `r` is either an integer or a `d+1` vector of integers, with first and last elements equaling `1`, specifying the ranks.

A random discrete tensor train with elements drawn from a continuous uniform density can be initialized through:
```julia
train = TTrandom(n,r)
```
where, again, `n` is a tuple of `d` integers and `r` is either an integer or a `d+1` vector of integers whose first and last elements equal `1`.

Once constructed, tensor trains can be made left or right orthogonal using:
```julia
new_train = TTorthogleft(train)
new_train = TTorthogright(train)
```
The size of the dense array implied by the tensor train can be found by:
```julia
n = TTsize(train)
```
The dense array itself can be generated through:
```julia
A = decompress(train)
```

Lastly, the Frobenius norm of a tensor train can be computed using:
```julia
frob_norm = TTnorm(train)
```

## Algebraic operations

The package allows some standard algebraic manipulations to be performed on discrete tensor trains.

To multiply a tensor train by a scalar:
```julia
new_train = TTmult(s,train)
new_train = TTmult(train,s)
```
where `s` is a real number and `train` is a discrete tensor train.

Two discrete tensor trains that are conformable is the sense of having the same number of points along each dimension can be added to each other or subtracted from each other using:
```julia
new_train = TTadd(train_a,train_b)
new_train = TTsubtract(train_a,train_b)
```

Similarly, the inner product of two tensor trains is computed using:
```julia
inner_prod = TTinner_prod(train_a,train_b)
```

The square and the integer power of tensor trains is found by:
```julia
new_train = TTsquare(train)
new_train = TTpower(train,p)
``` 
where `p` is the desired integer power.

Most of the algebraic operations described above cause the resulting tensor train to have expanded ranks.  To reduce the ranks it is often usful to perform a rounding operation:
```julia
new_train = TTrounding(train,epsilon)
```
where `epsilon` is a small tolerance parameter.

## Array compression

If we have a dense array, `A`, then this array can be approximated to a prescribed accuracy using either:
```julia
train = TTSVD(A,tol)
train = DMRGcross(A,mu,tol,maxsweeps)
```
where `tol` is an accuracy parameter, `mu` (greater than one) determines convergence of the maxvol procedure, and `maxsweeps` is an optional integer specifying the maximum number of sweeps performed by the DMRGcross algorithm.  The defaukt `maxsweeps` is `30`, which is usually more than enough.

## Function approximation

To approximate a multivariate function without having to construct the dense array populated by this function:
```julia
train = DMRGcross(fn,grid,mu,tol,maxsweeps)
```
where `fn` is the function being approximated and `grid` is a tuple of vectors specifying the approximation grid.  By way of example:
```julia
function hilbert(x)

  y = 1/sum(x)

  return y

end

nodes = [1.0:0.05:3.0;]
grid = (nodes,nodes,nodes,nodes)
train = DMRGcross(hilbert,grid,1.05,1e-12)
```

## Functional tensor trains

??????

## Integration

??????

## Sampling

??????

## Optimization

??????

## References

Bigoni, D., Engsig-Karup, A., and Y. Marzouk, (2016), "Spectral tensor-train decomposition," *SIAM Journal on Scientific Computing*, 38, 4, pp. A2405--A2439.

Chertkov, A.., Ryzhakov, G., Novikov, G., and I. Oseledets, (2022), "Optimization of Functions given in the tensor train format," arXiv:2209.14808v1.

Cui, T., Dolgov, S., and O Zahm, (2023), "Scalable conditional deep inverse Rosenblatt transports using tensor trains and gradient-based dimension reduction," *Journal of Computational Physics*, 485, 112103.

Dolgov, S., Anaya-Izquierdo, K., Fox, C., and R. Scheichi, (2020), "Approximation and sampling of multivariate probability distributions in the tensor train decomposition," *Statistics and Computing*, 30, pp. 603--625.

Dolgov. S., and D. Savostyanov, (2020), "Parallel cross interpolation for high-precision calculation of high-dimensional integrals," *Computer Physics Communication*, 246, 106869.

Gorodetsky, A., Karaman, S., and Y. Marzouk, (2018), "A continuous analogue of the tensor train decomposition," *Computer Methods in Applied Mechanics and Engineering*, 347, pp. 59--84.

Oseledets, I., (2009), "Tensor train decomposotion," *SIAM Journal on Scientific Computing*, 33, 5, pp. 2295--2317.

Oseledets, I., and E. Tyrtyshnikov, (2010), "TT-cross approximation for multidimensional arrays," *Linear Algebra and its Applications*, 432, pp. 70--88.