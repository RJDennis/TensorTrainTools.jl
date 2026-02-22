#### Structures that define and hold tensor trains

abstract type TensorTrain end
abstract type DiscreteTensorTrain <: TensorTrain end
abstract type ContinuousTensorTrain <: TensorTrain end
abstract type MixedTensorTrain <: TensorTrain end # Currently not used, placeholder

struct BaseTensorTrain{T<:AbstractFloat,S<:Integer} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{S,1}
  sweeps::S

end

struct LeftOrthBaseTensorTrain{T<:AbstractFloat,S<:Integer} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{S,1}
  sweeps::S

end

struct RightOrthBaseTensorTrain{T<:AbstractFloat,S<:Integer} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{S,1}
  sweeps::S

end

struct ExtendedTensorTrain{T<:AbstractFloat,S<:Integer} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{S,1}
  left_to_right_ind::Array{Array{S,1},1}
  right_to_left_ind::Array{Array{S,1},1}
  left_to_right_sub::Array{Array{Tuple{S,Vararg{S}}},1}
  right_to_left_sub::Array{Array{Tuple{S,Vararg{S}}},1}
  sweeps::S

end

struct LeftOrthExtendedTensorTrain{T<:AbstractFloat,S<:Integer} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{S,1}
  left_to_right_ind::Array{Array{S,1},1}
  right_to_left_ind::Array{Array{S,1},1}
  left_to_right_sub::Array{Array{Tuple{S,Vararg{S}}},1}
  right_to_left_sub::Array{Array{Tuple{S,Vararg{S}}},1}
  sweeps::S

end

struct RightOrthExtendedTensorTrain{T<:AbstractFloat,S<:Integer} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{S,1}
  left_to_right_ind::Array{Array{S,1},1}
  right_to_left_ind::Array{Array{S,1},1}
  left_to_right_sub::Array{Array{Tuple{S,Vararg{S}}},1}
  right_to_left_sub::Array{Array{Tuple{S,Vararg{S}}},1}
  sweeps::S

end

struct FunctionalTensorTrain{F,S<:Integer} <: TensorTrain

  cores::Array{Array{F,2},1}
  ranks::Array{S,1}
  sweeps::S

end

struct ChebyshevTensorTrain{T<:AbstractFloat,S<:Integer} <: ContinuousTensorTrain

  cores::Array{Array{Array{T,1},2},1}
  ranks::Array{S,1}
  sweeps::S

end

struct LegendreTensorTrain{T<:AbstractFloat,S<:Integer} <: ContinuousTensorTrain

  cores::Array{Array{Array{T,1},2},1}
  ranks::Array{S,1}
  sweeps::S

end

struct PiecewiseTensorTrain{T<:AbstractFloat,S<:Integer} <: ContinuousTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{S,1}
  sweeps::S

end

#### Utility functions

"""
Multiply two tensor train cores (3D arrays) to produce a 4D array.

Signature
=========

t = times_cores(A,B)
"""
function times_cores(A::AbstractArray{T,3},B::AbstractArray{T,3}) where {T<:Real}

  na = size(A)
  nb = size(B)

  if na[3] != nb[1]
    throw(DimensionMismatch("incompatible dimensions: A has $(na[3]) columns in dimension 3, B has $(nb[1]) rows in dimension 1"))
  end

  A = reshape(A,na[1]*na[2],na[3])
  B = reshape(B,nb[1],nb[2]*nb[3])
  C = reshape(A*B,na[1],na[2],nb[2],nb[3])

  return C

end

"""
Pre-multiply a 3d array, B, by a matrix, A, along the first dimension of the 3D array to produce a 3D array.

Signature
=========

t = times_dim_1(A,B)
"""
function times_dim_1(A::AbstractArray{T1,2},B::AbstractArray{T2,3}) where {T1<:Real,T2<:Real}

  T = promote_type(T1,T2)

  na = size(A)
  nb = size(B)

  if na[2] != nb[1]
    throw(DimensionMismatch("incompatible dimensions: A has $(na[2]) columns, B has $(nb[1]) rows in dimension 1"))
  end

  B = reshape(B,nb[1],nb[2]*nb[3])
  C = reshape(A*B,na[1],nb[2],nb[3])
  
  return C

end

"""
Pre-multiply a 4D array, B, by a matrix, A, along the first dimension of the 4D array to produce a 4D array.

Signature
=========

t = times_dim_1(A,B)
"""
function times_dim_1(A::AbstractArray{T1,2},B::AbstractArray{T2,4}) where {T1<:Real,T2<:Real}

  T = promote_type(T1,T2)

  na = size(A)
  nb = size(B)

  if na[2] != nb[1]
    throw(DimensionMismatch("incompatible dimensions: A has $(na[2]) columns, B has $(nb[1]) rows in dimension 1"))
  end

  B = reshape(B,nb[1],nb[2]*nb[3]*nb[4])
  C = reshape(A*B,na[1],nb[2],nb[3],nb[4])

  return C

end

"""
Pre-multiply a matrix, B, by a 3D array, A, along the third dimension of the 3D array to produce a 3D array.

Signature
=========

t = times_dim_3(A,B)
"""
function times_dim_3(A::AbstractArray{T1,3},B::AbstractArray{T2,2}) where {T1<:Real,T2<:Real}

  T = promote_type(T1,T2)

  na = size(A)
  nb = size(B)

  if na[3] != nb[1]
    throw(DimensionMismatch("incompatible dimensions: A has $(na[3]) columns in mode 3, B has $(nb[1]) rows"))
  end

  A = reshape(A,na[1]*na[2],na[3])
  C = reshape(A*B,na[1],na[2],nb[2])
  
  return C

end

"""
Pre-multiply a matrix, B, by a 4D array, A, along the fourth dimension of the 4D array to produce a 4D array.

Signature
=========

t = times_dim_4(A,B)
"""
function times_dim_4(A::AbstractArray{T1,4},B::AbstractArray{T2,2}) where {T1<:Real,T2<:Real}

  T = promote_type(T1,T2)

  na = size(A)
  nb = size(B)

  if na[4] != nb[1]
    throw(DimensionMismatch("incompatible dimensions: A has $(na[4]) columns in mode 4, B has $(nb[1]) rows"))
  end

  A = reshape(A,na[1]*na[2]*na[3],na[4])
  C = reshape(A*B,na[1],na[2],na[3],nb[2])

  return C

end

"""
Find the array sub-index from a linear index; returns a tuple or tuple of tuples if 'i' is a vector.

Signature
=========

s = ind2sub(i,dims)
"""
function ind2sub(i::S,dims::Tuple{S,Vararg{S}}) where {S<:Integer}

  if i < 1 || i > prod(dims)
    error("index is out of bounds.")
  end

  subs = Tuple(CartesianIndices(dims)[i])

  return subs

end

function ind2sub(i::AbstractArray{S,1},dims::Tuple{S,Vararg{S}}) where {S<:Integer}

  for x in i
    if x < 1 || x > prod(dims)
      error("index is out of bounds.")
    end
  end

  subs = Tuple.(CartesianIndices(dims)[i])

  return subs

end

function ind2sub(i::S,dims::AbstractArray{S,1}) where {S<:Integer}

  if i < 1 || i > prod(dims)
    error("index is out of bounds.")
  end

  subs = Tuple(CartesianIndices(Tuple(dims))[i])

  return subs

end

function ind2sub(i::AbstractArray{S,1},dims::AbstractArray{S,1}) where {S<:Integer}

  for x in i
    if x < 1 || x > prod(dims)
      error("index is out of bounds.")
    end
  end

  subs = Tuple(CartesianIndices(Tuple(dims))[i])

  return subs

end

"""
Compute the truncated SVD decomposition of a matrix with singular value threshold 'δ'.

Signatures
==========

t = tsvd(A,δ)
t = tsvd(A,δ,r)
"""
function tsvd(A::AbstractArray{T1,2},δ::T2) where {T1<:AbstractFloat,T2<:AbstractFloat} # Looks at norm of singular values

  u, s, v = svd(A)

  r = 0
  len = zero(T1)
  for i = length(s):-1:1
    len += s[i]^2
    if sqrt(len) >= max(δ,eps(T1))
      r = i
      break
    end
  end

  r = max(r,1)
  
  return u[:,1:r], s[1:r], v[:,1:r], r

end

function tsvd(A::AbstractArray{T,2},r::S) where {T<:AbstractFloat,S<:Integer} # Looks at norm of singular values

  n = size(A)
  if r > minimum(n) !! r < 1
    error("Invalid rank.")
  end

  u, s, v = svd(A)

  return u[:,1:r], s[1:r], v[:,1:r], r

end

#### Functions to initialize discrete tensor trains

"""
Create a tensor train with uniformly random entries of eltype(T) that has spacial dimensions 'n' and tensor ranks 'r' (an integer or vector of integers).

Signatures
==========

t = TTrand(n,r)
t = TTrand(n,r,T)
"""
function TTrand(n::NTuple{d,S},r::S,T::DataType=Float64) where {S<:Integer,d}

  r      = fill(r,d+1)
  r[1]   = 1
  r[d+1] = 1

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = rand(T,r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

function TTrand(n::NTuple{d,S},r::Array{S,1},T::DataType=Float64) where {S<:Integer,d}

  if length(r) != d+1
    error("Dimension mis-match between 'n' and 'r'.")
  end
  if r[begin] != r[end] || r[end] != 1
    error("The first and last ranks must equal 1.")
  end

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = rand(T,r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

"""
Create a tensor train with normally distributed random entries of eltype(T) that has spacial dimensions 'n' and tensor ranks 'r' (an integer or vector of integers).

Signatures
==========

t = TTrandn(n,r)
t = TTrandn(n,r,T)
"""
function TTrandn(n::NTuple{d,S},r::S,T::DataType=Float64) where {S<:Integer,d}

  r      = fill(r,d+1)
  r[1]   = 1
  r[d+1] = 1

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = randn(T,r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

function TTrandn(n::NTuple{d,S},r::Array{S,1},T::DataType=Float64) where {S<:Integer,d}

  if length(r) != d+1
    error("Dimension mis-match between 'n' and 'r'.")
  end
  if r[begin] != r[end] || r[end] != 1
    error("The first and last ranks must equal 1.")
  end

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = rand(T,r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

"""
Create a tensor train of a constant 'value' with spacial dimensions 'n' and tensor ranks 'r' (an integer or vector of integers).

Signature
=========

t = TTconstant(value,n,r)
"""
function TTconstant(value::T,n::NTuple{d,S},r::S) where {T<:AbstractFloat,S<:Integer,d}

  s = sign(value)
  v = abs(value)^(1/d)

  r      = fill(r,d+1)
  r[1]   = 1
  r[d+1] = 1

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = fill(v/r[i+1],r[i],n[i],r[i+1])
  end
  G[1] = s*G[1]

  return BaseTensorTrain(G,r,0)

end

function TTconstant(value::T,n::NTuple{d,S},r::Array{S,1}) where {T<:AbstractFloat,S<:Integer,d}

  if length(r) != d+1
    error("Dimension mis-match between 'n' and 'r'.")
  end
  if r[begin] != r[end] || r[end] != 1
    error("The first and last ranks must equal 1.")
  end
    
  s = sign(value)
  v = abs(value)^(1/d)

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = fill(v/r[i+1],r[i],n[i],r[i+1])
  end
  G[1] = s*G[1]

  return BaseTensorTrain(G,r,0)

end

"""
Create a tensor train of a ones of eltype(T) with spacial dimensions 'n' and tensor ranks 'r' (an integer or vector of integers).

Signatures
==========

t = TTones(n,r)
t = TTones(n,r,T)
"""
function TTones(n::NTuple{d,S},r::S,T::DataType=Float64) where {S<:Integer,d}

  r      = fill(r,d+1)
  r[1]   = 1
  r[d+1] = 1

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = fill(1.0/r[i+1],r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

function TTones(n::NTuple{d,S},r::Array{S,1},T::DataType=Float64) where {S<:Integer,d}

  if length(r) != d+1
    error("Dimension mis-match between 'n' and 'r'.")
  end
  if r[begin] != r[end] || r[end] != 1
    error("The first and last ranks must equal 1.")
  end
    
  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = fill(1.0/r[i+1],r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

"""
Create a tensor train of zeros of eltype(T) with spacial dimensions 'n' and tensor ranks 'r'.

Signatures
==========

t = TTzeros(n,r)
t = TTzeros(n,r,T)
"""
function TTzeros(n::NTuple{d,S},r::S,T::DataType=Float64) where {S<:Integer,d}

  r      = fill(r,d+1)
  r[1]   = 1
  r[d+1] = 1

  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = fill(T(0.0),r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

function TTzeros(n::NTuple{d,S},r::Array{S,1},T::DataType=Float64) where {S<:Integer,d}

  if length(r) != d+1
    error("Dimension mis-match between 'n' and 'r'.")
  end
  if r[begin] != r[end] || r[end] != 1
    error("The first and last ranks must equal 1.")
  end
    
  G = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    G[i] = fill(T(0.0),r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(G,r,0)

end

#### Functions to initialize continuous tensor trains

"""
Create a Chebyshev tensor train with ranks, 'r', orders, 'order', and Chebyshev coefficients, 'θ'.

Signature
=========

t = TTChebyshev(r,order,θ)
"""
function TTChebyshev(r::Array{S,1},order::Array{S,1},θ::Array{T,1}) where {S<:Integer,T<:AbstractFloat}

  d = length(order)

  if length(r) != d+1
    error("Inconsistency regarding the number of spacial dimensions.")
  end

  if r[begin] != r[end] || r[begin] != 1
    error("The first and last elements of 'r' must equal 1")
  end

  N = Array{S,1}(undef,d+1)
  N[1] = zero(S)
  for k = 1:d
    N[k+1] = N[k] + r[k]*r[k+1]*order[k]
  end

  if length(θ) != N[end]
    error("The length of θ should be $(N[end])")
  end

  cores = Array{Array{Array{T,1},2},1}(undef,d)
  for k = 1:d
    c = reshape(θ[N[k]+1:N[k+1]],r[k],order[k],r[k+1])
    cores[k] = [c[i,:,j] for i in 1:r[k], j in 1:r[k+1]]
  end

  return ChebyshevTensorTrain(cores,r,0)

end

"""
Create a Legendre tensor train with ranks, 'r', orders, 'order', and Legendre coefficients, 'θ'.

Signature
=========

t = TTLegendre(r,order,θ)
"""
function TTLegendre(r::Array{S,1},order::Array{S,1},θ::Array{T,1}) where {S<:Integer,T<:AbstractFloat}

  d = length(order)
  if length(r) != d+1
    error("Inconsistency regarding the number of spacial dimensions.")
  end

  if r[begin] != r[end] || r[begin] != 1
    error("The first and last elements of 'r' must equal 1")
  end

  N = Array{S,1}(undef,d+1)
  N[1] = zero(S)
  for k = 1:d
    N[k+1] = N[k] + r[k]*r[k+1]*order[k]
  end

  if length(θ) != N[end]
    error("The length of θ should be $(N[end])")
  end

  cores = Array{Array{Array{T,1},2},1}(undef,d)
  for k = 1:d
    c = reshape(θ[N[k]+1:N[k+1]],r[k],order[k],r[k+1])
    cores[k] = [c[i,:,j] for i in 1:r[k], j in 1:r[k+1]]
  end

  return LegendreTensorTrain(cores,r,0)

end

#### Functions to compute discrete tensor trains from an array

"""
Tensor compression of the dense d-dimensional array, A.

Signatures
==========

t = TTals(A,r,tol)
t = TTals(A,r,tol,maxsweeps)
t = TTals(A,r,tol,maxsweeps,seed)
"""
function TTals(A::AbstractArray{T,d},r::S,tol::T1,maxsweeps::S=100,seed::S=123456) where{T<:AbstractFloat,T1<:AbstractFloat,S<:Integer,d}

  if d == 1
    return BaseTensorTrain([reshape(A,1,:,1)],[1,1],0)
  end

  Random.seed!(seed)

  N = size(A)

  train = TTrand(N,r,T)
  cores = copy(train.cores)
  r     = copy(train.ranks)

  len = [T(Inf) for _ in 1:d]

  sweeps = 0

  while sweeps < maxsweeps

    for p = d:-1:1 # p is the core we are solving for

      M = copy(A)

      for i = 1:p-1 # sweep left-to-right left-orthogonalising the first p-1 cores

        M = reshape(M,r[i]*N[i],prod(N[i+1:end]))
        G = cores[i]
        n = size(G)
        G = reshape(G,n[1]*n[2],n[3])
        Q,R = qr(G)
        cores[i]   = reshape(Matrix(Q),n)
        cores[i+1] = times_dim_1(R,cores[i+1])
        M = Matrix(Q)'M

      end

      for i in d:-1:p+1 # sweep right-to-left right-orthogonalising the last p+1 cores

        M = reshape(M,r[p]*prod(N[p:i-1]),N[i]*r[i+1])
        G = cores[i]
        n = size(G)
        G = reshape(G,n[1],n[2]*n[3])
        R,Q = rq(G)
        cores[i]   = reshape(Q,n)
        cores[i-1] = times_dim_3(cores[i-1],R)
        M = M*Matrix(Q)'

      end

      new_core = reshape(M,r[p],N[p],r[p+1])
      len[p]   = norm(new_core-cores[p])
      cores[p] = new_core

    end

    sweeps += 1

    maximum(len) <= tol && break

  end

  return BaseTensorTrain(cores,r,sweeps)

end

function TTals(A::AbstractArray{T,d},r::Array{S,1},tol::T1,maxsweeps::S=100,seed::S=123456) where{T<:AbstractFloat,T1<:AbstractFloat,S<:Integer,d}

  if d == 1
    return BaseTensorTrain([reshape(A,1,:,1)],[1,1],0)
  end

  Random.seed!(seed)

  N = size(A)

  train = TTrand(N,r,T)
  cores = copy(train.cores)
  r     = copy(train.ranks)

  len = [T(Inf) for _ in 1:d]

  sweeps = 0

  while sweeps < maxsweeps

    for p = d:-1:1 # p is the core we are solving for

      M = copy(A)
      N = size(M)

      for i = 1:p-1 # sweep left-to-right left-orthogonalising the first p-1 cores

        M = reshape(M,r[i]*N[i],prod(N[i+1:end]))
        G = cores[i]
        n = size(G)
        G = reshape(G,n[1]*n[2],n[3])
        Q, R = qr(G)
        cores[i]   = reshape(Matrix(Q),n)
        cores[i+1] = times_dim_1(R,cores[i+1])
        M = Matrix(Q)'M

      end

      for i in d:-1:p+1 # sweep right-to-left right-orthogonalising the last p+1 cores

        M = reshape(M,r[p]*prod(N[p:i-1]),N[i]*r[i+1])
        G = cores[i]
        n = size(G)
        G = reshape(G,n[1],n[2]*n[3])
        R, Q = rq(G)
        cores[i]   = reshape(Matrix(Q),n)
        cores[i-1] = times_dim_3(cores[i-1],R)
        M = M*Matrix(Q)'

      end

      new_core = reshape(M,r[p],N[p],r[p+1])
      len[p] = norm(new_core-cores[p])
      cores[p] = new_core

    end

    sweeps += 1

    maximum(len) <= tol && break

  end

  return BaseTensorTrain(cores,r,sweeps)

end

"""
Tensor compression of the dense d-dimensional array, A. 

Signatures
==========

t = TTsvd(A,tol)
t = TTsvd(A,r)
"""
function TTsvd(A::AbstractArray{T1,d},tol::T2) where {T1<:AbstractFloat,T2<:AbstractFloat,d} # Based on the description given in Oseledets and Tyrtyshnikov (2010).

  if d == 1
    return BaseTensorTrain([reshape(A,1,:,1)],[1,1],0)
  end

  δ = (tol/sqrt(d-1))*norm(A)

  n = size(A)
  r = ones(Int,d+1)
  G = Array{Array{T1,3},1}(undef,d)

  Nl = n[1]
  Nr = prod(n[2:d]) # Potential for integer overflow
  M = reshape(A,r[1]*Nl,Nr)

  u,s,v,r[2] = tsvd(M,δ)
  G[1] = reshape(u,r[1],Nl,r[2])
  M = Diagonal(s)*v'

  for k = 2:(d-1)

    Nl = n[k]
    Nr = div(Nr,n[k])
    M = reshape(M,r[k]*Nl,Nr)

    u,s,v,r[k+1] = tsvd(M,δ)
    G[k] = reshape(u,r[k],n[k],r[k+1])
    M = Diagonal(s)*v'

  end

  G[d] = reshape(M,r[d],n[d],1)

  return BaseTensorTrain(G,r,1)

end

function TTsvd(A::AbstractArray{T,d},r::Array{S,1}) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Oseledets and Tyrtyshnikov (2010).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  if d == 1
    return BaseTensorTrain([reshape(A,1,:,1)],[1,1],0)
  end

  n = size(A)
  G = Array{Array{T,3},1}(undef,d)

  Nl = n[1]
  Nr = prod(n[2:d]) # Potential for integer overflow
  M = reshape(A,Nl,Nr)

  u,s,v,r[2] = tsvd(M,r[2])
  G[1] = reshape(u,r[1],Nl,r[2])
  M = Diagonal(s)*v'

  for k = 2:(d-1)

    Nl = n[k]
    Nr = div(Nr,n[k])
    M = reshape(M,r[k]*Nl,Nr)

    u,s,v,r[k+1] = tsvd(M,r[k+1])
    G[k] = reshape(u,r[k],n[k],r[k+1])
    M = Diagonal(s)*v'

  end

  G[d] = reshape(M,r[d],n[d],1)

  return BaseTensorTrain(G,r,1)

end

function TTsvd(A::AbstractArray{T,d},r::S) where {T<:AbstractFloat,S<:Integer,d}

  ranks       = ones(Int,d+1)
  ranks[2:d] .= r

  train = TTsvd(A,ranks)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array, B, with max-volume factor, μ > 1.0.

Signatures
==========

t = DMRGcross(B,μ,tol)
t = DMRGcross(B,μ,tol,maxsweeps)
t = DMRGcross(B,μ,r)
t = DMRGcross(B,μ,r,maxsweeps)
"""
function DMRGcross(B::AbstractArray{T,d},μ::T,tol::T,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020)

  # When d ≤ 2

  if d <= 2
    return TTsvd(B,tol)
  end

  # When d ≥ 3

  n = size(B)

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  # Create a container to hold the cores

  G = Array{Array{T,3},1}(undef,d)

  # Create containers to hold the linear indices and sub-indices

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indicies

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indicies

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    @views G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross(B::AbstractArray{T,d},μ::T,r::Array{S,1},maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  n = size(B)

  # When d ≤ 2

  if d <= 2
    return TTsvd(b,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    @views G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end
  
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross(B::AbstractArray{T,d},μ::T,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d}

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross(B,μ,ranks,maxsweeps)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array, B, with max-volume factor, μ > 1.0.

Signatures
==========

t = DMRGcross_generic(B,μ,tol)
t = DMRGcross_generic(B,μ,tol,maxsweeps)
t = DMRGcross_generic(B,μ,r)
t = DMRGcross_generic(B,μ,r,maxsweeps)
"""
function DMRGcross_generic(B::AbstractArray{T,d},μ::R,tol::R,maxsweeps::S = 6) where {T<:AbstractFloat,R<: AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  if d <= 2
    return TTsvd(B,tol)
  end

  # When d ≥ 3

  n = size(B)

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  # Create a container to hold the cores

  G = Array{Array{T,3},1}(undef,d)

  # Create containers to hold the linear indices and sub-indices

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.

  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indicies

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indicies

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic(B::AbstractArray{T,d},μ::R,r::Array{S,1},maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  n = size(B)

  # When d ≤ 2

  if d <= 2
    return TTsvd(b,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.

  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end
  
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic(B::AbstractArray{T,d},μ::R,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross_generic(B,μ,ranks,maxsweeps)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array, B, with max-volume factor, μ > 1.0.

Signatures
==========

t = DMRGcross_threaded(B,μ,tol)
t = DMRGcross_threaded(B,μ,tol,maxsweeps)
t = DMRGcross_threaded(B,μ,r)
t = DMRGcross_threaded(B,μ,r,maxsweeps)
"""
function DMRGcross_threaded(B::AbstractArray{T,d},μ::T,tol::T,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  if d <= 2
    return TTsvd(B,tol)
  end

  # When d ≥ 3

  n = size(B)

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  # Create a container to hold the cores

  G = Array{Array{T,3},1}(undef,d)

  # Create containers to hold the linear indices and sub-indices

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.

  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indicies

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indicies

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_threaded(B::AbstractArray{T,d},μ::T,r::Array{S,1},maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  n = size(B)

  # When d ≤ 2

  if d <= 2
    return TTsvd(b,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.

  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i+1] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end
  
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_threaded(B::AbstractArray{T,d},μ::T,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d}

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross_threaded(B,μ,ranks,maxsweeps)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array, B, with max-volume factor, μ > 1.0.

Signatures
==========

t = DMRGcross_generic_threaded(B,μ,tol)
t = DMRGcross_generic_threaded(B,μ,tol,maxsweeps)
t = DMRGcross_generic_threaded(B,μ,r)
t = DMRGcross_generic_threaded(B,μ,r,maxsweeps)
"""
function DMRGcross_generic_threaded(B::AbstractArray{T,d},μ::R,tol::R,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  if d <= 2
    return TTsvd(B,tol)
  end

  # When d ≥ 3

  n = size(B)

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  # Create a container to hold the cores

  G = Array{Array{T,3},1}(undef,d)

  # Create containers to hold the linear indices and sub-indices

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.

  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indicies

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indicies

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic_threaded(B::AbstractArray{T,d},μ::R,r::Array{S,1},maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  n = size(B)

  # When d ≤ 2

  if d <= 2
    return TTsvd(b,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.

  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize the left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize the right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  r_temp = similar(r)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end
  
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    r_temp .= r

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        A[j] = B[CartesianIndex(point_index)]
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      A[j] = B[CartesianIndex(point_index)]
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic_threaded(B::AbstractArray{T,d},μ::R,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross_generic_threaded(B,μ,ranks,maxsweeps)

  return train

end

#### Functions to compute discrete tensor trains from a function

"""
Compute a discrete tensor train based on the function 'f' that populates the array.

Signature
=========

t = TTals(f,nodes,r,tol)
t = TTals(f,nodes,r,tol,maxsweeps)
t = TTals(f,nodes,r,tol,maxsweeps,seed)
"""
function TTals(f::Function,nodes::NTuple{d,Array{T,1}},r::S,tol::T,maxsweeps::S=100,seed::S=123456) where {T<:AbstractFloat,S<:Integer,d}

  n = length.(nodes)

  A = Array{T,d}(undef,n)
  for i in CartesianIndices(A)
    point = zeros(d)
    for j in 1:d
      point[j] = nodes[j][i[j]]
    end
    A[i] = f(point)
  end

  train = TTals(A,r,tol,maxsweeps,seed)

  return train

end

"""
Compute a discrete tensor train based on the function 'f' that populates the array.

Signature
=========

t = TTsvd(f,nodes,tol)
"""
function TTsvd(f::Function,nodes::NTuple{d,Array{T,1}},tol::T) where {T<:AbstractFloat,d}

  n = length.(nodes)

  A = Array{T,d}(undef,n)
  for i in CartesianIndices(A)
    point = zeros(d)
    for j = 1:d
      point[j] = nodes[j][i[j]]
    end
    A[i] = f(point)
  end

  train = TTsvd(A,tol)

  return train

end

"""
Compute a discrete tensor train based on the function 'f' that populates the array.

Signature
=========

t = TTsvd_threaded(f,nodes,tol)
"""
function TTsvd_threaded(f::Function,nodes::NTuple{d,Array{T,1}},tol::T) where {T<:AbstractFloat,d}

  n = length.(nodes)

  A = Array{T,d}(undef,n)
  Threads.@threads for i in CartesianIndices(A)
    point = zeros(d)
    for j = 1:d
      point[j] = nodes[j][i[j]]
    end
    A[i] = f(point)
  end

  train = TTsvd(A,tol)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.

Signatures
==========

t = DMRGcross(f,nodes,μ,tol)
t = DMRGcross(f,nodes,μ,tol,maxsweeps)
t = DMRGcross(f,nodes,μ,tol,initial)
t = DMRGcross(f,nodes,μ,tol,initial,maxsweeps)
t = DMRGcross(f,nodes,initial)
t = DMRGcross(f,nodes,μ,r)
t = DMRGcross(f,nodes,μ,r,maxsweeps)
"""
function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  n = length.(nodes)

  # When d ≤ 2

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # When d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the left indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
            A[l,j[1],j[2],k] = f(point)
          end
        end
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views G[i]   = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       A[k,j[1],j[2]] = f(point)
      end
    end

    A = reshape(A,r[d-1]*n[d-1],n[d]*r[d+1])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
      
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end
    
    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)
    
    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,initial::ExtendedTensorTrain,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # The following is used when d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = deepcopy(initial.left_to_right_ind)
  right_to_left_indices = deepcopy(initial.right_to_left_ind)
  
  left_to_right_subs = deepcopy(initial.left_to_right_sub)
  right_to_left_subs = deepcopy(initial.right_to_left_sub)
  
  r = copy(initial.ranks)

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
            A[l,j[1],j[2],k] = f(point)
          end
        end
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[k,j[1],j[2],1] = f(point)
      end
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
      
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end
    
    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)
    
    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},initial::ExtendedTensorTrain) where {T<:AbstractFloat,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,initial.ranks)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,initial.ranks)
  end

  # The following is used when d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = deepcopy(initial.left_to_right_ind)
  right_to_left_indices = deepcopy(initial.right_to_left_ind)
  
  left_to_right_subs = deepcopy(initial.left_to_right_sub)
  right_to_left_subs = deepcopy(initial.right_to_left_sub)
  
  r = copy(initial.ranks)

  point_index = Array{eltype(r),1}(undef,d)
  point       = Array{T,1}(undef,d)

  # Sweep from left to right, keeping the indices nested

  # Solve for the first indices

  A = Array{T,3}(undef,(n[1],n[2],r[3]))
  for k = 1:r[3]
    point_index[3:end] .= right_to_left_subs[2][k]
    for j in CartesianIndices((1:n[1],1:n[2]))
      point_index[1:2] .= Tuple(j)
      for m = 1:d
        point[m] = nodes[m][point_index[m]]
      end
      A[j[1],j[2],k] = f(point)
    end
  end

  A = reshape(A,n[1],n[2]*r[3])

  u,s,v,junk = tsvd(A,r[2])

  @views G[1] = reshape(A[:,right_to_left_indices[1]]/A[left_to_right_indices[1],right_to_left_indices[1]],r[1],n[1],r[2])

  # Solve for the interior indices
    
  for i = 2:d-2

    A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
    for l = 1:r[i]
      point_index[1:i-1] .= left_to_right_subs[i-1][l]
      for k = 1:r[i+2]
        point_index[i+2:end] .= right_to_left_subs[i+1][k]
        for j in CartesianIndices((1:n[i],1:n[i+1]))
          point_index[i:i+1] .= Tuple(j)
          for m = 1:d
            point[m] = nodes[m][point_index[m]]
          end
          A[l,j[1],j[2],k] = f(point)
        end
      end
    end

    A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
    u,s,v,junk = tsvd(A,r[i+1])
  
    @views G[i] = reshape(A[:,right_to_left_indices[i]]/A[left_to_right_indices[i],right_to_left_indices[i]],r[i],n[i],r[i+1])

  end

  # Solve for the final indices

  A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
  for k = 1:r[d-1]
    point_index[1:d-2] .= left_to_right_subs[d-2][k]
    for j in CartesianIndices((1:n[d-1],1:n[d]))
      point_index[d-1:d] .= Tuple(j)
      for m = 1:d
        point[m] = nodes[m][point_index[m]]
      end
      A[k,j[1],j[2],1] = f(point)
    end
  end

  A = reshape(A,r[d-1]*n[d-1],n[d])
  u,s,v,junk = tsvd(A,r[d])

  @views G[d-1] = reshape(A[:,right_to_left_indices[d-1]]/A[left_to_right_indices[d-1],right_to_left_indices[d-1]],r[d-1],n[d-1],r[d])
  @views G[d]   = reshape(A[left_to_right_indices[d-1],:],r[d],n[d],r[d+1])
    
  return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,1)

end

function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,r::Array{S,1},maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,r)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the left indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views G[i]   = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       A[k,j[1],j[2]] = f(point)
      end
    end

    A = reshape(A,r[d-1]*n[d-1],n[d]*r[d+1])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
      
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end
    
    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])
    
    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross(f,nodes,μ,ranks,maxsweeps)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.

Signatures
==========

t = DMRGcross_generic(f,nodes,μ,tol)
t = DMRGcross_generic(f,nodes,μ,tol,maxsweeps)
t = DMRGcross_generic(f,nodes,μ,tol,initial)
t = DMRGcross_generic(f,nodes,μ,tol,initial,maxsweeps)
t = DMRGcross_generic(f,nodes,μ,r)
t = DMRGcross_generic(f,nodes,μ,r,maxsweeps)
"""
function DMRGcross_generic(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,tol::R,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # When d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       A[k,j[1],j[2],1] = f(point)
      end
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
      
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end
    
    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)
    
    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,tol::R,initial::ExtendedTensorTrain,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = deepcopy(initial.left_to_right_ind)
  right_to_left_indices = deepcopy(initial.right_to_left_ind)

  left_to_right_subs = deepcopy(initial.left_to_right_sub)
  right_to_left_subs = deepcopy(initial.right_to_left_sub)

  r = copy(initial.ranks)

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       A[k,j[1],j[2],1] = f(point)
      end
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
      
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
    
    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end
    
    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)
    
    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,r::Array{S,d},maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,r)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       A[k,j[1],j[2],1] = f(point)
      end
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           A[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
      
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        A[j[1],j[2],k] = f(point)
      end
    end
    
    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])
    
    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross_generic(f,nodes,μ,ranks,maxsweeps)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.

Signatures
==========

t = DMRGcross_threaded(f,nodes,μ,tol)
t = DMRGcross_threaded(f,nodes,μ,tol,maxsweeps)
t = DMRGcross_threaded(f,nodes,μ,tol,initial)
t = DMRGcross_threaded(f,nodes,μ,tol,initial,maxsweeps)
t = DMRGcross_threaded(f,nodes,initial)
t = DMRGcross_threaded(f,nodes,μ,r)
t = DMRGcross_threaded(f,nodes,μ,r,maxsweeps)
"""
function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # When d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     A[j[1],j[2],j[3],1] = f(p)
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]
  
    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,initial::ExtendedTensorTrain,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = deepcopy(initial.left_to_right_ind)
  right_to_left_indices = deepcopy(initial.right_to_left_ind)

  left_to_right_subs = deepcopy(initial.left_to_right_sub)
  right_to_left_subs = deepcopy(initial.right_to_left_sub)

  r = copy(initial.ranks)
  
  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     A[j[1],j[2],j[3],1] = f(p)
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},initial::ExtendedTensorTrain) where {T<:AbstractFloat,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,initial.ranks)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,initial.ranks)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = deepcopy(initial.left_to_right_ind)
  right_to_left_indices = deepcopy(initial.right_to_left_ind)

  left_to_right_subs = deepcopy(initial.left_to_right_sub)
  right_to_left_subs = deepcopy(initial.right_to_left_sub)

  r = copy(initial.ranks)
  
  # Sweep from left to right, keeping the indices nested

  # Solve for the first indices

  A = Array{T,3}(undef,(n[1],n[2],r[3]))
  Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
    point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
    p         = Array{T,1}(undef,d)
    for m = 1:d
      p[m] = nodes[m][point_ind[m]]
    end
    A[j] = f(p)
  end
  
  A = reshape(A,n[1],n[2]*r[3])
  u,s,v,junk = tsvd(A,r[2])
  
  @views G[1] = reshape(A[:,right_to_left_indices[1]]/A[left_to_right_indices[1],right_to_left_indices[1]],r[1],n[1],r[2])

  # Solve for the interior indices
    
  for i = 2:d-2

    A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
    Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
      point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
    u,s,v,junk = tsvd(A,r[i+1])
  
    @views G[i] = reshape(A[:,right_to_left_indices[i]]/A[left_to_right_indices[i],right_to_left_indices[i]],r[i],n[i],r[i+1])

  end

  # Solve for the final indices

  A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
  Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
    point_ind = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
    p         = Array{T,1}(undef,d)
    for m = 1:d
      p[m] = nodes[m][point_ind[m]]
    end
   A[j[1],j[2],j[3],1] = f(p)
  end

  A = reshape(A,r[d-1]*n[d-1],n[d])
  u,s,v,junk = tsvd(A,r[d])

  @views G[d-1] = reshape(A[:,right_to_left_indices[d-1]]/A[left_to_right_indices[d-1],right_to_left_indices[d-1]],r[d-1],n[d-1],r[d])
  @views G[d]   = reshape(A[left_to_right_indices[d-1],:],r[d],n[d],r[d+1])

  return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,1)

end

function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,r::Array{S,1},maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,r)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind =  (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     A[j[1],j[2],j[3],1] = f(p)
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol!(v,μ,300)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol!(v,μ,300)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]
  
    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross_threaded(f,nodes,μ,ranks,maxsweeps)

  return train

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.

Signatures
==========

t = DMRGcross_generic_threaded(f,nodes,μ,tol)
t = DMRGcross_generic_threaded(f,nodes,μ,tol,maxsweeps)
t = DMRGcross_generic_threaded(f,nodes,μ,tol,initial)
t = DMRGcross_generic_threaded(f,nodes,μ,tol,initial,maxsweeps)
t = DMRGcross_generic_threaded(f,nodes,μ,r)
t = DMRGcross_generic_threaded(f,nodes,μ,r,maxsweeps)
"""
function DMRGcross_generic_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,tol::R,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # When d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
    
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     A[j[1],j[2],j[3],1] = f(p)
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,tol::R,initial::ExtendedTensorTrain,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,tol)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,tol)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = deepcopy(initial.left_to_right_ind)
  right_to_left_indices = deepcopy(initial.right_to_left_ind)

  left_to_right_subs = deepcopy(initial.left_to_right_sub)
  right_to_left_subs = deepcopy(initial.right_to_left_sub)

  r = copy(initial.ranks)
  
  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     A[j[1],j[2],j[3],1] = f(p)
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[d] = tsvd(A,δ)

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from tight to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      δ = (tol/sqrt(d-1))*norm(A)
      u,s,v,r[i+1] = tsvd(A,δ)
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r[2] = tsvd(A,δ)

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,r::Array{S,1},maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  if length(r) != d+1
    error("Rank vector has incorrect length")
  end

  if r[1] != r[d+1] || r[1] != 1
    error("Rank vector must begin and end with 1")
  end

  # When d ≤ 2

  n = length.(nodes)

  if d == 1
    A = [f(x) for x in nodes[1]]
    return TTsvd(A,r)
  elseif d == 2
    A = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(A,r)
  end

  # When d ≥ 3

  G = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # We first sweep from left to right, so only the right indices really need to be initialized.  We initialize both 
  # set of indices so that convergence can be checked at the end of the first left-right sweep.
  # The initial linear indices are centered around the middle nodes.
  # The initial sub-indices are constructed from the linear indices.

  # Initialize left-to-right indices and sub-indices

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(x) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  # Initialize right-to-left indices and sub-indices

  for i = d-1:-1:1

    p = div(n[i+1] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(x) for x in left_to_right_indices_new[1]]

    G[1] = reshape(A[:,right_to_left_indices_new[1]]/A[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      G[i] = reshape(A[:,right_to_left_indices_new[i]]/A[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    A = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = (left_to_right_subs[d-2][j[1]]...,j[2],j[3])
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     A[j[1],j[2],j[3],1] = f(p)
    end

    A = reshape(A,r[d-1]*n[d-1],n[d])
    u,s,v,r[d] = tsvd(A,r[d])

    left_to_right_indices_new[d-1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[d-1], _ = maxvol_generic!(v,μ,300)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(x) for x in right_to_left_indices_new[d-1]]

    @views G[d-1] = reshape(A[:,right_to_left_indices_new[d-1]]/A[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views G[d]   = reshape(A[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    r_temp = copy(r)

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      A = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = (left_to_right_subs[i-1][j[1]]...,j[2],j[3],right_to_left_subs[i+1][j[4]]...)
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        A[j] = f(p)
      end

      A = reshape(A,r[i]*n[i],r[i+2]*n[i+1])
      u,s,v,r[i+1] = tsvd(A,r[i+1])
  
      left_to_right_indices_new[i], _ = maxvol_generic!(u,μ,300)
      right_to_left_indices_new[i], _ = maxvol_generic!(v,μ,300)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    A = Array{T,3}(undef,(n[1],n[2],r[3]))
    Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = (j[1],j[2],right_to_left_subs[2][j[3]]...)
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      A[j] = f(p)
    end

    A = reshape(A,n[1],n[2]*r[3])
    u,s,v,r[2] = tsvd(A,r[2])

    left_to_right_indices_new[1], _ = maxvol_generic!(u,μ,300)
    right_to_left_indices_new[1], _ = maxvol_generic!(v,μ,300)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    all(isempty.(setdiff.(left_to_right_indices_new, left_to_right_indices))) &&
    all(isempty.(setdiff.(right_to_left_indices_new, right_to_left_indices))) &&
    return ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      return isempty(setdiff(r,r_temp)) ?
        ExtendedTensorTrain(G,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps) :
        BaseTensorTrain(G,r_temp,sweeps)
    end

  end

end

function DMRGcross_generic_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,r::S,maxsweeps::S = 6) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d} # Based on the description given in Dolgov and Savostyanov (2020).

  ranks = ones(Int,d+1)
  ranks[2:d] .= r

  train = DMRGcross_generic_threaded(f,nodes,μ,ranks,maxsweeps)

  return train

end

#### Functions needed to transform discrete tensor trains into continuous tensor trains

"""
Normalize 'node' so that it resides within the interval [-1,1]. 'node' can be a scalar or a vector.

Signature
=========

new_node = normalize_node(node,domain)
"""
function normalize_node(node::R,domain::AbstractVector{T}) where {R<:Real,T<:AbstractFloat}

  if domain[1] == domain[2]
    norm_node = zero(R)
    return norm_node
  else
    return 2*(node - domain[2])/(domain[1] - domain[2]) - one(R)
  end

end

function normalize_node(node::Array{R,1},domain::AbstractVector{T}) where {R<:Real,T<:AbstractFloat}

  norm_nodes = map(x -> normalize_node(x,domain),node)
  return norm_nodes

end

#### Functions for working with Chebyshev nodes/polynomials

"""
Create a vector of 'n' Chebyshev nodes on 'domain' with element type 'T'.

Signatures
==========

nodes = cheb_nodes(n)
nodes = cheb_nodes(n,domain)
nodes = cheb_nodes(n,domain,T)
"""
function cheb_nodes(n::S,domain = [1.0,-1.0],T::DataType=Float64) where {S<:Integer}

  points = [(domain[1] + domain[2]) * T(0.5) for _ in 1:n]

  @inbounds for i = 1:div(n, 2)
    x = -cospi(T(i - 0.5)/n) * (domain[1] - domain[2]) * T(0.5)
    points[i] += x
    points[n-i+1] -= x
  end

  return points

end

"""
Generate Chebyshev polynomials of 'order' at 'point'. 'point' can be a scalar or a vector of scalars.

Signature
=========

P = chebyshev_polynomial(order,point)
"""
function cheb_polynomial(order::S,point::R) where {S<:Integer,R<:Real}

  poly = Array{R}(undef, 1, order + 1)
  poly[1] = one(R)

  @inbounds for i = 2:order+1
    if i == 2
      poly[i] = point
    else
      poly[i] = 2*point*poly[i-1] - poly[i-2]
    end
  end

  return poly

end

function cheb_polynomial(order::S,point::AbstractArray{R,1}) where {S<:Integer,R<:Real}

  poly = Array{R}(undef, length(point), order + 1)
  poly[:,1] .= ones(R, length(point))

  @inbounds for i = 2:order+1
    for j in eachindex(point)
      if i == 2
        poly[j,i] = point[j]
      else
        poly[j,i] = 2*point[j]*poly[j,i-1] - poly[j,i-2]
      end
    end
  end

  return poly

end

"""
Construct the derivative of a Chebyshev polynomial at 'point'. 'point' can be a scalar or a vector of scalars.

Signature
=========

deriv = cheb_polynomial_deriv(order,point)
"""
function cheb_polynomial_deriv(order::S,point::R) where {S<:Integer,R<:Real}

  poly_deriv = Array{R}(undef,1,order+1)
  poly_deriv[1] = zero(R)
  p = one(R)
  pl = NaN
  pll = NaN

  @inbounds for i = 2:order+1
    if i == 2
      pl, p = p, point
      poly_deriv[i] = one(R)
    else
      pll, pl = pl, p
      p = 2*point*pl - pll
      poly_deriv[i] = 2*pl + 2*point*poly_deriv[i-1] - poly_deriv[i-2]
    end
  end

  return poly_deriv

end

function cheb_polynomial_deriv(order::S,point::AbstractArray{R,1}) where {S<:Integer,R<:Real}

  poly_deriv = Array{R}(undef,order+1,length(point))
  poly_deriv[1, :] .= zeros(R,length(point))

  @inbounds for j in eachindex(point)
    p = one(R)
    pl = NaN
    pll = NaN
    for i = 2:order+1
      if i == 2
        pl, p = p, x[j]
        poly_deriv[i,j] = one(R)
      else
        pll, pl = pl, p
        p = 2*point[j]*pl - pll
        poly_deriv[i,j] = 2*pl+2*point[j]*poly_deriv[i-1,j] - poly_deriv[i-2,j]
      end
    end
  end

  return Matrix(transpose(poly_deriv))

end

"""
Construct the second derivative of a Chebyshev polynomial at 'point'. 'point' can be a scalar or a vector of scalars.

Signature
=========

deriv = cheb_polynomial_sec_deriv(order,point)
"""

function cheb_polynomial_sec_deriv(order::S,point::R) where {S<:Integer,R<:Real}

  poly_sec_deriv = Array{R}(undef, 1, order + 1)
  poly_sec_deriv[1] = zero(R)

  p = one(R)
  pl = NaN
  pll = NaN
  pd = zero(R)
  pdl = NaN
  pdll = NaN

  @inbounds for i = 2:order+1
    if i == 2
      pl, p = p, point
      pdl, pd = pd, one(R)
      poly_sec_deriv[i] = zero(R)
    else
      pll, pl = pl, p
      p = 2*point*pl - pll
      pdll, pdl = pdl, pd
      pd = 2*pl + 2*point*pdl - pdll
      poly_sec_deriv[i] = 2*point*poly_sec_deriv[i-1] + 4*pdl - poly_sec_deriv[i-2]
    end
  end

  return poly_sec_deriv

end

function cheb_polynomial_sec_deriv(order::S,point::AbstractArray{R,1}) where {S<:Integer,R<:Real}

  poly_sec_deriv = Array{R}(undef, length(point), order + 1)
  poly_sec_deriv[:,1] .= zero(R)

  @inbounds for j in eachindex(point)
    p = one(R)
    pl = NaN
    pll = NaN
    pd = zero(R)
    pdl = NaN
    pdll = NaN
    for i = 2:order+1
      if i == 2
        pl, p = p, point[j]
        pdl, pd = pd, one(R)
        poly_sec_deriv[j,i] = zero(R)
      else
        pll, pl = pl, p
        p = 2*point[j]*pl - pll
        pdll, pdl = pdl, pd
        pd = 2*pl + 2*point[j]*pdl - pdll
        poly_sec_deriv[j,i] = 2*point[j]*poly_sec_deriv[j,i-1] + 4*pdl - poly_sec_deriv[j,i-2]
      end
    end
  end

  return poly_sec_deriv

end

"""
Compute the weights of a univariate Chebyshev polynomial with order, 'order', on domain, 'domain'.

Signature
=========

w = cheb_weights(y,nodes,order,domain)
"""
function cheb_weights(y::AbstractArray{T,1},nodes::AbstractArray{T,1},order::S,domain::Array{R,1}) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer}

  normalized_nodes = normalize_node(nodes,domain)
  P = cheb_polynomial(order,normalized_nodes)

  weights = Array{T,1}(undef,order+1)
  for i in CartesianIndices(weights)

    numerator = zero(T)
    denominator = zero(T)

    for s in CartesianIndices(y)

      numerator   += y[s]*P[s,i]
      denominator += P[s,i]^2

    end

    weights[i] = numerator/denominator

  end

  return weights

end

"""
Evaluate a Chebyshev polynomial at point, 'point'.

Signature
=========

yhat = cheb_evaluate(w,point,order,domain)
"""
function cheb_evaluate(w::AbstractArray{T,1},point::Q,order::S,domain::Array{R,1}) where {T<:AbstractFloat,R<:AbstractFloat,Q<:Real,S<:Integer}

  normalized_point = normalize_node(point,domain)
  P = cheb_polynomial(order,normalized_point)

  yhat = zero(T)
  @inbounds for i in CartesianIndices(w)
    yhat += w[i]*P[i]
  end

  return yhat

end

"""
Create a function that evaluates a Chebyshev polynomial at point, 'point'.

f = chebyshev_interp(y,nodes,order,domain)
"""
function chebyshev_interp(y::AbstractArray{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer}

  w = cheb_weights(y,nodes,order,domain)

  function cheb_interp(point)

     yhat = cheb_evaluate(w, point, order, domain)

    return yhat

  end

  return cheb_interp

end

"""
Create a Chebyshev tensor train from a discrete tensor train.

Signatures
==========

trainCTT = TTcreateCTT(g,nodes,domain)
trainCTT = TTcreateCTT(g,order,nodes,domain)
"""
function TTcreateCTT(train::DiscreteTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T<:AbstractFloat,R<:AbstractFloat,d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    order = size(train.cores[i],2)-1

    f[i] = [cheb_weights(train.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return ChebyshevTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateCTT(train::DiscreteTensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    f[i] = [cheb_weights(train.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return ChebyshevTensorTrain(f,train.ranks,train.sweeps)

end

#### Functions for working with Legendre nodes/polynomials

"""
Create a vector of 'n' Legendre nodes on 'domain' with element type 'T'.

Signatures
==========

nodes= legendre_nodes(n)
nodes= legendre_nodes(n,domain)
nodes= legendre_nodes(n,domain,T)
"""
function legendre_nodes(n::S,domain=[1.0,-1.0],T::DataType=Float64) where {S<:Integer}

  λ, Q = eigen(SymTridiagonal(zeros(T,n), [i / sqrt(T(4i^2 - 1)) for i = 1:n-1]))
  nodes = (λ .+ 1) * (domain[1] - domain[2]) / 2 .+ domain[2]

  return nodes

end

"""
Generate Legendre polynomials of 'order' at 'point'. 'point' can be a scalar or a vector of scalars.

Signatures
==========

P = legendre_polynomial(order,point)
"""
function legendre_polynomial(order::S,point::R) where {S<:Integer,R<:Real}

  poly = Array{R}(undef, 1, order + 1)
  poly[1] = one(R)

  @inbounds for i = 2:order+1
    if i == 2
      poly[i] = point
    else
      poly[i] = ((2*i-1)/i)*point*poly[i-1] - ((i-1)/i)*poly[i-2]
    end
  end

  return poly

end

function legendre_polynomial(order::S,point::Array{R,1}) where {S<:Integer,R<:Real}

  poly = Array{R}(undef, length(point), order + 1)
  poly[:, 1] .= ones(R, length(point))

  @inbounds for i = 2:order+1
    for j in eachindex(point)
      if i == 2
        poly[j,i] = point[j]
      else
        poly[j,i] = ((2*i-1)/i)*point[j]*poly[j,i-1] - ((i-1)/i)*poly[j,i-2]
      end
    end
  end

  return poly

end

"""
Construct the derivative of a Legendre polynomial at 'point'. 'point' can be a scalar or a vector of scalars.

Signature
=========

deriv = legendre_polynomial_deriv(order,point)
"""
function legendre_polynomial_deriv(order::S,point::R) where {S<:Integer,R<:Real}

  poly       = Array{R}(undef,1,order+1)
  poly_deriv = Array{R}(undef,1,order+1)
  poly[1] = one(R)
  poly_deriv[1] = zero(R)

  @inbounds for i = 2:order+1
    if i == 2
      poly[i] = point
      poly_deriv[i] = one(R)
    else
      poly[i] = ((2*i-1)/i)*point*poly[i-1] - ((i-1)/i)*poly[i-2]
      poly_deriv[i] = i*(point*poly[i]-poly[i-1])/(point^2-1)
    end
  end

  return poly_deriv

end

function legendre_polynomial_deriv(order::S,point::AbstractArray{R,1}) where {S<:Integer,R<:Real}

  n = length(point)

  poly       = Array{R}(undef,n,order+1)
  poly_deriv = Array{R}(undef,n,order+1)
  poly[:,1] .= one(R)
  poly_deriv[:,1] .= zero(R)

  @inbounds for j = 1:n
    for i = 2:order+1
      if i == 2
        poly[j,i] = point[j]
        poly_deriv[j,i] = one(R)
      else
        poly[j,i] = ((2*i-1)/i)*point[j]*poly[i-1] - ((i-1)/i)*poly[i-2]
        poly_deriv[j,i] = i*(point[j]*poly[i]-poly[i-1])/(point[j]^2-1)
      end
    end
  end

  return poly_deriv

end

"""
Construct the second derivative of a Legendre polynomial at 'point'. ' point can be a scalar or a vector of scalars.

Signature
=========

deriv = legendre_polynomial_deriv(order,point)
"""
function legendre_polynomial_sec_deriv(order::S,point::R) where {S<:Integer,R<:Real}

  poly       = Array{R}(undef,1,order+1)
  poly_deriv = Array{R}(undef,1,order+1)
  poly_sec_deriv = Array{R}(undef,1,order+1)
  poly[1] = one(R)
  poly_deriv[1] = zero(R)
  poly_sec_deriv[1] = zero(R)

  @inbounds for i = 2:order+1
    if i == 2
      poly[i] = point
      poly_deriv[i] = one(R)
      poly_sec_deriv[i] = zero(R)
    else
      poly[i] = ((2*i-1)/i)*point*poly[i-1] - ((i-1)/i)*poly[i-2]
      poly_deriv[i] = i*(point*poly[i]-poly[i-1])/(point^2-1)
      poly_sec_deriv[i] = i*(poly[i]*((point^2-1)^(-1)-2*point^2/(point^2-1)^2)+(point/(point^2-1))*poly_deriv[i]) - i*(poly[i-1]*(-2*point/(point^2-1)^2) + (1/(point^2-1))*poly_deriv[i-1])
    end
  end

  return poly_sec_deriv

end

function legendre_polynomial_sec_deriv(order::S,point::Array{R,1}) where {S<:Integer,R<:Real}

  n = length(point)

  poly       = Array{R}(undef,n,order+1)
  poly_deriv = Array{R}(undef,n,order+1)
  poly_sec_deriv = Array{R}(undef,n,order+1)
  poly[:,1]  .= one(R)
  poly_deriv[:,1] .= zero(R)
  poly_sec_deriv[:,1] .= zero(R)

  @inbounds for j = 1:n
    for i = 2:order+1
      if i == 2
        poly[j,i] = point[j]
        poly_deriv[j,i] = one(R)
        poly_sec_deriv[j,i] = zero(R)
      else
        poly[j,i] = ((2*i-1)/i)*point[j]*poly[j,i-1] - ((i-1)/i)*poly[j,i-2]
        poly_deriv[j,i] = i*(point[j]*poly[j,i]-poly[j,i-1])/(point[j]^2-1)
        poly_sec_deriv[j,i] = i*(poly[j,i]*((point[j]^2-1)^(-1)-2*point[j]^2/(point[j]^2-1)^2)+(point[j]/(point[j]^2-1))*poly_deriv[j,i]) - i*(poly[j,i-1]*(-2*point[j]/(point[j]^2-1)^2) + (1/(point[j]^2-1))*poly_deriv[j,i-1])
      end
    end
  end

  return poly_sec_deriv

end

"""
Compute the weights of a univariate Legendre polynomial with order, 'order', on domain, 'domain'.

Signature
=========

w = legendre_weights(y,nodes,order,domain)
"""
function legendre_weights(y::AbstractArray{T,1},nodes::AbstractArray{T,1},order::S,domain::Array{R,1}) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer}

  normalized_nodes = normalize_node(nodes,domain)
  P = legendre_polynomial(order,normalized_nodes)

  weights = (P'P)\(P'y)

  return weights

end

"""
Evaluate a Legendre polynomial at point, 'point'.

Signature
=========

yhat = legendre_evaluate(w,point,order,domain)
"""
function legendre_evaluate(w::AbstractArray{T,1},point::Q,order::S,domain::Array{R,1}) where {T<:AbstractFloat,R<:AbstractFloat,Q<:Real,S<:Integer}

  normalized_point = normalize_node(point,domain)
  P = legendre_polynomial(order,normalized_point)

  yhat = zero(T)
  @inbounds for i in CartesianIndices(w)
    yhat += w[i]*P[i]
  end

  return yhat

end

"""
Create a function that evaluates a Legendre polynomial at point, 'point'.

Signature
=========

f = legendre_interp(y,nodes,order,domain)
"""
function legendre_interp(y::AbstractArray{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer}

  w = legendre_weights(y,nodes,order,domain)

  function legend_interp(point)

    yhat = legendre_evaluate(w, point, order, domain)
  
    return yhat
    
  end

  return legend_interp

end

"""
Create a Legendre tensor train from a discrete tensor train.

Signatures
==========

trainLTT = TTcreateLTT(g,order,nodes,domain)
"""
function TTcreateLTT(train::DiscreteTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T<:AbstractFloat,R<:AbstractFloat,d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    order = size(train.cores[i],2)-1

    f[i] = [legendre_weights(train.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return LegendreTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateLTT(train::DiscreteTensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    f[i] = [legendre_weights(train.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return LegendreTensorTrain(f,train.ranks,train.sweeps)

end

#### Functions for working with piecewise linear nodes

"""
Compute 'n' points uniformly placed on domain, 'domain', with eltype(T)

Signatures
==========

nodes = piecewise_linear_nodes(n)
nodes = piecewise_linear_nodes(n,domain)
nodes = piecewise_linear_nodes(n,domain,T)
"""
function piecewise_linear_nodes(n::S,domain = [1.0,-1.0],T::DataType=Float64) where {S<:Integer}

  if n <= 0
    error("The number of nodes must be positive.")
  end

  nodes = [T(domain[1]+domain[2])/2.0 for _ in 1:n]

  if isodd(n)
    inc = T(domain[1]-domain[2])/(n-1)
  else
    inc = T(domain[1]-domain[2])/n
  end
  @inbounds for i = 1:div(n,2)
    nodes[i]     += (i-1-div(n,2))*inc
    nodes[n-i+1] -= (i-1-div(n,2))*inc
  end

  return nodes

end

"""
Finds the indexes for the elements of 'x' that immediately bracket 'point'.

Signature
=========

(l,u) = bracket_nodes(x,point)
"""
function bracket_nodes(nodes::Array{T,1},point::R) where {T<:AbstractFloat,R<:Real} # not exported

  n = length(nodes)

  if real(point) <= nodes[1] # Real is used because complex numbers are occasionally used in NLboxsolve.jl
    return (1,2)
  elseif real(point) >= nodes[end]
    return (n-1,n)
  else
    y = 0
    for i in nodes
      if i < real(point)
        y += 1
      else
        break
      end
    end
    return (y,y+1)
  end

end

function piecewise_linear_weight(nodes::Array{T,1},point::R) where {T<:AbstractFloat,R<:Real} # not exported

  bracketing_nodes = bracket_nodes(nodes,point)

  weight = (point-nodes[bracketing_nodes[1]])/(nodes[bracketing_nodes[2]]-nodes[bracketing_nodes[1]])

  return weight

end

"""
Univariate piecewise linear evaluation at point, 'point'.

Signature
=========

yhat = piecewise_linear_evaluate(y,nodes,point)
"""

function piecewise_linear_evaluate(y::AbstractArray{T1,1},nodes::AbstractArray{T2,1},point::R) where {T1<:AbstractFloat,T2<:AbstractFloat,R<:Real}

  b = bracket_nodes(nodes,point)
  w = piecewise_linear_weight(nodes,point)

  y_estimate = y[b[1]] + w*(y[b[2]]-y[b[1]])

  return y_estimate

end

"""
Create a function that evaluates a piecewise linear approximation at point, 'point'.

Signature
=========

f = piecewise_linear_evaluate(y,nodes)
"""
function piecewise_linear_evaluate(y::AbstractArray{T1,1},nodes::AbstractArray{T2,1}) where {T1<:AbstractFloat,T2<:AbstractFloat}

function approximating_function(point::R) where {R<:Real}

    return piecewise_linear_evaluate(y,nodes,point)

end

return approximating_function

end

"""
Create a piecewise linear tensor train from a discrete tensor train.

Signature
=========

trainPTT = TTcreatePTT(g)
"""
function TTcreatePTT(train::DiscreteTensorTrain) # Creates a piecewise linear tensor train from a tensor train

  return PiecewiseTensorTrain(train.cores,train.ranks,train.sweeps)

end

#### Functions for creating functional tensor trains

"""
Create a functional tensor train.

Signatures
==========

trainFTT = TTcreateFTT(train,nodes,domain)       # train is a discrete tensor train, nodes are Chebyshev roots
trainFTT = TTcreateFTT(train,order,nodes,domain) # train is a discrete tensor train, nodes are Chebyshev roots
trainFTT = TTcreateFTT(train,domain)             # train is a ChebyshevTensorTrain or LegendreTensorTrain
trainFTT = TTcreateFTT(train,order,domain)       # train is a ChebyshevTensorTrain or LegendreTensorTrain
trainFTT = TTcreateFTT(train,nodes)              # train is a PiecewiseTensorTrain
"""
function TTcreateFTT(train::DiscreteTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T<:AbstractFloat,R<:AbstractFloat,d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    order = size(train.cores[i],2)-1

    f[i] = [chebyshev_interp(train.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateFTT(train::DiscreteTensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T<:AbstractFloat,R<:AbstractFloat,S<:Integer,d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    f[i] = [chebyshev_interp(train.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateFTT(train::ChebyshevTensorTrain,domain::Array{T,2}) where {T<:AbstractFloat}

  d = length(train.cores)

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    order = length(train.cores[i][1])-1

    f[i] = [interp(x) = cheb_evaluate(train.cores[i][j,k],x,order,domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateFTT(train::ChebyshevTensorTrain,order::NTuple{d,S},domain::Array{T,2}) where {T<:AbstractFloat,S<:Integer,d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    f[i] = [interp(x) = cheb_evaluate(train.cores[i][j,k],x,order[i],domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateFTT(train::LegendreTensorTrain,domain::Array{T,2}) where {T<:AbstractFloat}

  d = length(train.cores)

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    order = length(train.cores[i][1])-1

    f[i] = [interp(x) = legendre_evaluate(train.cores[i][j,k],x,order,domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateFTT(train::LegendreTensorTrain,order::NTuple{d,S},domain::Array{T,2}) where {T<:AbstractFloat,S<:Integer,d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    f[i] = [interp(x) = legendre_evaluate(train.cores[i][j,k],x,order[i],domain[:,i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,train.ranks,train.sweeps)

end

function TTcreateFTT(train::PiecewiseTensorTrain,nodes::NTuple{d,Array{T,1}}) where {T<:AbstractFloat,d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    f[i] = [piecewise_linear_evaluate(train.cores[i][j,:,k],nodes[i]) for j = 1:train.ranks[i], k = 1:train.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,train.ranks,train.sweeps)

end

"""
Create an interpolating function from a functional tensor train.

Signature
=========

f = TTinterp(train)
"""
function TTinterp(train::FunctionalTensorTrain)

  function TTinterp(point::Array{T,1}) where {T<:Real}

    d = length(train.cores)

    A = reshape([train.cores[1][k](point[1]) for k in eachindex(train.cores[1])],train.ranks[1],train.ranks[2])
    for j = 2:d
      A *= reshape([train.cores[j][k](point[j]) for k in eachindex(train.cores[j])],train.ranks[j],train.ranks[j+1])
    end

    return A[1]

  end

  return TTinterp

end

#### Functions for evaluating discrete/continuous tensor trains

"""
Recreates the dense array associated with a discrete tensor train.

Signature
=========

A = TTdecompress(train)
"""
function TTdecompress(train::DiscreteTensorTrain)

  d = length(train.cores)
  n = ntuple(i -> size(train.cores[i], 2), d)

  A = Array{eltype(train.cores[1]),d}(undef, n)

  @views for i in CartesianIndices(A)
    temp = train.cores[1][:,i[1],:]
    for j = 2:d
      temp *= train.cores[j][:,i[j],:]
    end
    A[i] = temp[1]
  end

  return A

end

"""
Recreates the dense array associated with a discrete tensor train.

Signature
=========

A = TTdecompress_threaded(train)
"""
function TTdecompress_threaded(train::DiscreteTensorTrain)

  d = length(train.cores)
  n = ntuple(i -> size(train.cores[i], 2), d)

  A = Array{eltype(train.cores[1]),d}(undef, n)

  @views Threads.@threads for i in CartesianIndices(A)
    temp = train.cores[1][:,i[1],:]
    for j = 2:d
      temp *= train.cores[j][:,i[j],:]
    end
    A[i] = temp[1]
  end

  return A

end

"""
Evaluate a discrete tensor train at an index of the dense array or a functional tensor train at a point.

Signatures
==========

yhat = TTevaluate(train,index)
yhat = TTevaluate(train,point)
"""
function TTevaluate(train::DiscreteTensorTrain,index::Array{S,1}) where {S<:Integer}

  d = length(train.cores)

  A = @views train.cores[1][:,index[1],:]
  for j = 2:d
    A *= @views train.cores[j][:,index[j],:]
  end

  return A[1]

end

function TTevaluate(train::DiscreteTensorTrain,index::NTuple{d,S}) where {S<:Integer,d}

  A = @views train.cores[1][:,index[1],:]
  for j = 2:d
    A *= @views train.cores[j][:,index[j],:]
  end

  return A[1]

end

function TTevaluate(train::DiscreteTensorTrain,index::Array{Array{S,1},1}) where {S<:Integer}

  T = eltype(train.cores[1])
  d = length(train.cores)

  N = length(index)

  y = Array{T}(undef,N)

  for i in 1:N
    A = @views train.cores[1][:,index[i][1],:]
    for j = 2:d
      A *= @views train.cores[j][:,index[i][j],:]
    end

    y[i] = A[1]

  end

  return y

end

function TTevaluate(train::DiscreteTensorTrain,index::Array{NTuple{d,S},1}) where {S<:Integer,d}

  T = eltype(train.cores[1])

  N = length(index)

  y = Array{T}(undef,N)

  for i in 1:N
    A = @views train.cores[1][:,index[i][1],:]
    for j = 2:d
      A *= @views train.cores[j][:,index[i][j],:]
    end

    y[i] = A[1]

  end

  return y

end

function TTevaluate(train::FunctionalTensorTrain,point::Array{T,1}) where {T<:Real}

  d = length(train.cores)

  A = reshape([train.cores[1][k](point[1]) for k in eachindex(train.cores[1])],train.ranks[1],train.ranks[2])
  for j = 2:d
    A *= reshape([train.cores[j][k](point[j]) for k in eachindex(train.cores[j])],train.ranks[j],train.ranks[j+1])
  end

  return A[1]

end

function TTevaluate(train::FunctionalTensorTrain,points::Array{Array{T,1}}) where {T<:Real}

  d = length(train.cores)

  N = length(points)

  y = zeros(N)

  for i in 1:N
    A = reshape([train.cores[1][k](points[i][1]) for k in eachindex(train.cores[1])],train.ranks[1],train.ranks[2])
    for j = 2:d
      A *= reshape([train.cores[j][k](points[i][j]) for k in eachindex(train.cores[j])],train.ranks[j],train.ranks[j+1])
    end

    y[i] = A[1]

  end

  return y

end

#### Functions to compute derivatives and gradients of continuous tensor trains

"""
Compute the derivative of a ChebyshevTensorTrain or LegendreTensorTrain with respect to one or two variables.

Signatures
==========

deriv = TTderivative(train,x,pos,domain)
deriv = TTderivative(train,x,pos1,pos2,domain)
"""
function TTderivative(train::ChebyshevTensorTrain,x::AbstractArray{R,1},pos::S,domain::Array{T,2}) where {R<:Real,T<:AbstractFloat,S<:Integer}

  d = length(train.cores)
  poly = Array{Array{R,2},1}(undef, d)
  @inbounds for i = 1:d
    order = length(train.cores[i][1])-1
    if i === pos
      poly[i] = cheb_polynomial_deriv(order, normalize_node(x[i], domain[:, i]))
    else
      poly[i] = cheb_polynomial(order, normalize_node(x[i], domain[:, i]))
    end
  end

  A = [(poly[1]*train.cores[1][i,k])[1] for i in 1:train.ranks[1], k in 1:train.ranks[2]]
  @inbounds for j = 2:d
    A *= [(poly[j]*train.cores[j][i,k])[1] for i in 1:train.ranks[j], k in 1:train.ranks[j+1]]
  end

  return A[1][1] * (2.0 / (domain[1, pos] - domain[2, pos]))

end

function TTderivative(train::LegendreTensorTrain,x::AbstractArray{R,1},pos::S,domain::Array{T,2}) where {R<:Real,T<:AbstractFloat,S<:Integer}

  d = length(train.cores)
  poly = Array{Array{R,2},1}(undef, d)
  @inbounds for i = 1:d
    order = length(train.cores[i][1])-1
    if i === pos
      poly[i] = legendre_polynomial_deriv(order, normalize_node(x[i], domain[:, i]))
    else
      poly[i] = legendre_polynomial(order, normalize_node(x[i], domain[:, i]))
    end
  end

  A = [(poly[1]*train.cores[1][i,k])[1] for i in 1:train.ranks[1], k in 1:train.ranks[2]]
  @inbounds for j = 2:d
    A *= [(poly[j]*train.cores[j][i,k])[1] for i in 1:train.ranks[j], k in 1:train.ranks[j+1]]
  end

  return A[1][1] * (2.0 / (domain[1, pos] - domain[2, pos]))

end

function TTderivative(train::ChebyshevTensorTrain,x::AbstractArray{R,1},pos1::S,pos2::S,domain::Array{T,2}) where {R<:Real,T<:AbstractFloat,S<:Integer}

  d = length(train.cores)
  poly = Array{Array{R,2},1}(undef, d)
  @inbounds for i = 1:d
    order = length(train.cores[i][1])-1
    if i === pos1 && i === pos2
      poly[i] = cheb_polynomial_sec_deriv(order, normalize_node(x[i], domain[:, i]))
    elseif i === pos1 || i === pos2
      poly[i] = cheb_polynomial_deriv(order, normalize_node(x[i], domain[:, i]))
    else
      poly[i] = cheb_polynomial(order, normalize_node(x[i], domain[:, i]))
    end
  end

  A = [(poly[1]*train.cores[1][i,k])[1] for i in 1:train.ranks[1], k in 1:train.ranks[2]]
  @inbounds for j = 2:d
    A *= [(poly[j]*train.cores[j][i,k])[1] for i in 1:train.ranks[j], k in 1:train.ranks[j+1]]
  end

  return A[1][1] * (2.0 / (domain[1, pos1] - domain[2, pos1])) * (2.0 / (domain[1, pos2] - domain[2, pos2]))

end

function TTderivative(train::LegendreTensorTrain,x::AbstractArray{R,1},pos1::S,pos2::S,domain::Array{T,2}) where {R<:Real,T<:AbstractFloat,S<:Integer}

  d = length(train.cores)
  poly = Array{Array{R,2},1}(undef, d)
  @inbounds for i = 1:d
    order = length(train.cores[i][1])-1
    if i === pos1 && i === pos2
      poly[i] = legendre_polynomial_sec_deriv(order, normalize_node(x[i], domain[:, i]))
    elseif i === pos1 || i === pos2
      poly[i] = legendre_polynomial_deriv(order, normalize_node(x[i], domain[:, i]))
    else
      poly[i] = legendre_polynomial(order, normalize_node(x[i], domain[:, i]))
    end
  end

  A = [(poly[1]*train.cores[1][i,k])[1] for i in 1:train.ranks[1], k in 1:train.ranks[2]]
  @inbounds for j = 2:d
    A *= [(poly[j]*train.cores[j][i,k])[1] for i in 1:train.ranks[j], k in 1:train.ranks[j+1]]
  end

  return A[1][1] * (2.0 / (domain[1, pos1] - domain[2, pos1])) * (2.0 / (domain[1, pos2] - domain[2, pos2]))

end

"""
Compute the gradient of a tensor train.

Signature
=========

grad = TTgradient(train,x,domain) # ChebyshevTensorTrain or LegendreTensorTrain
grad = TTgradient(train,nodes,x) # DiscreteTensorTrain
"""
function TTgradient(train::ChebyshevTensorTrain,x::AbstractArray{R,1},domain::Array{T,2}) where {T<:AbstractFloat,R<:Real}

  d = length(train.cores)

  gradient = Array{R,2}(undef,1,d)

  @inbounds for i = 1:d
    gradient[i] = TTderivative(train,x,i,domain)
  end

  return gradient

end

function TTgradient(train::LegendreTensorTrain,x::AbstractArray{R,1},domain::Array{T,2}) where {T<:AbstractFloat,R<:Real}

  d = length(train.cores)

  gradient = Array{R,2}(undef,1,d)

  @inbounds for i = 1:d
    gradient[i] = TTderivative(train,x,i,domain)
  end

  return gradient

end

"""
Compute the Hessian of a tensor train.

Signature
=========

hess = TThessian(train,x,domain) # ChebyshevTensorTrain or LegendreTensorTrain
hess = TThessian(train,nodes,x) # DiscreteTensorTrain
"""
function TThessian(train::ChebyshevTensorTrain,x::AbstractArray{R,1},domain::Array{T,2}) where {T<:AbstractFloat,R<:Real}

  d = length(train.cores)

  hessian = Array{R,2}(undef,d,d)

  @inbounds for i = 1:d
    @inbounds for j = 1:d
        hessian[i,j] = TTderivative(train,x,i,j,domain)
    end
  end

  return hessian

end

function TThessian(train::LegendreTensorTrain,x::AbstractArray{R,1},domain::Array{T,2}) where {T<:AbstractFloat,R<:Real}

  d = length(train.cores)

  hessian = Array{R,2}(undef,d,d)

  @inbounds for i = 1:d
    @inbounds for j = 1:d
        hessian[i,j] = TTderivative(train,x,i,j,domain)
    end
  end

  return hessian

end

#### Functions to integrate discrete tensor trains and functional tensor trains

"""
Integrate a discrete tensor train using Gauss-Chebyshev quadrature.

Signatures
==========

area = TTintegrate_GC(train,domain) # Integrates over all dimensions
area = TTintegrate_GC(train,domain,μ) # Integrates over dimensions 1 to μ
"""
function TTintegrate_GC(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R<:AbstractFloat} # Integrates over all dimensions

  T = eltype(train.cores[1])
  d = length(train.cores)

  integral = fill(T(1.0),1,1)
  for i = 1:d
    n = size(train.cores[i])
    term = zeros(n[1],n[3])
    nodes, weights = chebyshev(n[2])
    for j = 1:n[2]
      term += train.cores[i][:,j,:]*(1.0-nodes[j]^2.0)^(0.5)*weights[j]
    end
    integral *= term*((domain[1,i]-domain[2,i])/2)
  end

  return integral[1]

end

function TTintegrate_GC(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}},μ::S) where {R<:AbstractFloat,S<:Integer} # Integrates over dimensions 1 to μ

  d = length(train.cores)

  if μ > d
    error("Cannot integrate over more than $d dimensions.")
  elseif μ < 0
    error("Cannot integrate over negative dimensions.")
  elseif μ == 0
    return train
  elseif μ == d
    return TTintegrate_GC(train,domain)
  else # μ ∈ (1,d-1)
    T = eltype(train.cores[1])
  
    g = copy(train.cores[μ+1:end])
    r = copy(train.ranks[μ+1:end])

    integral = fill(T(1.0),1,1)
    for i = 1:μ
      n = size(train.cores[i])
      term = zeros(n[1],n[3])
      nodes, weights = chebyshev(n[2])
      for j = 1:n[2]
        term += train.cores[i][:,j,:]*(1.0-nodes[j]^2.0)^(0.5)*weights[j]
      end
      integral *= term*((domain[1,i]-domain[2,i])/2)
    end

    g[1] = times_dim_1(integral,g[1])
    r[1] = 1

    return BaseTensorTrain(g,r,0)
  end

end

"""
Integrate a discrete tensor train using Gauss-Legendre quadrature.

Signatures
==========

area = TTintegrate_GL(train,domain) # Integrates over all dimensions
area = TTintegrate_GL(train,domain,μ) # Integrates over dimensions 1 to μ
"""
function TTintegrate_GL(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R<:AbstractFloat} # Integrates over all dimensions

  T = eltype(train.cores[1])
  d = length(train.cores)

  integral = fill(T(1.0),1,1)
  for i = 1:d
    n = size(train.cores[i])
    term = zeros(n[1],n[3])
    nodes, weights = legendre(n[2])
    for j = 1:n[2]
      term += train.cores[i][:,j,:]*weights[j]
    end
    integral *= term*((domain[1,i]-domain[2,i])/2)
  end

  return integral[1]

end

function TTintegrate_GL(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}},μ::S) where {R<:AbstractFloat,S<:Integer} # Integrates over dimensions 1 to μ

  d = length(train.cores)

  if μ > d
    error("Cannot integrate over more than $d dimensions.")
  elseif μ < 0
    error("Cannot integrate over negative dimensions.")
  elseif μ == 0
    return train
  elseif μ == d
    return TTintegrate_GL(train,domain)
  else # μ ∈ (1,d-1)
    T = eltype(train.cores[1])
  
    g = copy(train.cores[μ+1:end])
    r = copy(train.ranks[μ+1:end])

    integral = fill(T(1.0),1,1)
    for i = 1:μ
      n = size(train.cores[i])
      term = zeros(n[1],n[3])
      nodes, weights = legendre(n[2])
      for j = 1:n[2]
        term += train.cores[i][:,j,:]*weights[j]
      end
      integral *= term*((domain[1,i]-domain[2,i])/2)
    end

    g[1] = times_dim_1(integral,g[1])
    r[1] = 1

    return BaseTensorTrain(g,r,0)
  end

end

"""
Integrate a discrete tensor train using Gauss-Hermite quadrature.

Signatures
==========

area = TTintegrate_GH(train) # Integrates over all dimensions
area = TTintegrate_GH(train,μ) # Integrates over dimensions 1 to μ
"""
function TTintegrate_GH(train::DiscreteTensorTrain) # Integrates over all dimensions

  T = eltype(train.cores[1])
  d = length(train.cores)

  integral = fill(T(1.0),1,1)
  for i = 1:d
    n = size(train.cores[i])
    term = zeros(n[1],n[3])
    nodes, weights = hermite(n[2])
    for j = 1:n[2]
      term += train.cores[i][:,j,:]*exp(nodes[j]^2.0)*weights[j]
    end
    integral *= term
  end

  return integral[1]

end

function TTintegrate_GH(train::DiscreteTensorTrain,μ::S) where {S<:Integer} # Integrates over dimensions 1 to μ

  d = length(train.cores)

  if μ > d
    error("Cannot integrate over more than $d dimensions.")
  elseif μ < 0
    error("Cannot integrate over negative dimensions.")
  elseif μ == 0
    return train
  elseif μ == d
    return TTintegrate_GH(train)
  else # μ ∈ (1,d-1)
    T = eltype(train.cores[1])
  
    g = copy(train.cores[μ+1:end])
    r = copy(train.ranks[μ+1:end])

    integral = fill(T(1.0),1,1)
    for i = 1:μ
      n = size(train.cores[i])
      term = zeros(n[1],n[3])
      nodes, weights = hermite(n[2])
      for j = 1:n[2]
        term += train.cores[i][:,j,:]*exp(nodes[j]^2.0)*weights[j]
      end
      integral *= term
    end

    g[1] = times_dim_1(integral,g[1])
    r[1] = 1

    return BaseTensorTrain(g,r,0)
  end

end

"""
Integrate a discrete tensor train using the trapazoidal method.

Signatures
==========

area = TTintegrate_PL(train,domain) # Integrates over all dimensions
area = TTintegrate_PL(train,domain,μ) # Integrates over dimensions 1 to μ
"""
function TTintegrate_PL(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R<:AbstractFloat} # Integrates over all dimensions

  T = eltype(train.cores[1])
  d = length(train.cores)

  integral = fill(T(1.0),1,1)
  for i = 1:d
    term = trapazoidal(train.cores[i],domain[:,i])
    integral *= term
  end

  return integral[1]

end

function TTintegrate_PL(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}},μ::S) where {R<:AbstractFloat,S<:Integer} # Integrates over dimensions 1 to μ

  d = length(train.cores)

  if μ > d
    error("Cannot integrate over more than $d dimensions.")
  elseif μ < 0
    error("Cannot integrate over negative dimensions.")
  elseif μ == 0
    return train
  elseif μ == d
   return TTintegrate_PL(train,domain)
  else # μ ∈ (1,d-1)
    T = eltype(train.cores[1])

    g = copy(train.cores[μ+1:end])
    r = copy(train.ranks[μ+1:end])

    integral = fill(T(1.0),1,1)
    for i = 1:d
      term = trapazoidal(train.cores[i],domain[:,i])
      integral *= term
    end

    g[1] = times_dim_1(integral,g[1])
    r[1] = 1

    return BaseTensorTrain(g,r,0)
  end

end

"""
Integrate a functional tensor train over all dimensions using Gauss-Chebyshev quadrature.

Signature
=========

area = TTintegrate_GC(train,nodes,domain)
"""
function TTintegrate_GC(train::FunctionalTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Union{Array{R,2},Array{R,1}}) where {T<:AbstractFloat,R<:AbstractFloat,d} # Integrates over all dimensions

  n = length.(nodes)

  integral = fill(T(1.0),1,1) 
  for i = 1:d
    node, weights = chebyshev(n[i])
    F = zeros(train.ranks[i],train.ranks[i+1])
    for j = 1:n[i]
      F += reshape([train.cores[i][k]([nodes[i][j]]) for k in eachindex(train.cores[i])]*(1.0-node[j]^2.0)^(0.5)*weights[j],train.ranks[i],train.ranks[i+1])
    end
    integral *= F*((domain[1,i]-domain[2,i])/2)
  end

  return integral[1]

end

"""
Integrate a functional tensor train over all dimensions using Gauss-Legendre quadrature.

Signature
=========

area = TTintegrate_GL(train,nodes,domain)
"""
function TTintegrate_LC(train::FunctionalTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Union{Array{R,2},Array{R,1}}) where {T<:AbstractFloat,R<:AbstractFloat,d} # Integrates over all dimensions

  n = length.(nodes)

  integral = fill(T(1.0),1,1) 
  for i = 1:d
    node, weights = legendre(n[i])
    F = zeros(train.ranks[i],train.ranks[i+1])
    for j = 1:n[i]
      F += reshape([train.cores[i][k]([nodes[i][j]]) for k in eachindex(train.cores[i])]*weights[j],train.ranks[i],train.ranks[i+1])
    end
    integral *= F*((domain[1,i]-domain[2,i])/2)
  end

  return integral[1]

end

"""
Integrate a functional tensor train over all dimensions using Gauss-Hermite quadrature.

Signature
=========

area = TTintegrate_GH(train,nodes)
"""
function TTintegrate_GH(train::FunctionalTensorTrain,nodes::NTuple{d,Array{T,1}}) where {T<:AbstractFloat,d} # Integrates over all dimensions

  n = length.(nodes)

  integral = fill(T(1.0),1,1) 
  for i = 1:d
    node, weights = hermite(n[i])
    F = zeros(train.ranks[i],train.ranks[i+1])
    for j = 1:n[i]
      F += reshape([train.cores[i][k]([nodes[i][j]]) for k in eachindex(train.cores[i])]*exp(node[j]^2.0)*weights[j],train.ranks[i],train.ranks[i+1])
    end
    integral *= F
  end

  return integral[1]

end

#### Functions to integrate a discrete tensor train over all dimensions except 'ind'.

"""
Integrate a discrete tensor train over all dimensions except 'ind' using Gauss-Chebyshev quadrature.

Signature
=========

margin = TTcompute_marginal_GC(train,domain,ind)
"""
function TTcompute_marginal_GC(train::DiscreteTensorTrain,domain::Array{T,2},ind::S) where {T<:AbstractFloat,S<:Integer}

  d = length(train.cores)

  if ind == 1
    integral = reshape(train.cores[1], train.ranks[1], size(train.cores[1], 2), train.ranks[2])
    for i = 2:d
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * (1.0 - nodes[j]^2.0)^(0.5) * weights[j]
      end
      integral = times_dim_3(integral, term) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  elseif ind == d
    integral = reshape(train.cores[d], train.ranks[d], size(train.cores[d], 2), train.ranks[d+1])
    for i = (d-1):-1:1
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * (1.0 - nodes[j]^2.0)^(0.5) * weights[j]
      end
      integral = times_dim_1(term, integral) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  else
    integral = reshape(train.cores[ind], train.ranks[ind], size(train.cores[ind], 2), train.ranks[ind+1])
    for i = ind+1:d
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * (1.0 - nodes[j]^2.0)^(0.5) * weights[j]
      end
      integral = times_dim_3(integral, term) * ((domain[1, i] - domain[2, i]) / 2)
    end
    for i = ind-1:-1:1
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * (1.0 - nodes[j]^2.0)^(0.5) * weights[j]
      end
      integral = times_dim_1(term, integral) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  end

end

"""
Integrate a discrete tensor train over all dimensions except 'ind' using Gauss-Legendre quadrature.

Signature
=========

margin = TTcompute_marginal_GL(train,domain,ind)
"""
function TTcompute_marginal_GL(train::DiscreteTensorTrain,domain::Array{T,2},ind::S) where {T<:AbstractFloat,S<:Integer}

  d = length(train.cores)

  if ind == 1
    integral = reshape(train.cores[1], train.ranks[1], size(train.cores[1], 2), train.ranks[2])
    for i = 2:d
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * weights[j]
      end
      integral = times_dim_3(integral, term) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  elseif ind == d
    integral = reshape(train.cores[d], train.ranks[d], size(train.cores[d], 2), train.ranks[d+1])
    for i = (d-1):-1:1
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * weights[j]
      end
      integral = times_dim_1(term, integral) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  else
    integral = reshape(train.cores[ind], train.ranks[ind], size(train.cores[ind], 2), train.ranks[ind+1])
    for i = ind+1:d
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * weights[j]
      end
      integral = times_dim_3(integral, term) * ((domain[1, i] - domain[2, i]) / 2)
    end
    for i = ind-1:-1:1
      n = size(train.cores[i])
      nodes, weights = chebyshev(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :] * weights[j]
      end
      integral = times_dim_1(term, integral) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  end

end

"""
Integrate a discrete tensor train over all dimensions except 'ind' using Gauss-Hermite quadrature.

Signature
=========

margin = TTcompute_marginal_GH(train,ind)
"""
function TTcompute_marginal_GH(train::DiscreteTensorTrain,ind::S) where {S<:Integer} # Assumes Gauss-Hermite quadrature

  d = length(train.cores)

  if ind == 1
    integral = reshape(train.cores[1], train.ranks[1], size(train.cores[1], 2), train.ranks[2])
    for i = 2:d
      n = size(train.cores[i])
      nodes, weights = hermite(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :]*exp(nodes[j]^2.0)*weights[j]
      end
      integral = times_dim_3(integral, term)
    end
    return integral[1, :, 1]
  elseif ind == d
    integral = reshape(train.cores[d], train.ranks[d], size(train.cores[d], 2), train.ranks[d+1])
    for i = (d-1):-1:1
      n = size(train.cores[i])
      nodes, weights = hermite(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :]*exp(nodes[j]^2.0)*weights[j]
      end
      integral = times_dim_1(term, integral)
    end
    return integral[1, :, 1]
  else
    integral = reshape(train.cores[ind], train.ranks[ind], size(train.cores[ind], 2), train.ranks[ind+1])
    for i = ind+1:d
      n = size(train.cores[i])
      nodes, weights = hermite(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :]*exp(nodes[j]^2.0)*weights[j]
      end
      integral = times_dim_3(integral, term)
    end
    for i = ind-1:-1:1
      n = size(train.cores[i])
      nodes, weights = hermite(n[2])
      term = zeros(n[1], n[3])
      for j = 1:n[2]
        @views term += train.cores[i][:, j, :]*exp(nodes[j]^2.0)*weights[j]
      end
      integral = times_dim_1(term, integral)
    end
    return integral[1, :, 1]
  end

end

"""
Integrate a discrete tensor train over all dimensions except 'ind' using the trapazoidal method.

Signature
=========

margin = TTcompute_marginal_PL(train,domain,ind)
"""
function TTcompute_marginal_PL(train::DiscreteTensorTrain,domain::Array{T,2},ind::S) where {T<:AbstractFloat,S<:Integer} # Assumes trapazoidal integration 

  d = length(train.cores)

  if ind == 1
    integral = reshape(train.cores[1], train.ranks[1], size(train.cores[1], 2), train.ranks[2])
    for i = 2:d
      term = trapazoidal(train.cores[i], domain[:, i])
      integral = times_dim_3(integral, term)
    end
    return integral[1, :, 1]
  elseif ind == d
    integral = reshape(train.cores[d], train.ranks[d], size(train.cores[d], 2), train.ranks[d+1])
    for i = (d-1):-1:1
      term = trapazoidal(train.cores[i], domain[:, i])
      integral = times_dim_1(term, integral) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  else
    integral = reshape(train.cores[ind], train.ranks[ind], size(train.cores[ind], 2), train.ranks[ind+1])
    for i = ind+1:d
      term = trapazoidal(train.cores[i], domain[:, i])
      integral = times_dim_3(integral, term) * ((domain[1, i] - domain[2, i]) / 2)
    end
    for i = ind-1:-1:1
      term = trapazoidal(train.cores[i], domain[:, i])
      integral = times_dim_1(term, integral) * ((domain[1, i] - domain[2, i]) / 2)
    end
    return integral[1, :, 1]
  end

end

#### Functions to algebraically manipulate discrete tensor trains

"""
Perform rounding on a discrete tensor train to find a more compact discrete tensor train representation.

Signature
=========

t = TTrounding(train,tol)
"""
function TTrounding(train::BaseTensorTrain,tol::T) where {T<:AbstractFloat}

  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple(size(train.cores[i])[2] for i in 1:d)

  G = deepcopy(train.cores)

  r_new = copy(r)

  for i = 1:d-1

    A = reshape(G[i],r_new[i]*n[i],r_new[i+1])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r_new[i+1] = tsvd(A,δ)
    G[i] = reshape(u,r_new[i],n[i],r_new[i+1])
    G[i+1] = times_dim_1(Matrix(transpose(v*Diagonal(s))),G[i+1])

  end

  return BaseTensorTrain(G,r_new,train.sweeps)

end

function TTrounding(train::ExtendedTensorTrain,tol::T) where {T<:AbstractFloat}

  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple(size(train.cores[i])[2] for i in 1:d)

  G = deepcopy(train.cores)

  r_new = copy(r)

  for i = 1:d-1

    A = reshape(G[i],r_new[i]*n[i],r_new[i+1])
    δ = (tol/sqrt(d-1))*norm(A)
    u,s,v,r_new[i+1] = tsvd(A,δ)
    G[i] = reshape(u,r_new[i],n[i],r_new[i+1])
    G[i+1] = times_dim_1(Matrix(transpose(v*Diagonal(s))),G[i+1])

  end

  return ExtendedTensorTrain(G,r_new,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Multiply a discrete tensor train and a scalar.

Signatures
==========

t = TTmult(train,s)
t = TTmult(s,train)
"""
function TTmult(train::DiscreteTensorTrain,s::T) where {T<:Real}

  G = deepcopy(train.cores)
  G[1] = G[1]*s

  return BaseTensorTrain(G,train.ranks,train.sweeps)

end

function TTmult(s::T,train::DiscreteTensorTrain) where {T<:Real}

  return TTmult(train,s)

end

"""
Add two discrete tensor trains.  When the trains have different lengths, anchor determines whether the trains align from the first core (default) or the last core.

Signatures
==========

t = TTadd(traina,trainb)
t = TTadd(traina,trainb,anchor)
"""
function TTadd(traina::DiscreteTensorTrain,trainb::DiscreteTensorTrain,anchor="start")

  Z1 = eltype(traina.cores[1])
  Z2 = eltype(trainb.cores[1])

  Z = promote_type(Z1,Z2)

  da = length(traina.cores)
  db = length(trainb.cores)

  if da > db # traina has more cores than trainb
    n = Tuple(size(traina.cores[i])[2] for i in 1:da)
    filler = TTconstant(one(Z),Tuple(n[db+1:end]),1)
    if anchor == "end"
      G = [filler.cores;trainb.cores]
    elseif anchor == "start"
      G = [trainb.cores;filler.cores]
    else
      error()
    end
    temp_train = BaseTensorTrain(G,[trainb.ranks;ones(Int,da-db)],0)
    return TTadd(traina,temp_train)
  end

  if db > da # trainb has more cores than traina
    n = Tuple(size(trainb.cores[i])[2] for i in 1:db)
    filler = TTconstant(one(Z),Tuple(n[da+1:end]),1)
    if anchor == "end"
      G = [filler.cores;traina.cores]
    elseif anchor == "start"
      G = [traina.cores;filler.cores]
    else
      error()
    end  
    temp_train = BaseTensorTrain(G,[traina.ranks;ones(Int,db-da)],0)
    return TTadd(temp_train,trainb)
  end

  # traina and trainb have the same number of cores

  na = Tuple(size(traina.cores[i])[2] for i in 1:da)
  nb = Tuple(size(trainb.cores[i])[2] for i in 1:db)

  for i in eachindex(na)
    if na[i] != nb[i]
      error("Trains have different numbers of nodes")
    end
  end

  coresa = deepcopy(traina.cores)
  ra     = copy(traina.ranks)

  coresb = deepcopy(trainb.cores)
  rb     = copy(trainb.ranks)

  G     = Array{Array{Z,3},1}(undef,da)
  r_new = copy(ra)

  if da == 1

    G[1] = coresa[1] .+ coresb[1]

  else

    for i = 1:da

      if i == 1
        g = [reshape(coresa[1],ra[1],na[1]*ra[2]) reshape(coresb[1],rb[1],nb[1]*rb[2])]
        G[1] = reshape(g,ra[i],na[1],(ra[2]+rb[2]))
        r_new[2] = ra[2]+rb[2]
      elseif i == da
        g = [reshape(coresa[da],ra[da],na[da]*ra[da+1]); reshape(coresb[db],rb[db],nb[db]*rb[db+1])]
        G[da] = reshape(g,ra[da]+rb[db],na[da],ra[da+1])
      else
        g = zeros((ra[i]+rb[i]),na[i]*(ra[i+1]+rb[i+1]))
        g[1:ra[i],1:(na[i]*ra[i+1])]         .= reshape(coresa[i],ra[i],na[i]*ra[i+1])
        g[ra[i]+1:end,(na[i]*ra[i+1])+1:end] .= reshape(coresb[i],rb[i],nb[i]*rb[i+1])
        G[i] = reshape(g,(ra[i]+rb[i]),na[i],(ra[i+1]+rb[i+1]))
        r_new[i+1] = ra[i+1]+rb[i+1]
      end

    end

  end

  return BaseTensorTrain(G,r_new,0)

end

"""
Subtract two discrete tensor trains.  When the trains have different lengths, anchor determines whether the trains align from the first core (default) or the last core.

Signatures
==========

t = TTsubstract(traina,trainb)
t = TTsubstract(traina,trainb,anchor)
"""
function TTsubtract(traina::DiscreteTensorTrain,trainb::DiscreteTensorTrain,anchor="start")

  da = length(traina.cores)
  db = length(trainb.cores)

  if da > db # traina has more cores than trainb
    n = Tuple(size(traina.cores[i])[2] for i in 1:da)
    filler = TTconstant(1.0,Tuple(n[db+1:end]),1)
    if anchor == "end"
      G = [filler.cores;trainb.cores]
    elseif anchor == "start"
      G = [trainb.cores;filler.cores]
    else
      error()
    end
    temp_train = BaseTensorTrain(G,[trainb.ranks;ones(Int,da-db)],0)
    return TTsubtract(traina,temp_train)
  end

  if db > da # trainb has more cores than traina
    n = Tuple(size(trainb.cores[i])[2] for i in 1:db)
    filler = TTconstant(1.0,Tuple(n[da+1:end]),1)
    if anchor == "end"
      G = [filler.cores;traina.cores]
    elseif anchor == "start"
      G = [traina.cores;filler.cores]
    else
      error()
    end  
    temp_train = BaseTensorTrain(G,[traina.ranks;ones(Int,db-da)],0)
    return TTsubtract(temp_train,trainb)
  end

  # traina and trainb have the same number of cores

  na = Tuple(size(traina.cores[i])[2] for i in 1:da)
  nb = Tuple(size(trainb.cores[i])[2] for i in 1:db)

  for i in eachindex(na)
    if na[i] != nb[i]
      error("Trains have different numbers of nodes")
    end
  end

  coresa = deepcopy(traina.cores)
  ra     = copy(traina.ranks)

  coresb = deepcopy(trainb.cores)
  rb     = copy(trainb.ranks)

  G     = Array{Array{Float64,3},1}(undef,da)
  r_new = copy(ra)

  coresb[1] = -coresb[1]

  if da == 1

    G[1] = coresa[1] .+ coresb[1]

  else

    for i = 1:da

      if i == 1
        g = [reshape(coresa[1],ra[1],na[1]*ra[2]) reshape(coresb[1],rb[1],nb[1]*rb[2])]
        G[1] = reshape(g,ra[i],na[1],(ra[2]+rb[2]))
        r_new[2] = ra[2]+rb[2]
      elseif i == da && da != 1
        g = [reshape(coresa[da],ra[da],na[da]*ra[da+1]); reshape(coresb[db],rb[db],nb[db]*rb[db+1])]
        G[da] = reshape(g,ra[da]+rb[db],na[da],ra[da+1])
      else
        g = zeros((ra[i]+rb[i]),na[i]*(ra[i+1]+rb[i+1]))
        g[1:ra[i],1:(na[i]*ra[i+1])]         .= reshape(coresa[i],ra[i],na[i]*ra[i+1])
        g[ra[i]+1:end,(na[i]*ra[i+1])+1:end] .= reshape(coresb[i],rb[i],nb[i]*rb[i+1])
        G[i] = reshape(g,(ra[i]+rb[i]),na[i],(ra[i+1]+rb[i+1]))
        r_new[i+1] = ra[i+1]+rb[i+1]
      end

    end

  end

  return BaseTensorTrain(G,r_new,0)

end

"""
Construct the square (element-by-element) of a tensor train.

Signature
=========

t = TTsquared(train)
"""
function TTsquared(train::DiscreteTensorTrain)

  T = eltype(train.cores[begin])
  
  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple(size(train.cores[i])[2] for i in 1:d)

  G = [zeros(T,r[i]^2,n[i],r[i+1]^2) for i = 1:d]
  
  for k = 1:d
    for i = 1:n[k]
      G[k][:,i,:] = kron(train.cores[k][:,i,:],train.cores[k][:,i,:])
    end
  end

  return BaseTensorTrain(G,r.^2,0)

end

"""
Construct the N'th power (element-by-element) of a tensor train.

Signature
=========

t = TTpower(train,N)
"""
function TTpower(train::DiscreteTensorTrain,N::S) where {S<:Integer}

  T = eltype(train.cores[begin])

  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple(size(train.cores[i])[2] for i in 1:d)

  G = [zeros(T,r[i]^N,n[i],r[i+1]^N) for i = 1:d]
  
  for k = 1:d
    for i = 1:n[k]
      A = train.cores[k][:,i,:]
      for j = 2:N
        A = kron(A,train.cores[k][:,i,:])
      end
      G[k][:,i,:] = A
    end

  end

  return BaseTensorTrain(G,r.^N,0)

end

"""
Take the Hadamard product of two tensor trains.

Signature
=========

t = TTHadamard(train1,train2)
"""
function TTHadamard(train1::DiscreteTensorTrain,train2::DiscreteTensorTrain) # Based on the description given in Daas, Ballard, and Benner (2020)

  d1 = length(train1.cores)
  d2 = length(train1.cores)

  T = eltype(train1.cores[1])

  if d1 != d2
    error("Tensor trains have different numbers of cores")
  end

  for i in 1:d1
    if size(train1.cores[i])[2] != size(train2.cores[i])[2]
      error("Tensor trains have different mode lengths for dimension $i")
    end
  end

  d = length(train1.cores)

  G = Array{Array{T,3},1}(undef,d)
  for k = 1:d
    temp = zeros(train1.ranks[k]*train2.ranks[k],size(train1.cores[k])[2],train1.ranks[k+1]*train2.ranks[k+1])
    for i in axes(train1.cores[k],2)
      temp[:,i,:] = kron(train1.cores[k][:,i,:],train2.cores[k][:,i,:])
    end
    G[k] = temp
  end
  
  return BaseTensorTrain(G,train1.ranks.*train2.ranks,0)

end

"""
Approximate the exponential of a tensor train in tensor train format.

Signature
=========

t = TTexp(train,N,tol)
"""
function TTexp(train::DiscreteTensorTrain,order::S,tol::T) where {S<:Integer,T<:AbstractFloat}

  Z = eltype(train.cores[1])

  exp_train = TTconstant(one(Z),TTsize(train),train.ranks)

  for i = 1:order
    extra_term = TTrounding(TTpower(train,i),tol)
    extra_term = TTmult(Z(1/factorial(big(i))),extra_term)
    exp_train = TTrounding(TTadd(exp_train,extra_term),tol)
  end

  return exp_train

end

#### Functions to orthogonalise discrete tensor trains

"""
Perform the RQ matrix decomposition.

Signature
=========

R,Q = rq(A)
"""
function rq(A::AbstractMatrix{T}) where {T<:Real} # not exported

  n = size(A)

  if n[1] > n[2]
    error("Expected fat matrix for RQ decomposition, got matrix with size $n")
  end

  q,r = qr(A')

  return r', Matrix(q)'

end

"""
Right orthogonalise a discrete tensor train.

Signature
=========

t = TTorthright(train)
"""
function TTorthright(train::L) where {L<:BaseTensorTrain}  # Tensor train orthogonalisation based on the description given in Chertkov, Ryzhakov, Novikov, and Oseledets (2022), algorthm 3.

  Π = deepcopy(train.cores)
  d = length(Π)
  n = Tuple(size(Π[k])[2] for k = 1:d)
  r = copy(train.ranks)
  
  for k = d:-1:2

    # Update the k'th core
    G = reshape(Π[k],r[k],n[k]*r[k+1])
    R, Q = rq(G)
    Π[k] = reshape(Q,r[k],n[k],r[k+1])
    # Update the (k-1)'th core
    G = reshape(Π[k-1],r[k-1]*n[k-1],r[k])
    G = G*R
    Π[k-1] = reshape(G,r[k-1],n[k-1],r[k])

  end

  return RightOrthBaseTensorTrain(Π,train.ranks,train.sweeps)

end

function TTorthright(train::L) where {L<:ExtendedTensorTrain}

  Π = deepcopy(train.cores)
  d = length(Π)
  n = Tuple(size(Π[k])[2] for k = 1:d)
  r = copy(train.ranks)
  
  for k = d:-1:2

    # Update the k'th core
    G = reshape(Π[k],r[k],n[k]*r[k+1])
    R, Q = rq(G)
    Π[k] = reshape(Q,r[k],n[k],r[k+1])
    # Update the (k-1)'th core
    G = reshape(Π[k-1],r[k-1]*n[k-1],r[k])
    G = G*R
    Π[k-1] = reshape(G,r[k-1],n[k-1],r[k])

  end

  return RightOrthExtendedTensorTrain(Π,train.ranks,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Left orthogonalise a discrete tensor train.

Signature
=========

t = TTorthleft(train)
"""
function TTorthleft(train::L) where {L<:BaseTensorTrain}

  Π = deepcopy(train.cores)
  d = length(Π)
  n = Tuple(size(Π[k])[2] for k = 1:d)
  r = copy(train.ranks)
  
  for k = 1:d-1

    # Update the k'th core
    G = reshape(Π[k],r[k]*n[k],r[k+1])
    Q, R = qr(G)
    Π[k] = reshape(Matrix(Q),r[k],n[k],r[k+1])
    # Update the (k+1)'th core
    G = reshape(Π[k+1],r[k+1]*n[k+1],r[k+2])
    G = G*R
    Π[k+1] = reshape(G,r[k+1],n[k+1],r[k+2])

  end

  return LeftOrthBaseTensorTrain(Π,train.ranks,train.sweeps)

end

function TTorthleft(train::L) where {L<:ExtendedTensorTrain}

  Π = deepcopy(train.cores)
  d = length(Π)
  n = Tuple(size(Π[k])[2] for k = 1:d)
  r = copy(train.ranks)
  
  for k = 1:d-1

    # Update the k'th core
    G = reshape(Π[k],r[k]*n[k],r[k+1])
    Q, R = qr(G)
    Π[k] = reshape(Matrix(Q),r[k],n[k],r[k+1])
    # Update the (k+1)'th core
    G = reshape(Π[k+1],r[k+1]*n[k+1],r[k+2])
    G = G*R
    Π[k+1] = reshape(G,r[k+1],n[k+1],r[k+2])

  end

  return LeftOrthExtendedTensorTrain(Π,train.ranks,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Left orthogonalise a train up to core 'μ' and right orthogonalise the remaining cores.

Signature
=========

t = TTorthleftright(train,μ)
"""
function TTorthleftright(train::L,μ::S) where {L<:BaseTensorTrain,S<:Integer}

  Π = deepcopy(train.cores)
  d = length(Π)
  n = Tuple(size(Π[k])[2] for k = 1:d)
  r = copy(train.ranks)
  
  for k = 1:μ-1

    # Update the k'th core
    G = reshape(Π[k],r[k]*n[k],r[k+1])
    Q, R = qr(G)
    Π[k] = reshape(Matrix(Q),r[k],n[k],r[k+1])
    # Update the (k+1)'th core
    G = reshape(Π[k+1],r[k+1]*n[k+1],r[k+2])
    G = G*R
    Π[k+1] = reshape(G,r[k+1],n[k+1],r[k+2])

  end

  for k = d:-1:μ+1

    # Update the k'th core
    G = reshape(Π[k],r[k],n[k]*r[k+1])
    R, Q = rq(G)
    Π[k] = reshape(Q,r[k],n[k],r[k+1])
    # Update the (k-1)'th core
    G = reshape(Π[k-1],r[k-1]*n[k-1],r[k])
    G = G*R
    Π[k-1] = reshape(G,r[k-1],n[k-1],r[k])

  end

  return BaseTensorTrain(Π,train.ranks,train.sweeps)

end

function TTorthleftright(train::L,μ::S) where {L<:ExtendedTensorTrain,S<:Integer}

  Π = deepcopy(train.cores)
  d = length(Π)
  n = Tuple(size(Π[k])[2] for k = 1:d)
  r = copy(train.ranks)
  
  for k = 1:μ-1

    # Update the k'th core
    G = reshape(Π[k],r[k]*n[k],r[k+1])
    Q, R = qr(G)
    Π[k] = reshape(Matrix(Q),r[k],n[k],r[k+1])
    # Update the (k+1)'th core
    G = reshape(Π[k+1],r[k+1]*n[k+1],r[k+2])
    G = G*R
    Π[k+1] = reshape(G,r[k+1],n[k+1],r[k+2])

  end

  for k = d:-1:μ+1

    # Update the k'th core
    G = reshape(Π[k],r[k],n[k]*r[k+1])
    R, Q = rq(G)
    Π[k] = reshape(Q,r[k],n[k],r[k+1])
    # Update the (k-1)'th core
    G = reshape(Π[k-1],r[k-1]*n[k-1],r[k])
    G = G*R
    Π[k-1] = reshape(G,r[k-1],n[k-1],r[k])

  end

  return ExtendedTensorTrain(Π,train.ranks,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Compute the norm of a discrete tensor train.

Signature
=========

TTnorm(train)
"""
function TTnorm(train::DiscreteTensorTrain)

  if typeof(train) <: Union{RightOrthBaseTensorTrain,RightOrthExtendedTensorTrain}
    return norm(orthtrain.cores[1])
  elseif typeof(train) <: Union{LeftOrthBaseTensorTrain,LeftOrthExtendedTensorTrain}
    return norm(orthtrain.cores[d])
  else
    try
      orthtrain = TTorthright(train)
      return norm(orthtrain.cores[1])
    catch
      orthtrain = TTorthleft(train)
      return norm(orthtrain.cores[d])
    end
  end

end

"""
Compute the (implied) size of a decompressed tensor train.

Signature
=========

n = TTsize(train)
"""
function TTsize(train::DiscreteTensorTrain)

  d = length(train.cores)
  n = Tuple(size(train.cores[i])[2] for i in 1:d)

  return n

end

"""
Compute the inner product of two tensor trains.

Signature
=========

inner_prod = TTinner_prod(traina,trainb)
"""
function TTinner_prod(traina::DiscreteTensorTrain,trainb::DiscreteTensorTrain)

  na = TTsize(traina)
  nb = TTsize(trainb)

  if length(na) != length(nb) || sum(na .- nb) != 0
    error("Tensor trains have incompatible dimensions")
  end

  d = length(na)

  ga = deepcopy(traina.cores)
  gb = deepcopy(trainb.cores)

  ra = copy(traina.ranks)
  rb = copy(trainb.ranks)

  for i = 1:d-1

    temp = transpose(transpose(reshape(ga[i],ra[i]*na[i],ra[i+1]))*reshape(gb[i],rb[i]*nb[i],rb[i+1]))
    ga[i+1] = reshape(temp*reshape(ga[i+1],ra[i+1],na[i+1]*ra[i+2]),rb[i+1],na[i+1],ra[i+2])
    ra[i+1] = rb[i+1]
    
  end

  inner_prod = reshape(ga[d],ra[d]*na[d],ra[d+1])'*reshape(gb[d],rb[d]*nb[d],rb[d+1])

  return inner_prod[1]

end

"""
Compute the inner product of two tensor trains where traina has fewer dimensions than trainb.

Signature
=========

trainc = TTinner_prod2(traina,trainb)
"""
function TTinner_prod2(traina::DiscreteTensorTrain,trainb::DiscreteTensorTrain)

  na = TTsize(traina)
  nb = TTsize(trainb)

  da = length(na)
  db = length(nb)

  if sum(na .- nb[1:da]) != 0
    error("Tensor trains have incompatible dimensions")
  end

  if length(na) == length(nb)
    error("Trains have equal dimensions: try using TTinner_prod(traina,trainb)")
  elseif length(na) > length(nb)
    error("traina has more dimensions than trainb")
  end

  ga = deepcopy(traina.cores)
  gb = deepcopy(trainb.cores)

  ra = copy(traina.ranks)
  rb = copy(trainb.ranks)

  for i = 1:da-1

    temp = transpose(transpose(reshape(ga[i],ra[i]*na[i],ra[i+1]))*reshape(gb[i],rb[i]*nb[i],rb[i+1]))
    ga[i+1] = reshape(temp*reshape(ga[i+1],ra[i+1],na[i+1]*ra[i+2]),rb[i+1],na[i+1],ra[i+2])
    ra[i+1] = rb[i+1]
    
  end

  inner_prod = reshape(ga[da],ra[da]*na[da],ra[da+1])'*reshape(gb[da],rb[da]*nb[da],rb[da+1])

  gc = deepcopy(trainb.cores[da+1:end])
  rc = [1;rb[da+1:end]]

  gc[1] = times_dim_1(inner_prod,gc[1])

  return BaseTensorTrain(gc,rc,0)

end

"""
Use bisection to find the fix-point of the continuous function 'f'.

Signature
=========

xstar, f_xstar, iters = bisection(f,x,tol,maxiters)
"""
function bisection(f::Function,x::Array{R,1},tol::T,maxiters::S) where {R<:AbstractFloat,T<:AbstractFloat,S<:Integer} # not exported

  b = x[2]
  a = x[1]

  c = (a+b)/2

  iter = 0
  while true

    if f(c)*f(b) <= 0.0
      a, b = c, b
    else
      a, b = a, c
    end

    len = abs(b-a)

    c = (a+b)/2

    iter += 1
    if iter >= maxiters || len <= tol
      break
    end

  end

  return c, f(c), iter

end

"""
Use the trapazoidal method to integrate the cores of a discrete tensor train.

Signature
=========

integral = trapazoidal(y,domain)
"""
function trapazoidal(y::AbstractArray{T1,3},domain::Array{T2,1}) where {T1<:AbstractFloat,T2<:AbstractFloat} # not exported

  n = size(y)

  integral_f = y[:,begin,:]+y[:,end,:]
  for i = 2:(n[2]-1)
    integral_f += 2*y[:,i,:]
  end

  integral_f = (1/(n[2]-1))*((domain[1]-domain[2])/2)*integral_f

  return integral_f

end

"""
Take a draw from a density by inverting the CDF.

Signatures
==========

c = invert_cdf(f,u)       # Unbounded domain
c = invert_cdf(f,u,lb,ub) # Bounded domain
"""
function invert_cdf(f::Function,u::T) where {T<:AbstractFloat} # not exported

  g(x) = f(x)-u

  c, fc, its = bisection(g,[floatmax(T),floatmin(T)],eps(),5_000)

  return c

end

function invert_cdf(f::Function,u::T,lb::T,ub::T) where {T<:AbstractFloat} # not exported

  g(x) = f(x)-u

  c, fc, its = bisection(g,[ub,lb],eps(),5_000)

  return c

end

#### Tensor train conditional distribution sampling based on the descriptions given in Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020), Statistics and Computing

"""
Conditional distribution sampling when the nodes are Gauss-Hermite.

Signatures
==========

sample = TTCD_GH(train,N)
sample = TTCD_GH(train,N,seed)
"""
function TTCD_GH(train::DiscreteTensorTrain,N::S,seed::S = 123456) where {S<:Integer} # Based on the descriptions given in Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020)

  d = length(train.cores)
  Π = deepcopy(train.cores)
  n = Tuple(size(Π[i]) for i in 1:d)
  r = copy(train.ranks)

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(N,d)

  P = Array{Array{Float64,2},1}(undef,d+1)
  Φ = Array{Array{Float64,2},1}(undef,d+1)
  Ψ = Array{Array{Float64,2},1}(undef,d)

  nodes   = Array{Array{Float64,1},1}(undef,d)
  weights = Array{Array{Float64,1},1}(undef,d)

  P[d+1] = [1.0;;]
  for k = d:-1:1
    term = zeros(n[k][1],n[k][3])
    nodes[k], weights[k] = hermite(n[k][2])
    for j = 1:n[k][2]
      term += Π[k][:,j,:]*exp(nodes[k][j]^2.0)*weights[k][j]
    end
    P[k] = term*P[k+1]
  end

  Φ[1] = ones(N,1)
  for k = 1:d
    ϕ = zeros(N,n[k][3])
    Ψ[k] = times_dim_3(Π[k],P[k+1])[:,:,1] # [Π[k][:,i,:]*P[k+1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l:l,:]*Ψ[k])[:] # p is now a vector with length n[k][2]
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      g = [piecewise_linear_evaluate(Π[k][i,:,j],nodes[k]) for i = 1:r[k], j = 1:r[k+1]]
      ϕ[l,:] = Φ[k][l,:]'*[g[i,j].(sample[l,k]) for i in axes(g,1),j in axes(g,2)]
    end
    Φ[k+1] = ϕ
  end

  return sample

end

"""
Conditional distribution sampling when the nodes are Gauss-Chebyshev.

Signatures
==========

sample = TTCD_GC(train,N,domain)
sample = TTCD_GC(train,N,domain,seed)
"""
function TTCD_GC(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T<:AbstractFloat,S<:Integer} # Based on the descriptions given in Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020)

  d = length(train.cores)
  Π = deepcopy(train.cores)
  n = Tuple(size(Π[i]) for i in 1:d)
  r = copy(train.ranks)

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(T,N,d)

  P = Array{Array{T,2},1}(undef,d+1)
  Φ = Array{Array{T,2},1}(undef,d+1)
  Ψ = Array{Array{T,2},1}(undef,d)

  nodes   = Array{Array{T,1},1}(undef,d)
  weights = Array{Array{T,1},1}(undef,d)

  P[d+1] = [1.0;;]
  for k = d:-1:1
    term = zeros(n[k][1],n[k][3])
    nodes[k], weights[k] = chebyshev(n[k][2])
    for j = 1:n[k][2]
      term += Π[k][:,j,:]*sqrt(1.0-nodes[k][j]^2)*weights[k][j]
    end
    P[k] = term*((domain[1,k]-domain[2,k])/2)*P[k+1]
    nodes[k] = (domain[1,k]+domain[2,k])/2 .+ nodes[k]*((domain[1,k]-domain[2,k])/2)
  end

  Φ[1] = ones(N,1)
  for k = 1:d
    ϕ = zeros(N,n[k][3])
    Ψ[k] = times_dim_3(Π[k],P[k+1])[:,:,1] # [Π[k][:,i,:]*P[k+1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l:l,:]*Ψ[k])[:] # p is now a vector with length n[k][2]
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      g = [piecewise_linear_evaluate(Π[k][i,:,j],nodes[k]) for i = 1:r[k], j = 1:r[k+1]]
      ϕ[l,:] = Φ[k][l,:]'*[g[i,j].(sample[l,k]) for i in axes(g,1),j in axes(g,2)]
    end
    Φ[k+1] = ϕ
  end

  return sample

end

"""
Conditional distribution sampling when the nodes are Gauss-Legendre.

Signatures
==========

sample = TTCD_GL(train,N,domain)
sample = TTCD_GL(train,N,domain,seed)
"""
function TTCD_GL(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T<:AbstractFloat,S<:Integer} # Based on the descriptions given in Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020)

  d = length(train.cores)
  Π = deepcopy(train.cores)
  n = Tuple(size(Π[i]) for i in 1:d)
  r = copy(train.ranks)

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(T,N,d)

  P = Array{Array{T,2},1}(undef,d+1)
  Φ = Array{Array{T,2},1}(undef,d+1)
  Ψ = Array{Array{T,2},1}(undef,d)

  nodes   = Array{Array{T,1},1}(undef,d)
  weights = Array{Array{T,1},1}(undef,d)

  P[d+1] = [1.0;;]
  for k = d:-1:1
    term = zeros(n[k][1],n[k][3])
    nodes[k], weights[k] = legendre(n[k][2])
    for j = 1:n[k][2]
      term += Π[k][:,j,:]*weights[k][j]
    end
    P[k] = term*((domain[1,k]-domain[2,k])/2)*P[k+1]
    nodes[k] = (domain[1,k]+domain[2,k])/2 .+ nodes[k]*((domain[1,k]-domain[2,k])/2)
  end

  Φ[1] = ones(N,1)
  for k = 1:d
    ϕ = zeros(N,n[k][3])
    Ψ[k] = times_dim_3(Π[k],P[k+1])[:,:,1] # [Π[k][:,i,:]*P[k+1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l:l,:]*Ψ[k])[:] # p is now a vector with length n[k][2]
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      g = [piecewise_linear_evaluate(Π[k][i,:,j],nodes[k]) for i = 1:r[k], j = 1:r[k+1]]
      ϕ[l,:] = Φ[k][l,:]'*[g[i,j].(sample[l,k]) for i in axes(g,1),j in axes(g,2)]
    end
    Φ[k+1] = ϕ
  end

  return sample

end

"""
Conditional distribution sampling when the nodes are uniformly spaced.

Signatures
==========

sample = TTCD_PL(train,N,domain)
sample = TTCD_PL(train,N,domain,seed)
"""
function TTCD_PL(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T<:AbstractFloat,S<:Integer} # Based on the descriptions given in Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020)

  d = length(train.cores)
  Π = deepcopy(train.cores)
  n = Tuple(size(Π[i]) for i in 1:d)
  r = copy(train.ranks)

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(T,N,d)

  P = Array{Array{T,2},1}(undef,d+1)
  Φ = Array{Array{T,2},1}(undef,d+1)
  Ψ = Array{Array{T,2},1}(undef,d)

  nodes   = Array{Array{T,1},1}(undef,d)

  P[d+1] = [1.0;;]
  for k = d:-1:1
    nodes[k] = piecewise_linear_nodes(n[k][2],domain[:,k])
    term = trapazoidal(Π[k],nodes[k])
    P[k] = term*P[k+1]
  end

  Φ[1] = ones(N,1)
  for k = 1:d
    ϕ = zeros(N,n[k][3])
    Ψ[k] = times_dim_3(Π[k],P[k+1])[:,:,1] # [Π[k][:,i,:]*P[k+1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l:l,:]*Ψ[k])[:] # p is now a vector with length n[k][2]
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      g = [piecewise_linear_evaluate(Π[k][i,:,j],nodes[k]) for i = 1:r[k], j = 1:r[k+1]]
      ϕ[l,:] = Φ[k][l,:]'*[g[i,j].(sample[l,k]) for i in axes(g,1),j in axes(g,2)]
    end
    Φ[k+1] = ϕ
  end

  return sample

end

"""
Metropolis-Hasting correction of a sample based on the density and its functional tensor train approximation.

Signatures
==========

sample = TTMH(pdf,train,sample)
sample = TTMH(pdf,train,sample,seed)
"""
function TTMH(density::Function,train::FunctionalTensorTrain,sample::Array{T,2},seed::S=123456) where {T<:AbstractFloat,S<:Integer} # Based on the description given in Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020), Statistics and Computing

  n = size(sample)

  trainfn = TTinterp(train)

  new_sample = similar(sample)

  rng = MersenneTwister(seed)
  new_sample[1,:] .= sample[1,:]
  draw_index = 1

  for i = 2:n[1]

    h = (density(sample[i,:]) / trainfn(sample[i,:])) * (trainfn(sample[draw_index,:]) / density(sample[draw_index,:]))

    if rand(rng) <= min(h,1.0) # Accept candidate draw
      new_sample[i,:] .= sample[i,:]
      draw_index = i
    else # Retain existing draw
      new_sample[i,:] .= sample[draw_index,:]
    end

  end

  return new_sample

end

"""
Metropolis-Hasting correction of a sample based on the log-density and its functional tensor train approximation.

Signatures
==========

sample = TTMHlog(logpdf,logtrain,sample)
sample = TTMHlog(logpdf,logtrain,sample,seed)
"""
function TTMHlog(logdensity::Function,logtrain::FunctionalTensorTrain,sample::Array{T,2},seed::S=123456) where {T<:AbstractFloat,S<:Integer} # Based on the description given in Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020), Statistics and Computing

  n = size(sample)

  logtrainfn = TTinterp(logtrain)

  new_sample = similar(sample)

  rng = MersenneTwister(seed)
  new_sample[1,:] .= sample[1,:]
  draw_index = 1

  for i = 2:n[1]

    h = exp(logdensity(sample[i,:]) - logtrainfn(sample[i,:]) + logtrainfn(sample[draw_index,:]) - logdensity(sample[draw_index,:]))

    if rand(rng) <= min(h,1.0) # Accept candidate draw
      new_sample[i,:] .= sample[i,:]
      draw_index = i
    else # Retain existing draw
      new_sample[i,:] .= sample[draw_index,:]
    end

  end

  return new_sample

end

"""
sample = TTSIRT_GC(train,N,domain,seed)
sample = TTSIRT_GC(train,N,domain)

Conditional distribution sampling from the square of a tensor train when the nodes are Gauss-Chebyshev.
"""
function TTSIRT_GC(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T<:AbstractFloat,S<:Integer} # Squared inverse Rosenblatt transport with GC integration

  temp_train = TTsquared(train)

  sample = TTCD_GC(temp_train,N,domain,seed)

  return sample

end

"""
sample = TTSIRT_GL(train,N,domain,seed)
sample = TTSIRT_GL(train,N,domain)

Conditional distribution sampling from the square of a tensor train when the nodes are Gauss-Legendre.
"""
function TTSIRT_GL(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T<:AbstractFloat,S<:Integer} # Squared inverse Rosenblatt transport with GL integration

  temp_train = TTsquared(train)

  sample = TTCD_GL(temp_train,N,domain,seed)

  return sample

end

"""
sample = TTSIRT_GH(train,N,seed)
sample = TTSIRT_GH(train,N)

Conditional distribution sampling from the square of a tensor train when the nodes are Gauss-Hermite.
"""
function TTSIRT_GH(train::DiscreteTensorTrain,N::S,seed::S = 123456) where {S<:Integer} # Squared inverse Rosenblatt transport with GH integration

  temp_train = TTsquared(train)

  sample = TTCD_GH(temp_train,N,seed)

  return sample

end

"""
sample = TTSIRT_PL(train,N,domain,seed)
sample = TTSIRT_PL(train,N,domain)

Conditional distribution sampling from the square of a tensor train when the nodes are uniformly spaced.
"""
function TTSIRT_PL(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T<:AbstractFloat,S<:Integer} # Squared inverse Rosenblatt transport with trapazoidal integration

  temp_train = TTsquared(train)

  sample = TTCD_PL(temp_train,N,domain,seed)

  return sample

end

#### Functions to optimize over a discrete tensor train

"""
Find the 'K' rows of a matrix with the highest norm.

Signature
=========

p = topK(M,K)
"""
function topK(M::Array{T,2},K::S) where {T<:AbstractFloat,S<:Integer} # not exported

  n = size(M)

  m = zeros(n[1])
  for i in eachindex(m)
    m[i] = norm(M[i,:])
  end
  p = sortperm(m,rev=true)

  return p[1:min(n[1],K)]

end

"""
Horizontally concatenate two matrices.

Signature
=========

c = stack(a,b)
"""
function stack(a::Array{T,2},b::Array{T,2}) where {T<:AbstractFloat} # not exported

  return [a b]

end

"""
Extremize over the cores of a tensor train.  The output could be a maxima or a minima.  If 'state' is provided, the initial cores in the 
tensor train are associated with the state variables. 'state' can be an index for a grid location or a point in the state space.

Signatures
==========

soln = TTextremize(train,K)
soln = TTextremize(train,K,state)       # state is a vector of integers
soln = TTextremize(train,K,nodes,state) # state is a vector of floating point numbers
"""
function TTextremize(train::DiscreteTensorTrain,K::S) where {S<:Integer} # Based on the description given in Chertkov, Ryzhakov, Novikov, and Oseledets (2022), algorthm 1.

  orth_train = TTorthright(train)  
  Π = deepcopy(orth_train.cores)
  
  d = length(Π)
  r = copy(orth_train.ranks)
  n = Tuple(size(Π[i])[2] for i = 1:d)

  Q = Π[1][1,:,:]
  ind = topK(Q,K)
  I = [1:1:n[1];;]
  Q = Q[ind,:]
  I = I[ind,:]

  for k = 2:d

    G = reshape(Π[k],r[k],n[k]*r[k+1])
    Q = Q*G
    Q = reshape(Q,K*n[k],r[k+1])
    ind = topK(Q,K)
    subs = [ind2sub(ind[i],(K,n[k]))[2] for i in 1:K]
    Q = Q[ind,:]
    I = [I subs]

  end

  return I[1,:]

end

function TTextremize(train::Union{RightOrthBaseTensorTrain,RightOrthExtendedTensorTrain},K::S) where {S<:Integer} # Based on the description given in Chertkov, Ryzhakov, Novikov, and Oseledets (2022), algorthm 1.

  Π = deepcopy(train.cores)
  
  d = length(Π)
  r = copy(train.ranks)
  n = Tuple(size(Π[i])[2] for i = 1:d)

  Q = Π[1][1,:,:]
  ind = topK(Q,K)
  I = [1:1:n[1];;]
  Q = Q[ind,:]
  I = I[ind,:]

  for k = 2:d

    G = reshape(Π[k],r[k],n[k]*r[k+1])
    Q = Q*G
    Q = reshape(Q,K*n[k],r[k+1])
    ind = topK(Q,K)
    subs = [ind2sub(ind[i],(K,n[k]))[2] for i in 1:K]
    Q = Q[ind,:]
    I = [I subs]

  end

  return I[1,:]

end

function TTextremize(train::DiscreteTensorTrain,K::S,state::Array{S,1}) where {S<:Integer}

  orth_train = TTorthright(train)  
  Π = deepcopy(orth_train.cores)
  
  d = length(Π)
  ds = length(state)
  r = copy(orth_train.ranks)
  n = Tuple(size(Π[i])[2] for i = 1:d)

  if ds == d
    return state
  end

  Q = Π[1][:,state[1],:]
  for i = 2:ds
    Q = Q*Π[i][:,state[i],:]
  end
  Q = Q*reshape(Π[ds+1],r[ds+1],n[ds+1]*r[ds+2])
  Q = reshape(Q,n[ds+1],r[ds+2])
  ind = topK(Q,K)
  Q = Q[ind,:]
  I = [1:1:n[ds+1];;]
  I = I[ind,:]

  for k = (ds+2):d

    G = reshape(Π[k],r[k],n[k]*r[k+1])
    Q = Q*G
    Q = reshape(Q,K*n[k],r[k+1])
    ind = topK(Q,K)
    subs = [ind2sub(ind[i],(K,n[k]))[2] for i in 1:K]
    Q = Q[ind,:]
    I = [I subs]

  end

  return I[1,:]

end

function TTextremize(train::Union{RightOrthBaseTensorTrain,RightOrthExtendedTensorTrain},K::S,state::Array{S,1}) where {S<:Integer}

  Π = deepcopy(train.cores)
  
  d = length(Π)
  ds = length(state)
  r = copy(train.ranks)
  n = Tuple(size(Π[i])[2] for i = 1:d)

  if ds == d
    return state
  end

  Q = Π[1][:,state[1],:]
  for i = 2:ds
    Q = Q*Π[i][:,state[i],:]
  end
  Q = Q*reshape(Π[ds+1],r[ds+1],n[ds+1]*r[ds+2])
  Q = reshape(Q,n[ds+1],r[ds+2])
  ind = topK(Q,K)
  Q = Q[ind,:]
  I = [1:1:n[ds+1];;]
  I = I[ind,:]

  for k = (ds+2):d

    G = reshape(Π[k],r[k],n[k]*r[k+1])
    Q = Q*G
    Q = reshape(Q,K*n[k],r[k+1])
    ind = topK(Q,K)
    subs = [ind2sub(ind[i],(K,n[k]))[2] for i in 1:K]
    Q = Q[ind,:]
    I = [I subs]

  end

  return I[1,:]

end

function TTextremize(train::DiscreteTensorTrain,K::S,nodes::NTuple{ds,Array{T,1}},state::Array{T,1}) where {S<:Integer,T<:AbstractFloat,ds}

  state_index = [findfirst(x->x==state[i],nodes[i]) for i in 1:ds]

  ind = TTextremize(train,K,state_index)
  return ind

end

"""
Optimize over the cores of a tensor train, returning the maximum and the minimum.  If 'state' is provided, the initial cores in the 
tensor train are associated with the state variables. 'state' can be an index for a grid location or a point in the state space.

Signatures
==========

(imin,imax),(ymin,ymax) = TToptimize(train,K,tol)
(imin,imax),(ymin,ymax) = TToptimize(train,K,state,tol) # state is a vector of integers
(imin,imax),(ymin,ymax) = TToptimize(train,K,nodes,state,tol) # state is a vector of floating point numbers
"""
function TToptimize(train::DiscreteTensorTrain,K::S,tol=1e-12) where {S<:Integer} # Based on the description given in Chertkov, Ryzhakov, Novikov, and Oseledets (2022), algorthm 2.

  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple(size(train.cores[i])[2] for i in 1:d)

  imax = TTextremize(train,K)
  ymax = TTevaluate(train,imax)

  c = TTconstant(ymax,n,r)

  difference = TTrounding(TTsubtract(train,c),tol)

  imin = TTextremize(difference,K)
  ymin = TTevaluate(train,imin)

  if ymax >= ymin

    return (imin, imax), (ymin, ymax)

  else

    return (imax, imin), (ymax, ymin)

  end

end

function TToptimize(train::DiscreteTensorTrain,K::S,state::Array{S,1},tol=1e-12) where {S<:Integer}

  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple(size(train.cores[i])[2] for i in 1:d)

  imax = TTextremize(train,K,state)
  ymax = TTevaluate(train,[state;imax])

  c = TTconstant(ymax,n,r)

  difference = TTrounding(TTsubtract(train,c),tol)

  imin = TTextremize(difference,K,state)
  ymin = TTevaluate(train,[state;imin])

  if ymax >= ymin

    return (imin, imax), (ymin, ymax)

  else

    return (imax, imin), (ymax, ymin)

  end

end

function TToptimize(train::DiscreteTensorTrain,K::S,nodes::NTuple{ds,Array{T,1}},state::Array{T,1},tol=1e-12) where {S<:Integer,T<:AbstractFloat,ds}

  state_index = [findfirst(x->x==state[i],nodes[i]) for i = 1:ds]

  (imax, imin), (ymax, ymin) = TToptimize(train,K,state_index,tol)

  return (imax, imin), (ymax, ymin)

end

function update_left(X::Array{T,2},x::Array{T,1},n::S,r::S,ind::Array{S,1}) where {T<:AbstractFloat,S<:Integer}

  W₁ = kron(ones(r),x)
  W₂ = kron(X,ones(n))
  W = [W₁ W₂]
  return W[ind,:]

end

function update_right(X::Array{T,2},x::Array{T,1},n::S,r::S,ind::Array{S,1}) where {T<:AbstractFloat,S<:Integer}

  W₁ = kron(ones(n),X)
  W₂ = kron(x,ones(r))
  W = [W₁ W₂]

  return W[ind,:]

end

"""
Minimize a function by discretizing and compressing it using a tensor train with maximum rank, rmax.

Signature
=========

xstar, xstar_index, f_xstar = TTOpt(f,nodes,rmax,sweeps)
"""
function TTOpt(f::Function,nodes::NTuple{d,Array{T,1}},rmax::S,sweeps::S,seed::S = 123456) where {T<:AbstractFloat,S<:Integer,d} # Based on Algorithm A1 as described in Sozykin, Chertkov, Schutski, Phan, Cichocki, and Oseledets (2022)

  Random.seed!(seed)

  n = Tuple(length.(nodes))

  X = Array{Array{T,2},1}(undef,d)
  
  r = Array{S,1}(undef,d+1)
  r[1] = 1
  for i in 2:d
    r[i] = min(r[i-1]*n[i-1],r[i-1]*n[i],rmax)
  end
  r[d+1] = 1
  
  for i in 1:d-1

    if i == 1
      g = randn(r[1]*n[1],r[2])
      q,R = qr(g)
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      W2 = kron(nodes[1],ones(r[1]))
      X[1] = W2[ind,:]
    else
      g = randn(r[i]*n[i],r[i+1])
      q,R = qr(g)
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      X[i] = update_right(X[i-1],nodes[i],n[i],r[i],ind)
    end
  end

  θₘᵢₙ = Array{T,1}(undef,d)
  Jₘᵢₙ = Inf

  for k = 1:sweeps

    # Iterate right to left
  
    # i == d
  
    W1 = kron(ones(n[d]*r[d+1]),X[d-1])
    W2 = kron(kron(ones(r[d+1]),nodes[d]),ones(r[d]))
    W = [W1 W2]
    Z = [f(W[i,:]) for i in axes(W,1)]
    if minimum(Z) < Jₘᵢₙ
      Jₘᵢₙ = minimum(Z)
      mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
      θₘᵢₙ = W[mₘᵢₙ,:]
    end
    Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

    Z = reshape(Z,r[d],n[d])
    q,R = qr(Z')
    ind,s = maxvol_generic!(Matrix(q),1.05,300)
    W1 = kron(ones(r[d+1]),nodes[d])
    X[d] = W1[ind,:]

    # Treat dimensions d-1 to 2
  
    for i = d-1:-1:2
  
      W1 = kron(ones(n[i]*r[i+1]),X[i-1])
      W2 = kron(kron(ones(r[i+1]),nodes[i]),ones(r[i]))
      W3 = kron(X[i+1],ones(r[i]*n[i]))
      W = [W1 W2 W3]
      Z = [f(W[i,:]) for i in axes(W,1)]
      if minimum(Z) < Jₘᵢₙ
        Jₘᵢₙ = minimum(Z)
        mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
        θₘᵢₙ = W[mₘᵢₙ,:]
      end
      Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

      Z = reshape(Z,r[i],n[i]*r[i+1])
      q,R = qr(Z')
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      X[i] = update_left(X[i+1],nodes[i],n[i],r[i+1],ind)
  
    end
  
    # Iterate left to right
  
    # Treat dimension 1
  
    W2 = kron(kron(ones(r[2]),nodes[1]),ones(r[1]))
    W3 = kron(X[2],ones(r[1]*n[1]))
    W = [W2 W3]
    Z = [f(W[i,:]) for i in axes(W,1)]
    if minimum(Z) < Jₘᵢₙ
      Jₘᵢₙ = minimum(Z)
      mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
      θₘᵢₙ = W[mₘᵢₙ,:]
    end
    Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

    Z = reshape(Z,n[1],r[2])
    q,R = qr(Z)
    ind,s = maxvol_generic!(Matrix(q),1.05,300)
    W2 = kron(nodes[1],ones(r[1]))
    X[1] = W2[ind,:]
  
    # Treat dimension 2 to d-1
  
    for i = 2:d-1
  
      W1 = kron(ones(n[i]*r[i+1]),X[i-1])
      W2 = kron(kron(ones(r[i+1]),nodes[i]),ones(r[i]))
      W3 = kron(X[i+1],ones(r[i]*n[i]))
      W = [W1 W2 W3]
      Z = [f(W[i,:]) for i in axes(W,1)]
      if minimum(Z) < Jₘᵢₙ
        Jₘᵢₙ = minimum(Z)
        mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
        θₘᵢₙ = W[mₘᵢₙ,:]
      end
      Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

      Z = reshape(Z,r[i]*n[i],r[i+1])
      q,R = qr(Z)
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      X[i] = update_right(X[i-1],nodes[i],n[i],r[i],ind)

    end

  end

  θᵢₙ = [findmin(abs.(nodes[i] .- θₘᵢₙ[i]))[2] for i = 1:d]

  return  θₘᵢₙ, θᵢₙ, Jₘᵢₙ

end

function TTOpt(f::Function,nodes::NTuple{d,Array{T,1}},ranks::Array{S,1},sweeps::S,seed::S=123456) where {T<:AbstractFloat,S<:Integer,d} # Based on Algorithm A1 as described in Sozykin, Chertkov, Schutski, Phan, Cichocki, and Oseledets (2022)

Random.seed!(seed)

  if length(ranks) != d+1
    error{"ranks must have length d+1"}
  end
  
  n = Tuple(length.(nodes))

  X = Array{Array{T,2},1}(undef,d)
  
  r = Array{S,1}(undef,d+1)
  r[1] = 1
  for i in 2:d
    r[i] = min(r[i-1]*n[i-1],r[i-1]*n[i],ranks[i])
  end
  r[d+1] = 1
  
  for i in 1:d-1

    if i == 1
      g = randn(r[1]*n[1],r[2])
      q,R = qr(g)
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      W2 = kron(nodes[1],ones(r[1]))
      X[1] = W2[ind,:]
    else
      g = randn(r[i]*n[i],r[i+1])
      q,R = qr(g)
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      X[i] = update_right(X[i-1],nodes[i],n[i],r[i],ind)
    end
  end

  θₘᵢₙ = Array{T,1}(undef,d)
  Jₘᵢₙ = Inf

  for k = 1:sweeps

    # Iterate right to left
  
    # i == d
  
    W1 = kron(ones(n[d]*r[d+1]),X[d-1])
    W2 = kron(kron(ones(r[d+1]),nodes[d]),ones(r[d]))
    W = [W1 W2]
    Z = [f(W[i,:]) for i in axes(W,1)]
    if minimum(Z) < Jₘᵢₙ
      Jₘᵢₙ = minimum(Z)
      mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
      θₘᵢₙ = W[mₘᵢₙ,:]
    end
    Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

    Z = reshape(Z,r[d],n[d])
    q,R = qr(Z')
    ind,s = maxvol_generic!(Matrix(q),1.05,300)
    W1 = kron(ones(r[d+1]),nodes[d])
    X[d] = W1[ind,:]

    # Treat dimensions d-1 to 2
  
    for i = d-1:-1:2
  
      W1 = kron(ones(n[i]*r[i+1]),X[i-1])
      W2 = kron(kron(ones(r[i+1]),nodes[i]),ones(r[i]))
      W3 = kron(X[i+1],ones(r[i]*n[i]))
      W = [W1 W2 W3]
      Z = [f(W[i,:]) for i in axes(W,1)]
      if minimum(Z) < Jₘᵢₙ
        Jₘᵢₙ = minimum(Z)
        mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
        θₘᵢₙ = W[mₘᵢₙ,:]
      end
      Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

      Z = reshape(Z,r[i],n[i]*r[i+1])
      q,R = qr(Z')
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      X[i] = update_left(X[i+1],nodes[i],n[i],r[i+1],ind)
  
    end
  
    # Iterate left to right
  
    # Treat dimension 1
  
    W2 = kron(kron(ones(r[2]),nodes[1]),ones(r[1]))
    W3 = kron(X[2],ones(r[1]*n[1]))
    W = [W2 W3]
    Z = [f(W[i,:]) for i in axes(W,1)]
    if minimum(Z) < Jₘᵢₙ
      Jₘᵢₙ = minimum(Z)
      mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
      θₘᵢₙ = W[mₘᵢₙ,:]
    end
    Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

    Z = reshape(Z,n[1],r[2])
    q,R = qr(Z)
    ind,s = maxvol_generic!(Matrix(q),1.05,300)
    W2 = kron(nodes[1],ones(r[1]))
    X[1] = W2[ind,:]
  
    # Treat dimension 2 to d-1
  
    for i = 2:d-1
  
      W1 = kron(ones(n[i]*r[i+1]),X[i-1])
      W2 = kron(kron(ones(r[i+1]),nodes[i]),ones(r[i]))
      W3 = kron(X[i+1],ones(r[i]*n[i]))
      W = [W1 W2 W3]
      Z = [f(W[i,:]) for i in axes(W,1)]
      if minimum(Z) < Jₘᵢₙ
        Jₘᵢₙ = minimum(Z)
        mₘᵢₙ = findfirst(x->x==Jₘᵢₙ,Z)[1]
        θₘᵢₙ = W[mₘᵢₙ,:]
      end
      Z .= π/2 .- atan.(Z .- Jₘᵢₙ)

      Z = reshape(Z,r[i]*n[i],r[i+1])
      q,R = qr(Z)
      ind,s = maxvol_generic!(Matrix(q),1.05,300)
      X[i] = update_right(X[i-1],nodes[i],n[i],r[i],ind)

    end

  end

  θᵢₙ = [findmin(abs.(nodes[i] .- θₘᵢₙ[i]))[2] for i = 1:d]

  return  θₘᵢₙ, θᵢₙ, Jₘᵢₙ

end

"""
Reverse the cores in a discrete tensor train (reversing the order of the variables).

Signature
=========

t = Ttreverse(train)
"""
function TTreverse_indices(train::DiscreteTensorTrain)

  d     = length(train.cores)
  cores = train.cores
  r     = train.ranks
  n     = Tuple(size(cores[i])[2] for i = 1:d)

  G = similar(cores)
  new_r = copy(r)

  for i in d:-1:1
    temp = zeros(r[i+1],n[i],r[i])
    for j in CartesianIndices(cores[i])
      temp[j[3],j[2],j[1]] = cores[i][j]
    end
    G[1+abs(i-d)]     = temp
    new_r[1+abs(i-d)] = r[i+1]
  end

  return BaseTensorTrain(G,new_r,0)

end

function TTderivative(train::DiscreteTensorTrain,nodes::AbstractArray{R,1},ind::S) where {R<:Real,S<:Integer}

  G = deepcopy(train.cores)
  n = size(G[ind])

  for i in 1:n[1]
    for k in 1:n[3]
      for j in 1:n[2]
        if j == 1
          G[ind][i,j,k] = (train.cores[ind][i,j+1,k] - train.cores[ind][i,j,k])/(nodes[j+1] - nodes[j])
        elseif j == n[2]
          G[ind][i,j,k] = (train.cores[ind][i,j,k] - train.cores[ind][i,j-1,k])/(nodes[j] - nodes[j-1])
        else
          G[ind][i,j,k] = (train.cores[ind][i,j+1,k] - train.cores[ind][i,j-1,k])/(nodes[j+1] - nodes[j-1])
        end
      end
    end
  end

  return BaseTensorTrain(G,train.ranks,0)

end

function TTgradient(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::Array{T,1}) where {T<:AbstractFloat,d}

  point_index = [findfirst(x->x==point[i],nodes[i]) for i = 1:d]

  grad = [TTevaluate(TTderivative(train,nodes[i],i),point_index) for i in 1:d]

  return grad

end

function TThessian(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::Array{T,1}) where {T<:AbstractFloat,d}

  point_index = [findfirst(x->x==point[i],nodes[i]) for i = 1:d]

  hess = [TTevaluate(TTderivative(TTderivative(train,nodes[i],i),nodes[j],j),point_index) for i in 1:d, j in 1:d]

  return hess

end

"""
Minimizes a discrete tensor train using Newton's method.

Signatures
==========

new_point, point_index, f_new_point, iters = TTnewton(train,nodes,point,tol,maxiters)
new_point, point_index, f_new_point, iters = TTnewton(train,nodes,state,point,tol,maxiters)
"""
function TTnewton(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::AbstractArray{R,1},tol::T,maxiters::S) where {R<:Real,T<:AbstractFloat,S<:Integer,d}

  new_point = copy(point)
  iters = 0
  len = Inf

  point_index     = [findmin(abs.(new_point[i] .- nodes[i]))[2] for i = 1:d]
  new_point_index = similar(point_index)

  while true

    grad = [TTevaluate(TTderivative(train,nodes[i],i),point_index) for i in 1:d]
    hess = [TTevaluate(TTderivative(TTderivative(train,nodes[i],i),nodes[j],j),point_index) for i in 1:d, j in 1:d]

    new_point .= point .- hess\grad

    for i in 1:d
      new_point_index[i] = findmin(abs.(new_point[i] .- nodes[i]))[2]
    end

    len          = maximum(abs,new_point_index-point_index)
    point_index .= new_point_index
    
    iters += 1

    if len <= tol || iters >= maxiters
      break
    end

  end

  new_point = [nodes[i][point_index[i]] for i in 1:d]

  return new_point, point_index, TTevaluate(train,point_index), iters

end

function TTnewton(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},state::AbstractArray{T,1},point::AbstractArray{R,1},tol::T,maxiters::S) where {R<:Real,T<:AbstractFloat,S<:Integer,d}

  ds = length(state)
  dx = length(point)
  if ds+dx != d
    error()
  end

  state_index = [findfirst(x->x==state[i],nodes[i]) for i = 1:ds]

  g = deepcopy(train.cores)

  new_g = Array{Array{T,3},1}(undef,dx)

  for i in dx:-1:2
    new_g[i] = g[ds+i]
  end
  temp = g[1][:,state_index[1],:]
  for i in 2:ds
    temp *= g[i][:,state_index[i],:]
  end
  new_g[1] = times_dim_1(temp,g[ds+1])

  condensed_train = BaseTensorTrain(new_g,[1;train.ranks[ds+1:end]],0)

  point, point_index, f_point, iters = TTnewton(condensed_train,nodes[ds+1:end],point,tol,maxiters)

  return point, point_index, f_point, iters

end

"""
Minimizes a continuous tensor train using Newton's method.

Signature
=========

new_point, point_index, f_new_point, iters = TTnewton(train,nodes,point,tol,maxiters)
"""
function TTnewton(train::ContinuousTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::AbstractArray{R,1},domain::Array{T,2},tol::T,maxiters::S) where {R<:Real,T<:AbstractFloat,S<:Integer,d}

  new_point = copy(point)
  iters = 0
  len = Inf

  point_index     = [findfirst(x->x==point[i],nodes[i]) for i = 1:d]
  new_point_index = similar(point_index)

  while true

    grad = TTgradient(train,point,domain)
    hess = TThessian(train,point,domain)

    new_point .= point .- hess\grad

    for i in 1:d
      new_point_index[i] = findmin(abs.(new_point[i] .- nodes[i]))[2]
    end

    len          = maximum(abs,new_point_index-point_index)
    point_index .= new_point_index
    
    iters += 1

    if len <= tol || iters >= maxiters
      break
    end

  end

  new_point = [nodes[i][point_index[i]] for i in 1:d]

  return new_point, point_index, TTevaluate(train,point_index), iters

end

"""
Perfoms one step of Newton's method on a discrete tensor train.

Signatures
==========

new_point, point_index, f_new_point, iters = TTnewton_step(train,nodes,point,tol,maxiters)
new_point, point_index, f_new_point, iters = TTnewton_stap(train,nodes,state,point,tol,maxiters)
"""
function TTnewton_step(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::AbstractArray{R,1}) where {R<:Real,T<:AbstractFloat,d}

  new_point   = copy(point)
  point_index = [findmin(abs.(new_point[i] .- nodes[i]))[2] for i = 1:d]

  grad = [TTevaluate(TTderivative(train,nodes[i],i),point_index) for i in 1:d]
  hess = [TTevaluate(TTderivative(TTderivative(train,nodes[i],i),nodes[j],j),point_index) for i in 1:d, j in 1:d]

  new_point .= point .- hess\grad

  for i in 1:d
    point_index[i] = findmin(abs.(new_point[i] .- nodes[i]))[2]
  end

  new_point = [nodes[i][point_index[i]] for i in 1:d]

  return new_point, point_index, TTevaluate(train,point_index)

end

function TTnewton_step(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},state::AbstractArray{T,1},point::AbstractArray{R,1}) where {R<:Real,T<:AbstractFloat,d}

  ds = length(state)
  dx = length(point)
  if ds+dx != d
    error()
  end

  state_index = [findfirst(x->x==state[i],nodes[i]) for i = 1:ds]

  g = deepcopy(train.cores)

  new_g = Array{Array{T,3},1}(undef,dx)

  for i in dx:-1:2
    new_g[i] = g[ds+i]
  end
  temp = g[1][:,state_index[1],:]
  for i in 2:ds
    temp *= g[i][:,state_index[i],:]
  end
  new_g[1] = times_dim_1(temp,g[ds+1])

  condensed_train = BaseTensorTrain(new_g,[1;train.ranks[ds+1:end]],0)

  point, point_index, f_point = TTnewton_step(condensed_train,nodes[ds+1:end],point)

  return point, point_index, f_point

end

"""
Minimizes a discrete tensor train using gradient descent.

Signatures
==========

new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point,α)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point,α,tol)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point,α,tol,maxiters)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,state,point)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,state,point,α)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,state,point,α,tol)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,state,point,α,tol,maxiters)
"""
function TTdescent(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::AbstractArray{R,1},α::T=0.001,tol::T=1e-8,maxiters::S=100) where {R<:Real,T<:AbstractFloat,S<:Integer,d}

  new_point = copy(point)
  iters = 0
  len = Inf

  point_index     = [findmin(abs.(new_point[i] .- nodes[i]))[2] for i = 1:d]
  new_point_index = similar(point_index)

  while true

    grad = [TTevaluate(TTderivative(train,nodes[i],i),point_index) for i in 1:d]

    new_point .= point .- α*grad

    for i in 1:d
      new_point_index[i] = findmin(abs.(new_point[i] .- nodes[i]))[2]
    end

    len          = maximum(abs,new_point_index-point_index)
    point_index .= new_point_index
    
    iters += 1

    if len <= tol || iters >= maxiters
      break
    end

  end

  new_point = [nodes[i][point_index[i]] for i in 1:d]

  return new_point, point_index, TTevaluate(train,point_index), iters

end

function TTdescent(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},state::AbstractArray{T,1},point::AbstractArray{R,1},α::T=0.001,tol::T=1e-8,maxiters::S=100) where {R<:Real,T<:AbstractFloat,S<:Integer,d}

  ds = length(state)
  dx = length(point)
  if ds+dx != d
    error()
  end

  state_index = [findfirst(x->x==state[i],nodes[i]) for i = 1:ds]

  g = deepcopy(train.cores)

  new_g = Array{Array{T,3},1}(undef,dx)

  for i in dx:-1:2
    new_g[i] = g[ds+i]
  end
  temp = g[1][:,state_index[1],:]
  for i in 2:ds
    temp *= g[i][:,state_index[i],:]
  end
  new_g[1] = times_dim_1(temp,g[ds+1])

  condensed_train = BaseTensorTrain(new_g,[1;train.ranks[ds+1:end]],0)

  point, point_index, f_point, iters = TTdescent(condensed_train,nodes[ds+1:end],point,α,tol,maxiters)

  return point, point_index, f_point, iters

end

"""
Minimizes a continuous tensor train using gradient descent.

Signatures
==========

new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point,domain)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point,domain,α)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point,domain,α,tol)
new_point, point_index, f_new_point, iters = TTdescent(train,nodes,point,domain,α,tol,maxiters)
"""
function TTdescent(train::ContinuousTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::AbstractArray{R,1},domain::AbstractArray{T,2},α::T=0.001,tol::T=1e-8,maxiters::S=100) where {R<:Real,T<:AbstractFloat,S<:Integer,d}

  new_point = copy(point)
  iters = 0
  len = Inf

  point_index     = [findfirst(x->x==point[i],nodes[i]) for i = 1:d]
  new_point_index = similar(point_index)

  while true

    grad = TTgradient(train,point,domain)

    new_point .= point .- α*grad

    for i in 1:d
      new_point_index[i] = findmin(abs.(new_point[i] .- nodes[i]))[2]
    end

    len          = maximum(abs,new_point_index-point_index)
    point_index .= new_point_index
    
    iters += 1

    if len <= tol || iters >= maxiters
      break
    end

  end

  new_point = [nodes[i][point_index[i]] for i in 1:d]

  return new_point, point_index, TTevaluate(train,point_index), iters

end

"""
Perfoms one step of gradient descent on a discrete tensor train.

Signatures
==========

new_point, point_index, f_new_point, iters = TTdescent_step(train,nodes,point)
new_point, point_index, f_new_point, iters = TTdescent_step(train,nodes,point,α)
new_point, point_index, f_new_point, iters = TTdescent_stap(train,nodes,state,point)
new_point, point_index, f_new_point, iters = TTdescent_stap(train,nodes,state,point,α)
"""
function TTdescent_step(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},point::AbstractArray{R,1},α::T=0.001) where {R<:Real,T<:AbstractFloat,d}

  new_point   = copy(point)
  point_index = [findmin(abs.(new_point[i] .- nodes[i]))[2] for i = 1:d]

  grad = [TTevaluate(TTderivative(train,nodes[i],i),point_index) for i in 1:d]

  new_point .= point .- α*grad

  for i in 1:d
    point_index[i] = findmin(abs.(new_point[i] .- nodes[i]))[2]
  end

  new_point = [nodes[i][point_index[i]] for i in 1:d]

  return new_point, point_index, TTevaluate(train,point_index)

end

function TTdescent_step(train::DiscreteTensorTrain,nodes::NTuple{d,AbstractArray{T,1}},state::AbstractArray{T,1},point::AbstractArray{R,1},α::T=0.001) where {R<:Real,T<:AbstractFloat,d}

  ds = length(state)
  dx = length(point)
  if ds+dx != d
    error()
  end

  state_index = [findfirst(x->x==state[i],nodes[i]) for i = 1:ds]

  g = deepcopy(train.cores)

  new_g = Array{Array{T,3},1}(undef,dx)

  for i in dx:-1:2
    new_g[i] = g[ds+i]
  end
  temp = g[1][:,state_index[1],:]
  for i in 2:ds
    temp *= g[i][:,state_index[i],:]
  end
  new_g[1] = times_dim_1(temp,g[ds+1])

  condensed_train = BaseTensorTrain(new_g,[1;train.ranks[ds+1:end]],0)

  point, point_index, f_point = TTdescent_step(condensed_train,nodes[ds+1:end],point,α)

  return point, point_index, f_point

end

"""
Finds the maximum element in an extended tensor train.

Signature
=========

xstar, ind_xstar, f_xstar = TTOpt(train,nodes)
"""
function TTOpt(train::ExtendedTensorTrain,nodes::NTuple{d,AbstractArray{T,1}}) where {T<:AbstractFloat,d} # Motivated by Dolgov and Savostyanov (2025)

  f_opt = -Inf
  point_index_opt = Array{Int,1}(undef,d)

  for i = 1:d-1
    for x in train.left_to_right_sub[i]
      for y in train.right_to_left_sub[i]
        point_index = (x...,y...)
        f_cand = TTevaluate(train,point_index)
        if f_cand > f_opt
          f_opt = f_cand
          point_index_opt .= point_index
        end
      end
    end
  end

  point_opt = [nodes[i][point_index_opt[i]] for i = 1:d]

  return point_opt, point_index_opt, f_opt

end