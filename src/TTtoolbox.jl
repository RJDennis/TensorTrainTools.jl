# Toolbox for Tensor train methods

# Structures that define and hold tensor trains

abstract type TensorTrain end
abstract type DiscreteTensorTrain <: TensorTrain end
abstract type ContinuousTensorTrain <: TensorTrain end
abstract type MixedTensorTrain <: TensorTrain end # Currently not used, placeholder

struct BaseTensorTrain{T} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct ExtendedTensorTrain{T} <: DiscreteTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{Int,1}
  left_to_right_ind::Array{Array{Int,1},1}
  right_to_left_ind::Array{Array{Int,1},1}
  left_to_right_sub::Array{Array{Tuple{Int,Vararg{Int}}},1}
  right_to_left_sub::Array{Array{Tuple{Int,Vararg{Int}}},1}
  sweeps::Int

end

struct FunctionalTensorTrain <: TensorTrain

  cores::Array{Array{Function,2},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct ChebyshevTensorTrain{T} <: ContinuousTensorTrain

  cores::Array{Array{T,2},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct LegendreTensorTrain{T} <: ContinuousTensorTrain

  cores::Array{Array{T,2},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct PiecewiseTensorTrain{T} <: ContinuousTensorTrain

  cores::Array{Array{T,3},1}
  ranks::Array{Int,1}
  sweeps::Int


end

#### Utility functions

"""
Multiply two cores of a tensor train.
"""
function times_cores(a::Array{T1,3},b::Array{T2,3}) where {T1 <: Number, T2 <: Number}

  na = size(a)
  nb = size(b)
  a = reshape(a,prod(na[1:2]),na[3])
  b = reshape(b,nb[1],prod(nb[2:3]))
  c = a*b
  c = reshape(c,na[1],na[2],nb[2],nb[3])

  return c

end

"""
Multiply a 3D array by a matrix along the first dimension of the 3D array.
"""
function times_dim_1(a::Array{T1,2},b::Array{T2,3}) where {T1 <: Number, T2 <: Number}

  na = size(a)
  nb = size(b)
  if na[2] != nb[1]
    error("Arrays are not conformable")
  end
  d = zeros(na[1],nb[2],nb[3])
  for i in 1:nb[3]
    d[:,:,i] .= a*b[:,:,i]
  end

  return d

end

"""
Multiply a 4D array by a matrix along the first dimension of the 4D array.
"""
function times_dim_1(a::Array{T1,2},b::Array{T2,4}) where {T1 <: Number, T2 <: Number}

  na = size(a)
  nb = size(b)
  if na[2] != nb[1]
    error("Arrays are not conformable")
  end
  d = zeros(na[1],nb[2],nb[3],nb[4])
  for j in 1:nb[4]
    for i in 1:nb[3]
      d[:,:,i,j] .= a*b[:,:,i,j]
    end
  end

  return d

end

"""
Multiply a matrix by a 3D array along the third dimension of the 3D array.
"""
function times_dim_3(a::Array{T1,3},b::Array{T2,2}) where {T1 <: Number, T2 <: Number}

  na = size(a)
  nb = size(b)
  if na[3] != nb[1]
    error("Arrays are not conformable")
  end
  d = zeros(na[1],na[2],nb[2])
  for i in 1:na[1]
    d[i,:,:] .= a[i,:,:]*b
  end
  
  return d

end

"""
Multiply a matrix by a 4D array along the third dimension of the 4D array.
"""
function times_dim_4(a::Array{T1,4},b::Array{T2,2}) where {T1 <: Number, T2 <: Number}

  na = size(a)
  nb = size(b)
  if na[4] != nb[1]
    error("Arrays are not conformable")
  end
  d = zeros(na[1],na[2],na[3],nb[2])
  for i in 1:na[1]
    for j in 1:na[2]
      d[i,j,:,:] .= a[i,j,:,:]*b
    end
  end

  return d

end

"""
Find the array sub-indices from a linear index.
"""
function ind2sub(i::S,dims::Tuple{S,Vararg{S}}) where {S <: Integer}

  if i < 1 || i > prod(dims)
    error("index is out of bounds.")
  end

  subs = Tuple(CartesianIndices(dims)[i])

  return subs

end

"""
Find the array sub-indices from a linear index.
"""
function ind2sub(i::Array{S,1},dims::Tuple{S,Vararg{S}}) where {S <: Integer}

  for x in i
    if x < 1 || x > prod(dims)
      error("index is out of bounds.")
    end
  end

  subs = Tuple(CartesianIndices(dims)[i])

  return subs

end

"""
Find the array sub-indices from a linear index.
"""
function ind2sub(i::S,dims::Array{S,1}) where {S <: Integer}

  if i < 1 || i > prod(dims)
    error("index is out of bounds.")
  end

  subs = Tuple(CartesianIndices(Tuple(dims))[i])

  return subs

end

"""
Find the array sub-indices from a linear index.
"""
function ind2sub(i::Array{S,1},dims::Array{S,1}) where {S <: Integer}

  for x in i
    if x < 1 || x > prod(dims)
      error("index is out of bounds.")
    end
  end

  subs = Tuple(CartesianIndices(Tuple(dims))[i])

  return subs

end

function compute_integrals(sd::T,order::S) where {T <: AbstractFloat, S <: Integer}

  integrals = Array{Float64}(undef,order+1)
  for i = 1:(order+1)
    integrals[i] = exp((sd^2*(i-1)^2)/2)
  end

  return integrals

end

function scale_weights(weights::Array{T,1},integrals::Array{T,1}) where {T <: AbstractFloat}

  scaled_weights = integrals.*weights

  return scaled_weights

end

#### Functions that initialize discrete tensor trains

"""
Create a random tensor train with spacial dimensions 'n' and tensor ranks 'r'.
"""
function TTrandom(n::NTuple{d,S},r::S,T::DataType=Float64) where {S <: Integer, d}

  r = fill(r,d+1)
  r[1] = 1
  r[d+1] = 1
  g = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    g[i] = rand(T,r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(g,r,0)

end

"""
Create a random tensor train with spacial dimensions 'n' and tensor ranks 'r'.
"""
function TTrandom(n::NTuple{d,S},r::Array{S,1},T::DataType=Float64) where {S <: Integer, d}

  g = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    g[i] = rand(T,r[i],n[i],r[i+1])
  end

  return BaseTensorTrain(g,r,0)

end

"""
Create a tensor train of a constant 'value' with spacial dimensions 'n' and tensor ranks 'r'.
"""
function TTconstant(value::T,n::NTuple{d,S},r::S) where {T <: AbstractFloat, S <: Integer, d}

  s = sign(value)
  value = abs(value)

  r = fill(r,d+1)
  r[1] = 1
  r[d+1] = 1
  g = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    g[i] = fill(value^(1/d)/r[i+1],r[i],n[i],r[i+1])
  end
  g[1] = s*g[1]

  return BaseTensorTrain(g,r,0)

end

"""
Create a tensor train of a constant 'value' with spacial dimensions 'n' and tensor ranks 'r'.
"""
function TTconstant(value::T,n::NTuple{d,S},r::Array{S,1}) where {T <: AbstractFloat,S<:Integer,d}

  s = sign(value)
  value = abs(value)

  g = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    g[i] = fill(value^(1/d)/r[i+1],r[i],n[i],r[i+1])
  end
  g[1] = s*g[1]

  return BaseTensorTrain(g,r,0)

end

### Functions that initialise continuous tensor trains

"""
Create a Chebyshev tensor train with ranks 'r', orders, 'order', and Chebyshev coefficients, 'θ'.
"""
function TTChebyshev(r::Array{S,1},order::Array{S,1},θ::Array{T,1}) where {S <: Integer, T <: AbstractFloat}

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
Create a Legendre tensor train with ranks 'r', orders, 'order', and Legendre coefficients, 'θ'.
"""
function TTLegendre(r::Array{S,1},order::Array{S,1},θ::Array{T,1}) where {S <: Integer, T <: AbstractFloat}

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

#### Functions to compute discrete tensor trains from a function or an array

"""
Compute the Frobenius norm of a d-dimensional array.
""" 
function frobenius(M::Array{T,d}) where {T <: AbstractFloat, d}

  frobenius_norm = zero(T)
  for x in M
      frobenius_norm += abs(x)^2
  end

  return sqrt(frobenius_norm)

end

"""
Compute the truncated SVD decomposition of a matrix with singular value threshold 'δ'.
"""
function tsvd(M::Array{T,2},δ::T) where {T <: AbstractFloat} # Looks at norm of singular values

  u, s, v = svd(M)

  r = 0
  len = zero(T)
  for i = length(s):-1:1
    len += s[i]^2
    if sqrt(len) >= max(δ,eps(T))
      r = i
      break
    end
  end

  if r > 0
    return u[:,1:r], s[1:r], v[:,1:r], r
  else
    return u[:,1], s[1:1], v[:,1], 1
  end

end

"""
Compute a tensor train approximation of a dense d-dimensional array.
Tensor compression based on Oseledets and Tyrtyshnikov (2010).
"""
function TTsvd(a::Array{T,d},tol::R) where {T <: AbstractFloat, R <: AbstractFloat, d}

  if d == 1
    return TensorTrain([reshape(a,1,length(a),1)],[1,1],0)
  end

  δ = (tol/sqrt(d-1))*frobenius(a)

  n = size(a)
  r = ones(Int,d+1)
  g = Array{Array{T,3},1}(undef,d)

  Nl = n[1]
  Nr = prod(n[2:d]) # Potential for integer overflow
  M = reshape(a,Nl,Nr)

  u,s,v,r[2] = tsvd(M,δ)
  g[1] = reshape(u,r[1],Nl,r[2])
  M = Diagonal(s)*v'

  for k = 2:(d-1)

      Nl = n[k]
      Nr = div(Nr,n[k])
      M = reshape(M,r[k]*Nl,Nr)

      u,s,v,r[k+1] = tsvd(M,δ)
      g[k] = reshape(u,r[k],n[k],r[k+1])
      M = Diagonal(s)*v'

  end

  g[d] = reshape(M,r[d],n[d],1)

  return BaseTensorTrain(g,r,1)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,maxsweeps::S = 10) where {T <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

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

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           a[l,j[1],j[2],k] = f(point)
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views g[i]   = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       a[k,j[1],j[2]] = f(point)
      end
    end

    a = reshape(a,r[d-1]*n[d-1],n[d]*r[d+1])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end
  
    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           a[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)
    
      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end
    
    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)
    
    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,initial::TensorTrain,maxsweeps::S = 10) where {T <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = copy(initial.left_to_right_ind)
  right_to_left_indices = copy(initial.right_to_left_ind)
  
  left_to_right_subs = copy(initial.left_to_right_sub)
  right_to_left_subs = copy(initial.right_to_left_sub)
  
  r = copy(initial.ranks)

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
            a[l,j[1],j[2],k] = f(point)
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[k,j[1],j[2],1] = f(point)
      end
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end
  
    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           a[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)
    
      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end
    
    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)
    
    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic(f::Function,nodes::NTuple{d,Array{T,1}},μ::R1,tol::R2,maxsweeps::S = 10) where {T <: AbstractFloat, R1 <: AbstractFloat, R2 <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

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

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           a[l,j[1],j[2],k] = f(point)
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       a[k,j[1],j[2],1] = f(point)
      end
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end
  
    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           a[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)
    
      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end
    
    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)
    
    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic(f::Function,nodes::NTuple{d,Array{T,1}},μ::R1,tol::R2,initial::TensorTrain,maxsweeps::S = 10) where {T <: AbstractFloat, R1 <: AbstractFloat, R2 <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = copy(initial.left_to_right_ind)
  right_to_left_indices = copy(initial.right_to_left_ind)

  left_to_right_subs = copy(initial.left_to_right_sub)
  right_to_left_subs = copy(initial.right_to_left_sub)

  r = copy(initial.ranks)

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)
  point       = Array{T,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           a[l,j[1],j[2],k] = f(point)
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j in CartesianIndices((1:n[d-1],1:n[d]))
        point_index[d-1:d] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
       a[k,j[1],j[2],1] = f(point)
      end
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end
  
    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j in CartesianIndices((1:n[i],1:n[i+1]))
            point_index[i:i+1] .= Tuple(j)
            for m = 1:d
              point[m] = nodes[m][point_index[m]]
            end
           a[l,j[1],j[2],k] = f(point)
          end
        end
      end
    
      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)
    
      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)
      
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]
    
    end
    
    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j in CartesianIndices((1:n[1],1:n[2]))
        point_index[1:2] .= Tuple(j)
        for m = 1:d
          point[m] = nodes[m][point_index[m]]
        end
        a[j[1],j[2],k] = f(point)
      end
    end
    
    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)
    
    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)
        
    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,maxsweeps::S = 10) where {T <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

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

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    @sync Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = collect((left_to_right_subs[d-2][j[1]]...,(j[2],j[3])...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     a[j[1],j[2],j[3],1] = f(p)
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T,initial::TensorTrain,maxsweeps::S = 10) where {T <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = copy(initial.left_to_right_ind)
  right_to_left_indices = copy(initial.right_to_left_ind)

  left_to_right_subs = copy(initial.left_to_right_sub)
  right_to_left_subs = copy(initial.right_to_left_sub)

  r = copy(initial.ranks)
  
  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    @sync Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = collect((left_to_right_subs[d-2][j[1]]...,(j[2],j[3])...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     a[j[1],j[2],j[3],1] = f(p)
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    if sweeps >= maxsweeps
      break
    end

    # Sweep from tight to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::R1,tol::R2,maxsweeps::S = 10) where {T <: AbstractFloat, R1 <: AbstractFloat, R2 <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  
  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

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

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    @sync Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = collect((left_to_right_subs[d-2][j[1]]...,(j[2],j[3])...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     a[j[1],j[2],j[3],1] = f(p)
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    if sweeps >= maxsweeps
      break
    end

    # Sweep from tight to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::R1,tol::R2,initial::TensorTrain,maxsweeps::S = 10) where {T <: AbstractFloat, R1 <: AbstractFloat, R2 <: AbstractFloat, S <: Integer, d}

  n = length.(nodes)

  if d == 1
    a = [f(x) for x in nodes[1]]
    return TTsvd(a,tol)
  elseif d == 2
    a = [f([nodes[1][i],nodes[2][j]]) for i in 1:n[1], j in 1:n[2]]
    return TTsvd(a,tol)
  end

  # The following is used when d ≥ 3

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = copy(initial.left_to_right_ind)
  right_to_left_indices = copy(initial.right_to_left_ind)

  left_to_right_subs = copy(initial.left_to_right_sub)
  right_to_left_subs = copy(initial.right_to_left_sub)

  r = copy(initial.ranks)
  
  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
    
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]
  
      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    @sync Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_ind = collect((left_to_right_subs[d-2][j[1]]...,(j[2],j[3])...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
     a[j[1],j[2],j[3],1] = f(p)
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, keeping the indices nested
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])
    
    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    if sweeps >= maxsweeps
      break
    end

    # Sweep from tight to left, keeping the indices nested

    # Solve for the interior indices
    
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_ind = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        p         = Array{T,1}(undef,d)
        for m = 1:d
          p[m] = nodes[m][point_ind[m]]
        end
        a[j[1],j[2],j[3],j[4]] = f(p)
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_ind = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      p         = Array{T,1}(undef,d)
      for m = 1:d
        p[m] = nodes[m][point_ind[m]]
      end
      a[j[1],j[2],j[3]] = f(p)
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array.
DMRG tensor compression based on Dolgov and Savostyanov (2020).
"""
function DMRGcross(b::Array{T,d},μ::T,tol::T,maxsweeps::S = 10) where {T <: AbstractFloat, S <: Integer, d}

  n = size(b)

  if d <= 2
    return TTsvd(b,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j = 1:n[2]
        for i = 1:n[1]
          point_index[1:2] .= (i, j)
          a[i,j,k] = b[(point_index...)]
        end
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j = 1:n[i+1]
            for ii = 1:n[i]
              point_index[i:i+1] .= (ii, j)
              a[l,ii,j,k] = b[(point_index...)]
            end
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j = 1:n[d]
        for i = 1:n[d-1]
          point_index[d-1:d] .= (i, j)
          a[k,i,j] = b[(point_index...)]
        end
      end
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j = 1:n[i+1]
            for ii = 1:n[i]
              point_index[i:i+1] .= (ii, j)
              a[l,ii,j,k] = b[(point_index...)]
            end
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j = 1:n[2]
        for i = 1:n[1]
          point_index[1:2] .= (i, j)
          a[i,j,k] = b[(point_index...)]
        end
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array.
DMRG tensor compression based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic(b::Array{T,d},μ::R1,tol::R2,maxsweeps::S = 10) where {T <: AbstractFloat, R1 <: AbstractFloat, R2 <: AbstractFloat, S <: Integer, d}

  n = size(b)

  if d <= 2
    return TTsvd(b,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j = 1:n[2]
        for i = 1:n[1]
          point_index[1:2] .= (i, j)
          a[i,j,k] = b[(point_index...)]
        end
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j = 1:n[i+1]
            for ii = 1:n[i]
              point_index[i:i+1] .= (ii, j)
              a[l,ii,j,k] = b[(point_index...)]
            end
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,3}(undef,(r[d-1],n[d-1],n[d]))
    for k = 1:r[d-1]
      point_index[1:d-2] .= left_to_right_subs[d-2][k]
      for j = 1:n[d]
        for i = 1:n[d-1]
          point_index[d-1:d] .= (i, j)
          a[k,i,j] = b[(point_index...)]
        end
      end
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      for l = 1:r[i]
        point_index[1:i-1] .= left_to_right_subs[i-1][l]
        for k = 1:r[i+2]
          point_index[i+2:end] .= right_to_left_subs[i+1][k]
          for j = 1:n[i+1]
            for ii = 1:n[i]
              point_index[i:i+1] .= (ii, j)
              a[l,ii,j,k] = b[(point_index...)]
            end
          end
        end
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    for k = 1:r[3]
      point_index[3:end] .= right_to_left_subs[2][k]
      for j = 1:n[2]
        for i = 1:n[1]
          point_index[1:2] .= (i, j)
          a[i,j,k] = b[(point_index...)]
        end
      end
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array.
DMRG tensor compression based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_threaded(b::Array{T,d},μ::T,tol::T,maxsweeps::S = 10) where {T <: AbstractFloat, S <: Integer, d}

  n = size(b)

  if d <= 2
    return TTsvd(b,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      a[j[1],j[2],j[3]] = b[(point_index...)]
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        a[j[1],j[2],j[3],j[4]] = b[(point_index...)]
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    @sync Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = collect((left_to_right_subs[d-2][j[1]]...,(j[2],j[3])...))
      a[j[1],j[2],j[3],1] = b[(point_index...)]
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        a[j[1],j[2],j[3],j[4]] = b[(point_index...)]
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol!(v,μ,100)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      a[j[1],j[2],j[3]] = b[(point_index...)]
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array.
DMRG tensor compression based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic_threaded(b::Array{T,d},μ::R1,tol::R2,maxsweeps::S = 10) where {T <: AbstractFloat, R1 <: AbstractFloat, R2 <: AbstractFloat, S <: Integer, d}

  n = size(b)

  if d <= 2
    return TTsvd(b,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2

  r = ones(S,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{S,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{S,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{S,Vararg{S}}},1}(undef,d-1)

  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i] - r[i+1], 2)
    left_to_right_indices[i] = [p+1:p+r[i+1];]

    if i == 1
    
      left_to_right_subs[i] = [tuple(ind2sub(x,(r[i],n[i]))[2]) for x in left_to_right_indices[i]]

    else

      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

    end
  end

  for i = d-1:-1:1

    p = div(n[i] - r[i+1], 2)
    right_to_left_indices[i] = [p+1:p+r[i+1];]

    if i == d-1
    
      right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices[d-1]]

    else

      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end
  
  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{S,1}(undef,d)

  sweeps = 0
  while true

    # Sweep from left to right, keeping the indices nested

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      a[j[1],j[2],j[3]] = b[(point_index...)]
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    @views g[1] = reshape(a[:,right_to_left_indices_new[1]]/a[left_to_right_indices_new[1],right_to_left_indices_new[1]],r[1],n[1],r[2])

    # Solve for the interior indices
  
    for i = 2:d-2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        a[j[1],j[2],j[3],j[4]] = b[(point_index...)]
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)

      # Update left_to_right_subs, keeping the indices nested
      temp = [ind2sub(x,(r[i],n[i])) for x in left_to_right_indices_new[i]]
      left_to_right_subs[i] = [tuple(left_to_right_subs[i-1][temp[j][1]]...,temp[j][2]...) for j in eachindex(temp)]

      @views g[i] = reshape(a[:,right_to_left_indices_new[i]]/a[left_to_right_indices_new[i],right_to_left_indices_new[i]],r[i],n[i],r[i+1])

    end

    # Solve for the final indices

    a = Array{T,4}(undef,(r[d-1],n[d-1],n[d],r[d+1]))
    @sync Threads.@threads for j in CartesianIndices((1:r[d-1],1:n[d-1],1:n[d]))
      point_index = collect((left_to_right_subs[d-2][j[1]]...,(j[2],j[3])...))
      a[j[1],j[2],j[3],1] = b[(point_index...)]
    end

    a = reshape(a,r[d-1]*n[d-1],n[d])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[d] = tsvd(a,δ)

    if r[d] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[d-1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[d-1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    temp = [ind2sub(x,(r[d-1],n[d-1])) for x in left_to_right_indices_new[d-1]]
    left_to_right_subs[d-1] = [tuple(left_to_right_subs[d-2][temp[i][1]]...,temp[i][2]...) for i in eachindex(temp)]

    # Update right_to_left_subs, nesting is not kept
    right_to_left_subs[d-1] = [tuple(ind2sub(x,(n[d],r[d+1]))[1]) for x in right_to_left_indices_new[d-1]]

    @views g[d-1] = reshape(a[:,right_to_left_indices_new[d-1]]/a[left_to_right_indices_new[d-1],right_to_left_indices_new[d-1]],r[d-1],n[d-1],r[d])
    @views g[d]   = reshape(a[left_to_right_indices_new[d-1],:],r[d],n[d],r[d+1])

    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    if sweeps >= maxsweeps
      break
    end

    # Sweep from right to left, keeping the indices nested

    # Solve for the interior indices
  
    for i = d-2:-1:2

      a = Array{T,4}(undef,(r[i],n[i],n[i+1],r[i+2]))
      @sync Threads.@threads for j in CartesianIndices((1:r[i],1:n[i],1:n[i+1],1:r[i+2]))
        point_index = collect((left_to_right_subs[i-1][j[1]]...,(j[2],j[3])...,right_to_left_subs[i+1][j[4]]...))
        a[j[1],j[2],j[3],j[4]] = b[(point_index...)]
      end

      a = reshape(a,r[i]*n[i],r[i+2]*n[i+1])
      δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
      u,s,v,r[i+1] = tsvd(a,δ)

      if r[i+1] == 1
        u = reshape(u,length(u),1)
        v = reshape(v,length(v),1)
      end
  
      left_to_right_indices_new[i], sweep = maxvol_generic!(u,μ,100)
      right_to_left_indices_new[i], sweep = maxvol_generic!(v,μ,100)
  
      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

    end

    # Solve for the first indices

    a = Array{T,3}(undef,(n[1],n[2],r[3]))
    @sync Threads.@threads for j in CartesianIndices((1:n[1],1:n[2],1:r[3]))
      point_index = collect(((j[1],j[2])...,right_to_left_subs[2][j[3]]...))
      a[j[1],j[2],j[3]] = b[(point_index...)]
    end

    a = reshape(a,n[1],n[2]*r[3])
    δ = tol*frobenius(a)#δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

    sweeps += 1
  
    if sum(isempty.(setdiff.(left_to_right_indices_new,left_to_right_indices))) == d-1 && sum(isempty.(setdiff.(right_to_left_indices_new,right_to_left_indices))) == d-1
      left_to_right_indices .= left_to_right_indices_new
      right_to_left_indices .= right_to_left_indices_new
      break
    end

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      break
    end

  end

  return ExtendedTensorTrain(g,r,left_to_right_indices,right_to_left_indices,left_to_right_subs,right_to_left_subs,sweeps)

end

#### Functions needed to transform discrete tensor trains into continuous tensor trains

#### Utility functions

"""
Normalize 'node' so that it resides within the interval [-1,1].
"""
function normalize_node(node::R, domain::Array{T,1}) where {R <: Number, T <: AbstractFloat}

  if domain[1] == domain[2]
    norm_node = zero(T)
    return norm_node
  else
    norm_node = 2*(node - domain[2])/(domain[1] - domain[2]) - 1
    return norm_node
  end

end

"""
Normalize 'node' so that all elements reside within the interval [-1,1].
"""
function normalize_node(node::Array{R,1}, domain::Array{T,1}) where {R <: Number, T <: AbstractFloat}

  norm_nodes = map(x -> normalize_node(x, domain), node)
  return norm_nodes

end

#### Functions for working with Chebyshev nodes/polynomials

"""
Create a vector of 'n' Chebyshev nodes on 'domain' with element type 'T'.
"""
function chebyshev_nodes(n::S,domain=[1.0,-1.0],T::DataType=Float64) where {S <: Integer}

  points = fill((domain[1] + domain[2]) * T(0.5), n)

  @inbounds for i = 1:div(n, 2)
    x = -cos(T(i - 0.5) * π / n) * (domain[1] - domain[2]) * T(0.5)
    points[i] += x
    points[n-i+1] -= x
  end

  return points

end

"""
Generate Chebyshev polynomials of 'order' at 'point'.
"""
function chebyshev_polynomial(order::S, point::R) where {S <: Integer, R <: Number}

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

"""
Generate Chebyshev polynomials of 'order' at each element of 'points'.
"""
function chebyshev_polynomial(order::S, points::Array{R,1}) where {S <: Integer, R <: Number}

  poly = Array{R}(undef, length(points), order + 1)
  poly[:,1] .= ones(R, length(points))

  @inbounds for i = 2:order+1
    for j in eachindex(points)
      if i == 2
        poly[j,i] = points[j]
      else
        poly[j,i] = 2*points[j]*poly[j,i-1] - poly[j,i-2]
      end
    end
  end

  return poly

end
"""
Construct the derivative of a Chebyshev polynomial at 'point'.
"""
function chebyshev_polynomial_deriv(order::S, point::R) where {S <: Integer, R <: Number}

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

"""
Construct the derivative of a Chebyshev polynomial at each element of 'points'.
"""
function chebyshev_polynomial_deriv(order::S, points::Array{R,1}) where {S <: Integer, R <: Number}

  poly_deriv = Array{R}(undef,order+1,length(points))
  poly_deriv[1, :] .= zeros(R,length(points))

  @inbounds for j in eachindex(points)
    p = one(R)
    pl = NaN
    pll = NaN
    for i = 2:order+1
      if i == 2
        pl, p = p, x[j]
        poly_deriv[i,j] = one(R)
      else
        pll, pl = pl, p
        p = 2*xpoints[j]*pl - pll
        poly_deriv[i,j] = 2*pl+2*points[j]*poly_deriv[i-1,j] - poly_deriv[i-2,j]
      end
    end
  end

  return Matrix(transpose(poly_deriv))

end

"""
Compute the weights for a univariate Chebyshev polynomial with 'order' on 'domain'.
"""
function chebyshev_weights(y::Array{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer}

  normalized_nodes = normalize_node(nodes,domain)
  P = chebyshev_polynomial(order,normalized_nodes)

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
Evaluate a Chebyshev polynomial at 'point'.
"""
function chebyshev_evaluate(w::Array{T,1},point::Q,order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, Q <: Number, S <: Integer}

  normalized_point = normalize_node(point,domain)
  P = chebyshev_polynomial(order,normalized_point)

  yhat = zero(T)
  @inbounds for i in CartesianIndices(w)
    yhat += w[i]*P[i]
  end

  return yhat

end

"""
Create a function that evaluates a Chebyshev polynomial at 'point'.
"""
function chebyshev_interp(y::Array{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer}

  w = chebyshev_weights(y,nodes,order,domain)

  function cheb_interp(point)

     yhat = chebyshev_evaluate(w, point, order, domain)

    return yhat

  end

  return cheb_interp

end

"""
Create a Chebyshev tensor train from a discrete tensor train.
"""
function createCTT(g::DiscreteTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    order = size(g.cores[i],2)-1

    f[i] = [chebyshev_weights(g.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return ChebyshevTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a Chebyshev tensor train from a discrete tensor train.
"""
function createCTT(g::DiscreteTensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    f[i] = [chebyshev_weights(g.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return ChebyshevTensorTrain(f,g.ranks,g.sweeps)

end

#### Functions for working with Legendre nodes/polynomials

"""
Create a vector of 'n' Legendre nodes on 'domain' with element type 'T'.
"""
function legendre_nodes(n::S,domain=[1.0,-1.0],T::DataType=Float64) where {S <: Integer}

  λ, Q = eigen(SymTridiagonal(zeros(T,n), [i / sqrt(T(4i^2 - 1)) for i = 1:n-1]))
  nodes = (λ .+ 1) * (domain[1] - domain[2]) / 2 .+ domain[2]

  return nodes

end

"""
Generate Legendre polynomials of 'order' at 'point'.
"""
function legendre_polynomial(order::S, point::R) where {S <: Integer, R <: Number}

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

"""
Generate Legendre polynomials of 'order' at each element of 'points'.
"""
function legendre_polynomial(order::S, points::Array{R,1}) where {S <: Integer, R <: Number}

  poly = Array{R}(undef, length(points), order + 1)
  poly[:, 1] .= ones(R, length(points))

  @inbounds for i = 2:order+1
    for j in eachindex(points)
      if i == 2
        poly[j,i] = points[j]
      else
        poly[j,i] = ((2*i-1)/i)*points[j]*poly[j,i-1] - ((i-1)/i)*poly[j,i-2]
      end
    end
  end

  return poly

end

"""
Construct the derivative of a Legendre polynomial at 'point'.
"""
function legendre_polynomial_deriv(order::S, point::R) where {S <: Integer, R <: Number}

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

"""
Construct the derivative of a Legendre polynomial at each element of 'points'.
"""
function legendre_polynomial_deriv(order::S, points::Array{R,1}) where {S <: Integer, R <: Number}

  n = length(x)

  poly       = Array{R}(undef,n,order+1)
  poly_deriv = Array{R}(undef,n,order+1)
  poly[:,1]  .= one(R)
  poly_deriv[:,1] .= zero(R)

  @inbounds for j = 1:n
    for i = 2:order+1
      if i == 2
        poly[j,i] = points[j]
        poly_deriv[j,i] = one(R)
      else
        poly[j,i] = ((2*i-1)/i)*points[j]*poly[j,i-1] - ((i-1)/i)*poly[j,i-2]
        poly_deriv[j,i] = i*(points[j]*poly[j,i]-poly[j,i-1])/(points[j]^2-1)
      end
    end
  end

  return poly_deriv

end

"""
Compute the weights for a univariate Legendre polynomial with 'order' on 'domain'.
"""
function legendre_weights(y::Array{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer}

  normalized_nodes = normalize_node(nodes,domain)
  P = legendre_polynomial(order,normalized_nodes)

  weights = (P'P)\(P'y)

  return weights

end

"""
Evaluate a Legendre polynomial at 'point'.
"""
function legendre_evaluate(w::Array{T,1},point::Q,order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, Q <: Number, S <: Integer}

  normalized_point = normalize_node(point,domain)
  P = legendre_polynomial(order,normalized_point)

  yhat = zero(T)
  @inbounds for i in CartesianIndices(w)
    yhat += w[i]*P[i]
  end

  return yhat

end

"""
Create a function that evaluates a Legendre polynomial at 'point'.
"""
function legendre_interp(y::Array{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer}

  w = legendre_weights(y,nodes,order,domain)

  function legend_interp(point)

    yhat = legendre_evaluate(w, point, order, domain)
  
    return yhat
    
  end

  return legend_interp

end

"""
Create a Legendre tensor train from a discrete tensor train.
"""
function createLTT(g::DiscreteTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    order = size(g.cores[i],2)-1

    f[i] = [legendre_weights(g.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return LegendreTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a Legendre tensor train from a discrete tensor train.
"""
function createLTT(g::DiscreteTensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    f[i] = [legendre_weights(g.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return LegendreTensorTrain(f,g.ranks,g.sweeps)

end

#### Functions for working with piecewise linear nodes

function piecewise_linear_nodes(n::S,domain = [1.0,-1.0],T::DataType=Float64) where {S <: Integer}

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

function bracket_nodes(x::Array{T,1},point::R) where {T <: AbstractFloat, R <: Number}

  n = length(x)

  if real(point) <= x[1] # Real is used because complex numbers are occasionally used in NLboxsolve.jl
    return (1,2)
  elseif real(point) >= x[end]
    return (n-1,n)
  else
    y = 0
    for i in x
      if i < real(point)
        y += 1
      else
        break
      end
    end
    return (y,y+1)
  end

end

function piecewise_linear_weight(x::Array{T,1},point::R) where {T <: AbstractFloat, R <: Number}

  bracketing_nodes = bracket_nodes(x,point)

  weight = (point-x[bracketing_nodes[1]])/(x[bracketing_nodes[2]]-x[bracketing_nodes[1]])

  return weight

end

function piecewise_linear_evaluate(y::Array{T1,1},x::Array{T2,1},point::R) where {T1 <: AbstractFloat, T2 <: AbstractFloat, R <: Number}

  b = bracket_nodes(x,point)
  w = piecewise_linear_weight(x,point)

  y_estimate = y[b[1]] + w*(y[b[2]]-y[b[1]])

  return y_estimate

end

function piecewise_linear_evaluate(y::Array{T1,1},nodes::Array{T2,1}) where {T1 <: AbstractFloat, T2 <: AbstractFloat}

function approximating_function(point::R) where {R <: Number}

    return piecewise_linear_evaluate(y,nodes,point)

end

return approximating_function

end

"""
Create a piecewise linear tensor train from a discrete tensor train.
"""
function createPTT(g::DiscreteTensorTrain) # Creates a piecewise linear tensor train from a tensor train

  return PiecewiseTensorTrain(g.cores,g.ranks,g.sweeps)

end

#### Functions for creating functional tensor trains

"""
Create a functional tensor train from a discrete tensor train, assuming the nodes are Chebyshev roots.
"""
function createFTT(g::DiscreteTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    order = size(g.cores[i],2)-1

    f[i] = [chebyshev_interp(g.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a discrete tensor train, assuming the nodes are Chebyshev roots.
"""
function createFTT(g::DiscreteTensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{T,2}) where {T <: AbstractFloat, S <: Integer, d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    f[i] = [chebyshev_interp(g.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a Chebyshev tensor train.
"""
function createFTT(g::ChebyshevTensorTrain,domain::Array{T,2}) where {T <: AbstractFloat}

  d = length(g.cores)

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    order = length(g.cores[i][1])-1

    f[i] = [interp(x) = chebyshev_evaluate(g.cores[i][j,k],x,order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a Chebyshev tensor train.
"""
function createFTT(g::ChebyshevTensorTrain,order::NTuple{d,S},domain::Array{T,2}) where {T <: AbstractFloat, S <: Integer, d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    f[i] = [interp(x) = chebyshev_evaluate(g.cores[i][j,k],x,order[i],domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a Legendre tensor train.
"""
function createFTT(g::LegendreTensorTrain,domain::Array{T,2}) where {T <: AbstractFloat}

  d = length(g.cores)

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    order = length(g.cores[i][1])-1

    f[i] = [interp(x) = legendre_evaluate(g.cores[i][j,k],x,order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a Legendre tensor train.
"""
function createFTT(g::LegendreTensorTrain,order::NTuple{d,S},domain::Array{T,2}) where {T <: AbstractFloat, S <: Integer, d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    f[i] = [interp(x) = legendre_evaluate(g.cores[i][j,k],x,order[i],domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a piecewise tensor train.
"""
function createFTT(g::PiecewiseTensorTrain,nodes::NTuple{d,Array{T,1}}) where {T <: AbstractFloat, d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    #f[i] = [interp = piecewise_linear_evaluate(g.cores[i][j,:,k],nodes[i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]
    f[i] = [piecewise_linear_evaluate(g.cores[i][j,:,k],nodes[i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create an interpolating function from a functional tensor train.
"""
function TTinterp(train::FunctionalTensorTrain)

  function TTinterp(point::Array{T,1}) where {T <: Number}

    d = length(train.cores)

    a = reshape([train.cores[1][k](point[1]) for k in eachindex(train.cores[1])],train.ranks[1],train.ranks[2])
    for j = 2:d
      a *= reshape([train.cores[j][k](point[j]) for k in eachindex(train.cores[j])],train.ranks[j],train.ranks[j+1])
    end

    return a[1]

  end

  return TTinterp

end

#### Functions for evaluating discrete tensor trains and continuous tensor trains

"""
Recreates the dense array from a discrete tensor train.
"""
function decompress(train::DiscreteTensorTrain)

  T = eltype(train.cores[1])
  d = length(train.cores)

  n = Tuple([size(train.cores[i])[2] for i = 1:d])

  a = Array{T,d}(undef,n)
  for i in CartesianIndices(a)
    temp = train.cores[1][:,i[1],:]
    for j = 2:d
      temp *= train.cores[j][:,i[j],:]
    end
    a[i] = temp[1]
  end

  return a

end

"""
Recreates the dense array from a discrete tensor train.
"""
function decompress_threaded(train::DiscreteTensorTrain)

  T = eltype(train.cores[1])
  d = length(train.cores)

  n = Tuple([size(train.cores[i])[2] for i = 1:d])

  a = Array{T,d}(undef,n)
  @sync Threads.@threads for i in CartesianIndices(a)
    temp = train.cores[1][:,i[1],:]
    for j = 2:d
      temp *= train.cores[j][:,i[j],:]
    end
    a[i] = temp[1]
  end

  return a

end

"""
Evaluate a discrete tensor train at an index of the dense array.
"""
function TTevaluate(train::DiscreteTensorTrain,index::Array{S,1}) where { S <: Integer}

  d = length(train.cores)

  a = train.cores[1][:,index[1],:]
  for j = 2:d
    a *= train.cores[j][:,index[j],:]
  end

  return a[1]

end

"""
Evaluate a discrete tensor train at a vector of indexes of the dense array.
"""
function TTevaluate(train::DiscreteTensorTrain,index::Array{Array{S,1},1}) where {S <: Integer}

  T = eltype(train.cores[1])
  d = length(train.cores)

  N = length(index)

  y = zeros(T,N)

  for i in 1:N
    a = train.cores[1][:,index[i][1],:]
    for j = 2:d
      a *= train.cores[j][:,index[i][j],:]
    end

    y[i] = a[1]

  end

  return y

end

"""
Evaluate a functional tensor train at 'point'.
"""
function TTevaluate(train::FunctionalTensorTrain,point::Array{T,1}) where {T <: AbstractFloat}

  d = length(train.cores)

  a = reshape([train.cores[1][k](point[1]) for k in eachindex(train.cores[1])],train.ranks[1],train.ranks[2])
  for j = 2:d
    a *= reshape([train.cores[j][k](point[j]) for k in eachindex(train.cores[j])],train.ranks[j],train.ranks[j+1])
  end

  return a[1]

end

"""
Evaluate a functional tensor train at a vector of 'points'.
"""
function TTevaluate(train::FunctionalTensorTrain,points::Array{Array{T,1}}) where {T <: AbstractFloat}

  d = length(train.cores)

  N = length(points)

  y = zeros(N)

  for i in 1:N
    a = reshape([train.cores[1][k](points[i][1]) for k in eachindex(train.cores[1])],train.ranks[1],train.ranks[2])
    for j = 2:d
      a *= reshape([train.cores[j][k](points[i][j]) for k in eachindex(train.cores[j])],train.ranks[j],train.ranks[j+1])
    end

    y[i] = a[1]

  end

  return y

end

#### Functions to compute derivatives and gradients of continuous tensor trains

"""
Compute the derivative of a Chebyshev tensor train with respect to a variable.
"""
function TTderivative(train::ChebyshevTensorTrain,x::Array{R,1},pos::S,domain::Array{T,2}) where {R <: Number,T <: AbstractFloat, S <: Integer}

  d = length(train.cores)
  poly = Array{Array{R,2},1}(undef, d)
  @inbounds for i = 1:d
    order = length(train.cores[i][1])-1
    if i === pos
      poly[i] = chebyshev_polynomial_deriv(order, normalize_node(x[i], domain[:, i]))
    else
      poly[i] = chebyshev_polynomial(order, normalize_node(x[i], domain[:, i]))
    end
  end

  a = [(poly[1]*train.cores[1][i,k])[1] for i in 1:train.ranks[1], k in 1:train.ranks[2]]
  @inbounds for j = 2:d
    a *= [(poly[j]*train.cores[j][i,k])[1] for i in 1:train.ranks[j], k in 1:train.ranks[j+1]]
  end

  return a[1][1] * (2.0 / (domain[1, pos] - domain[2, pos]))

end

"""
Compute the derivative of a Legendre tensor train with respect to a variable.
"""
function TTderivative(train::LegendreTensorTrain,x::Array{R,1},pos::S,domain::Array{T,2}) where {R <: Number,T <: AbstractFloat, S <: Integer}

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

  a = [(poly[1]*train.cores[1][i,k])[1] for i in 1:train.ranks[1], k in 1:train.ranks[2]]
  @inbounds for j = 2:d
    a *= [(poly[j]*train.cores[j][i,k])[1] for i in 1:train.ranks[j], k in 1:train.ranks[j+1]]
  end

  return a[1][1] * (2.0 / (domain[1, pos] - domain[2, pos]))

end

"""
Compute the gradient vector of a Chebyshev tensor train.
"""
function TTgradient(train::ChebyshevTensorTrain,x::Array{R,1},domain::Array{T,2}) where {T <: AbstractFloat, R <: Number}

  d = length(train.cores)

  gradient = Array{R,2}(undef,1,d)

  @inbounds for i = 1:d
    gradient[i] = TTderivative(train,x,i,domain)
  end

  return gradient

end

"""
Compute the gradient vector of a Legendre tensor train.
"""
function TTgradient(train::LegendreTensorTrain,x::Array{R,1},domain::Array{T,2}) where {T <: AbstractFloat, R <: Number}

  d = length(train.cores)

  gradient = Array{R,2}(undef,1,d)

  @inbounds for i = 1:d
    gradient[i] = TTderivative(train,x,i,domain)
  end

  return gradient

end

#### Functions to integrate discrete tensor trains and functional tensor trains

"""
Integrate a discrete tensor train using Gauss-Chebyshev quadrature.
"""
function TTintegrate_GC(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R <: AbstractFloat} # Integrates over all dimensions

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

"""
Integrate a discrete tensor train using Gauss-Legendre quadrature.
"""
function TTintegrate_GL(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R <: AbstractFloat} # Integrates over all dimensions

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

"""
Integrate a discrete tensor train using Gauss-Hermite quadrature.
"""
function TTintegrate_GH(train::DiscreteTensorTrain) # Integrates over all dimensions

  T = eltype(train.cores[1])
  d = length(train.cores)

  integral = fill(T(1.0),1,1)
  for i = 1:d
    n = size(train.cores[i])
    term = zeros(n[1],n[3])
    node, weights = hermite(n[2])
    for j = 1:n[2]
      term += train.cores[i][:,j,:]*exp(node[j]^2.0)*weights[j]
    end
    integral *= term
  end

  return integral[1]

end

"""
Integrate a discrete tensor train using the trapazoidal method.
"""
function TTintegrate_PL(train::DiscreteTensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R <: AbstractFloat} # Integrates over all dimensions

  T = eltype(train.cores[1])
  d = length(train.cores)

  integral = fill(T(1.0),1,1)
  for i = 1:d
    term = trapazoidal(train.cores[i],domain[:,i])
    integral *= term
  end

  return integral[1]

end

"""
Integrate a functional tensor train using Gauss-Chebyshev quadrature.
"""
function TTintegrate_GC(train::FunctionalTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Union{Array{R,2},Array{R,1}}) where {T <: AbstractFloat, R <: AbstractFloat, d} # Integrates over all dimensions

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
Integrate a functional tensor train using Gauss-Legendre quadrature.
"""
function TTintegrate_GC(train::FunctionalTensorTrain,nodes::NTuple{d,Array{T,1}},domain::Union{Array{R,2},Array{R,1}}) where {T <: AbstractFloat, R <: AbstractFloat, d} # Integrates over all dimensions

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
Integrate a functional tensor train using Gauss-Hermite quadrature.
"""
function TTintegrate_GH(train::FunctionalTensorTrain,nodes::NTuple{d,Array{T,1}}) where {T <: AbstractFloat,d} # Integrates over all dimensions

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
Integrate a discrete tensor train over all dimensions except 'ind' using Gauss-Hermite quadrature.
"""
function compute_marginal_GC(train::TensorTrain, ind::S) where {S<:Integer} # Assumes Gauss-Hermite quadrature

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
Integrate a discrete tensor train over all dimensions except 'ind' using Gauss-Chebyshev quadrature.
"""
function compute_marginal_GC(train::TensorTrain, ind::S, domain::Array{T,2}) where {T<:AbstractFloat,S<:Integer} # Assumes Gauss-Chebyshev quadrature

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
"""
function compute_marginal_GL(train::TensorTrain, ind::S, domain::Array{T,2}) where {T<:AbstractFloat,S<:Integer} # Assumes Gauss-Legendre quadrature

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
Integrate a discrete tensor train over all dimensions except 'ind' using the trapazoidal method.
"""
function compute_marginal_PL(train::TensorTrain, ind::S, domain::Array{T,2}) where {T<:AbstractFloat,S<:Integer} # Assumes trapazoidal integration 

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
"""
function TTrounding(train::BaseTensorTrain,tol::T) where {T <: AbstractFloat}

  d = length(train.cores)
  r = copy(train.ranks)
  n = [size(train.cores[i])[2] for i in 1:d]

  cores = copy(train.cores)

  r_new = copy(r)

  for i = 1:d-1

    a = reshape(cores[i],r_new[i]*n[i],r_new[i+1])
    δ = tol*frobenius(a)
    u,s,v,r_new[i+1] = tsvd(a,δ)
    cores[i] = reshape(u,r_new[i],n[i],r_new[i+1])
    cores[i+1] = times_dim_1(Matrix(transpose(v*Diagonal(s))),cores[i+1])

  end

  return BaseTensorTrain(cores,r_new,train.sweeps)

end

"""
Perform rounding on a discrete tensor train to find a compact discrete tensor train representation.
"""
function TTrounding(train::ExtendedTensorTrain,tol::T) where {T <: AbstractFloat}

  d = length(train.cores)
  r = copy(train.ranks)
  n = [size(train.cores[i])[2] for i in 1:d]

  cores = copy(train.cores)

  r_new = copy(r)

  for i = 1:d-1

    a = reshape(cores[i],r_new[i]*n[i],r_new[i+1])
    δ = tol*frobenius(a)
    u,s,v,r_new[i+1] = tsvd(a,δ)
    cores[i] = reshape(u,r_new[i],n[i],r_new[i+1])
    cores[i+1] = times_dim_1(Matrix(transpose(v*Diagonal(s))),cores[i+1])

  end

  return ExtendedTensorTrain(cores,r_new,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Multiply a discrete tensor train and a scalar.
"""
function TTmult(train::BaseTensorTrain,s::T) where {T <: AbstractFloat}

  cores = copy(train.cores)
  cores[1] = cores[1]*s

  return BaseTensorTrain(cores,train.ranks,train.sweeps)

end

"""
Multiply a discrete tensor train and a scalar.
"""
function TTmult(train::ExtendedTensorTrain,s::T) where {T <: AbstractFloat}

  cores = copy(train.cores)
  cores[1] = cores[1]*s

  return ExtendedTensorTrain(cores,train.ranks,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Multiply a scalar and a discrete tensor train.
"""
function TTmult(s::T,train::DiscreteTensorTrain) where {T <: AbstractFloat}

  return TTmult(train,s)

end

"""
Add two discrete tensor trains.
"""
function TTadd(traina::DiscreteTensorTrain,trainb::DiscreteTensorTrain)

  da = length(traina.cores)
  db = length(trainb.cores)

  if da != db
    error("Trains have different number of cores")
  end

  na = [size(traina.cores[i])[2] for i in 1:da]
  nb = [size(trainb.cores[i])[2] for i in 1:db]

  for i in eachindex(na)
    if na[i] != nb[i]
      error("Trains have different numbers of nodes")
    end
  end

  coresa = copy(traina.cores)
  ra     = copy(traina.ranks)

  coresb = copy(trainb.cores)
  rb     = copy(trainb.ranks)

  cores = Array{Array{Float64,3},1}(undef,da)
  r_new = copy(ra)

  for i = 1:da

    if i == 1
      g = [reshape(coresa[1],ra[1],na[1]*ra[2]) reshape(coresb[1],rb[1],nb[1]*rb[2])]
      cores[1] = reshape(g,ra[i],na[1],(ra[2]+rb[2]))
      r_new[2] = ra[2]+rb[2]
    elseif i == da
      g = [reshape(coresa[da],ra[da],na[da]*ra[da+1]); reshape(coresb[db],rb[db],nb[db]*rb[db+1])]
      cores[da] = reshape(g,ra[da]+rb[db],na[da],ra[da+1])
    else
      g = zeros((ra[i]+rb[i]),na[i]*(ra[i+1]+rb[i+1]))
      g[1:ra[i],1:(na[i]*ra[i+1])]         .= reshape(coresa[i],ra[i],na[i]*ra[i+1])
      g[ra[i]+1:end,(na[i]*ra[i+1])+1:end] .= reshape(coresb[i],rb[i],nb[i]*rb[i+1])
      cores[i] = reshape(g,(ra[i]+rb[i]),na[i],(ra[i+1]+rb[i+1]))
      r_new[i+1] = ra[i+1]+rb[i+1]
    end

  end

  return BaseTensorTrain(cores,r_new,0)

end

"""
Subtract two discrete tensor trains.
"""
function TTsubtract(traina::DiscreteTensorTrain,trainb::DiscreteTensorTrain)

  da = length(traina.cores)
  db = length(trainb.cores)

  if da != db
    error("Trains have different number of cores")
  end

  na = [size(traina.cores[i])[2] for i in 1:da]
  nb = [size(trainb.cores[i])[2] for i in 1:db]

  for i in eachindex(na)
    if na[i] != nb[i]
      error("Trains have different numbers of nodes")
    end
  end

  coresa = copy(traina.cores)
  ra     = copy(traina.ranks)

  coresb = copy(trainb.cores)
  rb     = copy(trainb.ranks)

  coresb[1] = -coresb[1]

  cores = Array{Array{Float64,3},1}(undef,da)
  r_new = copy(ra)

  for i = 1:da

    if i == 1
      g = [reshape(coresa[1],ra[1],na[1]*ra[2]) reshape(coresb[1],rb[1],nb[1]*rb[2])]
      cores[1] = reshape(g,ra[i],na[1],(ra[2]+rb[2]))
      r_new[2] = ra[2]+rb[2]
    elseif i == da
      g = [reshape(coresa[da],ra[da],na[da]*ra[da+1]); reshape(coresb[db],rb[db],nb[db]*rb[db+1])]
      cores[da] = reshape(g,ra[da]+rb[db],na[da],ra[da+1])
    else
      g = zeros((ra[i]+rb[i]),na[i]*(ra[i+1]+rb[i+1]))
      g[1:ra[i],1:(na[i]*ra[i+1])]         .= reshape(coresa[i],ra[i],na[i]*ra[i+1])
      g[ra[i]+1:end,(na[i]*ra[i+1])+1:end] .= reshape(coresb[i],rb[i],nb[i]*rb[i+1])
      cores[i] = reshape(g,(ra[i]+rb[i]),na[i],(ra[i+1]+rb[i+1]))
      r_new[i+1] = ra[i+1]+rb[i+1]
    end

  end

  return BaseTensorTrain(cores,r_new,0)

end

#### Functions to orthogonalise discrete tensor trains

"""
Perform the RQ matrix decomposition.
"""
function rq(A::Array{T,2}) where {T <: Real}

  F = lq(reverse(A, dims=1))
  R, Q = reverse(F.L), reverse(Matrix(F.Q), dims=1)

  return R, Q

end

"""
Right orthogonalise a discrete tensor train.
"""
function TTorthright(train::L) where {L <: BaseTensorTrain}  #Tensor train orthogonalisation according to Chertkov, Ryzhakov, Novikov, and Oseledets (2022), algorthm 3.

  Π = copy(train.cores)
  d = length(Π)
  n = [size(Π[k])[2] for k = 1:d]
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

  return BaseTensorTrain(Π,train.ranks,train.sweeps)

end

"""
Right orthogonalise a discrete tensor train.
"""
function TTorthright(train::L) where {L <: ExtendedTensorTrain}

  Π = copy(train.cores)
  d = length(Π)
  n = [size(Π[k])[2] for k = 1:d]
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

  return ExtendedTensorTrain(Π,train.ranks,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Left orthogonalise a discrete tensor train.
"""
function TTorthleft(train::L) where {L <: BaseTensorTrain}

  Π = copy(train.cores)
  d = length(Π)
  n = [size(Π[k])[2] for k = 1:d]
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

  return BaseTensorTrain(Π,train.ranks,train.sweeps)

end

"""
Left orthogonalise a discrete tensor train.
"""
function TTorthleft(train::L) where {L <: ExtendedTensorTrain}

  Π = copy(train.cores)
  d = length(Π)
  n = [size(Π[k])[2] for k = 1:d]
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

  return ExtendedTensorTrain(Π,train.ranks,train.left_to_right_ind,train.right_to_left_ind,train.left_to_right_sub,train.right_to_left_sub,train.sweeps)

end

"""
Left orthogonalise a train up to core 'μ' and right orthogonalise the remaining cores.
"""
function TTorthleftright(train::L,μ::S) where {L <: BaseTensorTrain, S <: Integer}

  Π = copy(train.cores)
  d = length(Π)
  n = [size(Π[k])[2] for k = 1:d]
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

"""
Left orthogonalise a train up to core 'μ' and right orthogonalise the remaining cores.
"""
function TTorthleftright(train::L,μ::S) where {L <: ExtendedTensorTrain, S <: Integer}

  Π = copy(train.cores)
  d = length(Π)
  n = [size(Π[k])[2] for k = 1:d]
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
"""
function TTnorm(train::DiscreteTensorTrain)

  try
    orthtrain = TTorthright(train)
    return norm(orthtrain.cores[1])
  catch
    orthtrain = TTorthleft(train)
    return norm(orthtrain.cores[d])
  end

end

"""
Use bisection to find the fix-point of the continuous function 'f'.
"""
function bisection(f::Function,x::Array{T1,1},tol::T2,maxiters::S) where {T1 <: AbstractFloat, T2 <: AbstractFloat, S <: Integer}

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
"""
function trapazoidal(y::Array{T1,3},domain::Array{T2,1}) where {T1 <: AbstractFloat, T2 <: AbstractFloat}

  n = size(y)

  integral_f = y[:,begin,:]+y[:,end,:]
  for i = 2:(n[2]-1)
    integral_f += 2*y[:,i,:]
  end

  integral_f = (1/(n[2]-1))*((domain[1]-domain[2])/2)*integral_f

  return integral_f

end

"""
Take a draw from a density with unbounded domain by inverting the CDF.
"""
function invert_cdf(f::Function,u::T) where {T <: AbstractFloat}

  g(x) = f(x)-u

  c, fc, its = bisection(g,[floatmax(T),floatmin(T)],eps(),5_000)

  return c

end

"""
Take a draw from a density with finite domain by inverting the CDF.
"""
function invert_cdf(f::Function,u::T,lb::T,ub::T) where {T <: AbstractFloat}

  g(x) = f(x)-u

  c, fc, its = bisection(g,[ub,lb],eps(),5_000)

  return c

end

#### Tensor train conditional distribution sampling based on Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020), Statistics and Computing

"""
Conditional distribution sampling when the nodes are Gauss-Hermite.
"""
function TTCD_GH(train::DiscreteTensorTrain,N::S,seed::S = 123456) where {S <: Integer} # Gauss-Hermite quadrature used to marginalise

  d = length(train.cores)
  Π = copy(train.cores)
  n = [size(Π[i]) for i in 1:d]

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(N,d)

  P = Array{Array{Float64,2},1}(undef,d+1)
  Φ = Array{Array{Float64,2},1}(undef,d+1)
  Ψ = Array{Array{Float64,1},1}(undef,d)

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

  Φ[1] = ones(N,n[1][2])
  for k = 1:d
    ϕ = zeros(N,n[k][2])
    Ψ[k] = [(Π[k][:,i,:]*P[k+1])[1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l,:].*Ψ[k])
      g = piecewise_linear_evaluate(p,nodes[k])
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      ϕ[l,:] = Φ[k][l,:]*g(sample[l,k])
    end
    Φ[k+1] = ϕ
  end

  return sample

end

"""
Conditional distribution sampling when the nodes are Gauss-Chebyshev.
"""
function TTCD_GC(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T <: AbstractFloat, S <: Integer} # Gauss-Chebyshev quadrature used to marginalise

  d = length(train.cores)
  Π = copy(train.cores)
  n = [size(Π[i]) for i in 1:d]

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(T,N,d)

  P = Array{Array{T,2},1}(undef,d+1)
  Φ = Array{Array{T,2},1}(undef,d+1)
  Ψ = Array{Array{T,1},1}(undef,d)

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

  Φ[1] = ones(N,n[1][2])
  for k = 1:d
    ϕ = zeros(N,n[k][2])
    Ψ[k] = [(Π[k][:,i,:]*P[k+1])[1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l,:].*Ψ[k])
      g = piecewise_linear_evaluate(p,nodes[k])
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      ϕ[l,:] = Φ[k][l,:]*g(sample[l,k])
    end
    Φ[k+1] = ϕ
  end

  return sample

end

"""
Conditional distribution sampling when the nodes are Gauss-Legendre.
"""
function TTCD_GL(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T <: AbstractFloat, S <: Integer} # Gauss-Legendre quadrature used to marginalise

  d = length(train.cores)
  Π = copy(train.cores)
  n = [size(Π[i]) for i in 1:d]

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(T,N,d)

  P = Array{Array{T,2},1}(undef,d+1)
  Φ = Array{Array{T,2},1}(undef,d+1)
  Ψ = Array{Array{T,1},1}(undef,d)

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

  Φ[1] = ones(N,n[1][2])
  for k = 1:d
    ϕ = zeros(N,n[k][2])
    Ψ[k] = [(Π[k][:,i,:]*P[k+1])[1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l,:].*Ψ[k])
      g = piecewise_linear_evaluate(p,nodes[k])
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      ϕ[l,:] = Φ[k][l,:]*g(sample[l,k])
    end
    Φ[k+1] = ϕ
  end

  return sample

end

"""
Conditional distribution sampling when the nodes are uniformly spaced.
"""
function TTCD_PL(train::DiscreteTensorTrain,N::S,domain::Array{T,2},seed::S = 123456) where {T <: AbstractFloat, S <: Integer} # Trapazopidal integration used to marginalise

  d = length(train.cores)
  Π = copy(train.cores)
  n = [size(Π[i]) for i in 1:d]

  rng = MersenneTwister(seed)
  q = rand(rng,N,d)
  sample = zeros(T,N,d)

  P = Array{Array{T,2},1}(undef,d+1)
  Φ = Array{Array{T,2},1}(undef,d+1)
  Ψ = Array{Array{T,1},1}(undef,d)

  nodes   = Array{Array{T,1},1}(undef,d)

  P[d+1] = [1.0;;]
  for k = d:-1:1
    nodes[k] = piecewise_linear_nodes(n[k][2],domain[:,k])
    term = trapazoidal(Π[k],nodes[k])
    P[k] = term*P[k+1]
  end

  Φ[1] = ones(N,n[1][2])
  for k = 1:d
    ϕ = zeros(N,n[k][2])
    Ψ[k] = [(Π[k][:,i,:]*P[k+1])[1] for i in 1:n[k][2]]
    for l = 1:N
      p = abs.(Φ[k][l,:].*Ψ[k])
      g = piecewise_linear_evaluate(p,nodes[k])
      p .= cumsum(p)
      p .= p./p[end]
      f = piecewise_linear_evaluate(p,nodes[k])
      sample[l,k] = invert_cdf(f,q[l,k],nodes[k][begin],nodes[k][end])
      ϕ[l,:] = Φ[k][l,:]*g(sample[l,k])
    end
    Φ[k+1] = ϕ
  end

  return sample

end

# Tensor train Metropolis-Hastings correction based on Dolgov, Anaya-Izquierdo, Fox, and Scheichl (2020), Statistics and Computing

# This function hasn't been thoroughly tested yet

function TTMH(logdensity::Function,logtrain::FunctionalTensorTrain,sample::Array{T,2},seed::S=123456) where {T <: AbstractFloat, S <: Integer}

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

#### Functions to optimize over a discrete tensor train

"""
Find the 'K' rows of a matrix with the highest norm.
"""
function topK(M::Array{T,2},K::S) where {T <: AbstractFloat, S <: Integer}

  n = size(M)

  m = zeros(n[1])
  for i in eachindex(m)
    m[i] = norm(M[i,:])
  end
  p = sortperm(m,rev=true)

  return p[1:min(n[1],K)]

end

"""
Horozontally concatenate two matrices.
"""
function stack(a::Array{T,2},b::Array{T,2}) where {T <: AbstractFloat}

  return [a b]

end

# Tensor train extremization according to Chertkov, Ryzhakov, Novikov, and Oseledets (2022), algorthm 1.

function TTextremize(train::L,K::S) where {L <: DiscreteTensorTrain, S <: Integer} # Output could be maxima or minima

  orth_train = TTorthright(train)  
  Π = copy(orth_train.cores)
  
  d = length(Π)
  r = copy(orth_train.ranks)
  n = [size(Π[i])[2] for i = 1:d]

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

# Extremize over the final (d-ds) cores, evaluating the first ds cores at the indices given in state

function TTextremize(train::L,K::S,state::Array{S,1}) where {L <: DiscreteTensorTrain, S <: Integer} # Output could be maxima or minima

  orth_train = TTorthright(train)  
  Π = copy(orth_train.cores)
  
  d = length(Π)
  ds = length(state)
  r = copy(orth_train.ranks)
  n = [size(Π[i])[2] for i = 1:d]

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
  I = [1:1:n[1];;]
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

# Tensor train optimization according to Chertkov, Ryzhakov, Novikov, and Oseledets (2022), algorthm 2.

function TToptimize(train::L,K::S) where {L <: DiscreteTensorTrain, S <: Integer}

  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple([size(train.cores[i])[2] for i in 1:d])

  imax = TTextremize(train,K)
  ymax = TTevaluate(train,imax)

  c = TTconstant(ymax,n,r)

  difference = TTsubtract(train,c) # TTrounding could be used here, but requires a tolerance

  imin = TTextremize(difference,K)
  ymin = TTevaluate(train,imin)

  if ymax >= ymin

    return (imin, imax), (ymin, ymax)

  else

    return (imax, imin), (ymax, ymin)

  end

end

# Optimizes over the final (d-ds) cores, evaluating the first ds cores at the indices given in state

function TToptimize(train::L,K::S,state::Array{S,1}) where {L <: DiscreteTensorTrain, S <: Integer}

  d = length(train.cores)
  r = copy(train.ranks)
  n = Tuple([size(train.cores[i])[2] for i in 1:d])

  imax = TTextremize(train,K,state)
  ymax = TTevaluate(train,[state;imax])

  c = TTconstant(ymax,n,r)

  difference = TTsubtract(train,c) # TTrounding could be used here, but requires a tolerance

  imin = TTextremize(difference,K,state)
  ymin = TTevaluate(train,[state;imin])

  if ymax >= ymin

    return (imin, imax), (ymin, ymax)

  else

    return (imax, imin), (ymax, ymin)

  end

end