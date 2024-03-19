# Toolbox for Tensor train methods

struct TensorTrain{T}

  cores::Array{Array{T,3},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct FunctionalTensorTrain

  cores::Array{Array{Function,2},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct ChebyshevTensorTrain{T} 

  cores::Array{Array{T,2},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct LegendreTensorTrain{T}

  cores::Array{Array{T,2},1}
  ranks::Array{Int,1}
  sweeps::Int

end

struct PiecewiseTensorTrain{T}

  cores::Array{Array{T,3},1}
  ranks::Array{Int,1}
  sweeps::Int

end

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
Create a vector of 'n' Legendre nodes on 'domain' with element type 'T'.
"""
function legendre_nodes(n::S,domain=[1.0,-1.0],T::DataType=Float64) where {S <: Integer}

  λ, Q = eigen(SymTridiagonal(zeros(T,n), [i / sqrt(T(4i^2 - 1)) for i = 1:n-1]))
  nodes = (λ .+ 1) * (domain[1] - domain[2]) / 2 .+ domain[2]

  return nodes

end

"""
Create a vector of 'n' uniformly spaced nodes on 'domain' with element type 'T'.
"""
function piecewise_linear_nodes(n::S,domain = [1.0,-1.0],T::DataType=Float64) where {S <: Integer}

  if n <= 0
    error("The number of nodes must be positive.")
  end

  nodes = T.([(domain[1]+domain[2])/2.0 for _ in 1:n])

  if isodd(n)
    inc = T.((domain[1]-domain[2])/(n-1))
  else
    inc = T.((domain[1]-domain[2])/n)
  end
  @inbounds for i = 1:div(n,2)
    nodes[i]     += (i-1-div(n,2))*inc
    nodes[n-i+1] -= (i-1-div(n,2))*inc
  end

  return nodes

end

"""
Normalize 'node' so that it resides within [1,-1].
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
Normalize 'node' so that all elements reside within [1,-1].
"""
function normalize_node(node::Array{R,1}, domain::Array{T,1}) where {R <: Number, T <: AbstractFloat}

  norm_nodes = map(x -> normalize_node(x, domain), node)
  return norm_nodes

end

"""
Generate Chebyshev polynomials of 'order' at 'node'.
"""
function chebyshev_polynomial(order::S, node::R) where {S <: Integer, R <: Number}

  poly = Array{R}(undef, 1, order + 1)
  poly[1] = one(R)

  @inbounds for i = 2:order+1
    if i == 2
      poly[i] = node
    else
      poly[i] = 2*node*poly[i-1] - poly[i-2]
    end
  end

  return poly

end

"""
Generate Chebyshev polynomials of 'order' at each element of 'nodes'.
"""
function chebyshev_polynomial(order::S, nodes::Array{R,1}) where {S <: Integer, R <: Number}

  poly = Array{R}(undef, length(nodes), order + 1)
  poly[:,1] .= ones(R, length(nodes))

  @inbounds for i = 2:order+1
    for j in eachindex(nodes)
      if i == 2
        poly[j,i] = nodes[j]
      else
        poly[j,i] = 2*nodes[j]*poly[j,i-1] - poly[j,i-2]
      end
    end
  end

  return poly

end

"""
Generate Legendre polynomials of 'order' at 'node'.
"""
function legendre_polynomial(order::S, node::R) where {S <: Integer, R <: Number}

  poly = Array{R}(undef, 1, order + 1)
  poly[1] = one(R)

  @inbounds for i = 2:order+1
    if i == 2
      poly[i] = node
    else
      poly[i] = ((2*i-1)/i)*node*poly[i-1] - ((i-1)/i)*poly[i-2]
    end
  end

  return poly

end

"""
Generate Legendre polynomials of 'order' at each element of 'nodes'.
"""
function legendre_polynomial(order::S, nodes::Array{R,1}) where {S <: Integer, R <: Number}

  poly = Array{R}(undef, length(nodes), order + 1)
  poly[:, 1] .= ones(R, length(nodes))

  @inbounds for i = 2:order+1
    for j in eachindex(nodes)
      if i == 2
        poly[j,i] = nodes[j]
      else
        poly[j,i] = ((2*i-1)/i)*nodes[j]*poly[j,i-1] - ((i-1)/i)*poly[j,i-2]
      end
    end
  end

  return poly

end

"""
Construct the derivative of a Chebyshev polynomial at 'x'.
"""
function chebyshev_polynomial_deriv(order::S, x::R) where {S <: Integer, R <: Number}

  poly_deriv = Array{R}(undef, 1, order + 1)
  poly_deriv[1] = zero(R)
  p = one(R)
  pl = NaN
  pll = NaN

  @inbounds for i = 2:order+1
    if i == 2
      pl, p = p, x
      poly_deriv[i] = one(R)
    else
      pll, pl = pl, p
      p = 2 * x * pl - pll
      poly_deriv[i] = 2 * pl + 2 * x * poly_deriv[i-1] - poly_deriv[i-2]
    end
  end

  return poly_deriv

end

"""
Construct the derivative of a Chebyshev polynomial at each element of 'x'.
"""
function chebyshev_polynomial_deriv(order::S, x::Array{R,1}) where {S <: Integer, R <: Number}

  poly_deriv = Array{R}(undef, order + 1, length(x))
  poly_deriv[1, :] .= zeros(R, length(x))

  @inbounds for j in eachindex(x)
    p = one(R)
    pl = NaN
    pll = NaN
    for i = 2:order+1
      if i == 2
        pl, p = p, x[j]
        poly_deriv[i, j] = one(R)
      else
        pll, pl = pl, p
        p = 2 * x[j] * pl - pll
        poly_deriv[i, j] = 2 * pl + 2 * x[j] * poly_deriv[i-1, j] - poly_deriv[i-2, j]
      end
    end
  end

  return Matrix(transpose(poly_deriv))

end

"""
Construct the derivative of a Legendre polynomial at 'x'.
"""
function legendre_polynomial_deriv(order::S, x::R) where {S <: Integer, R <: Number}

  poly       = Array{R}(undef, 1, order + 1)
  poly_deriv = Array{R}(undef, 1, order + 1)
  poly[1] = one(R)
  poly_deriv[1] = zero(R)

  @inbounds for i = 2:order+1
    if i == 2
      poly[i] = x
      poly_deriv[i] = one(R)
    else
      poly[i] = ((2*i-1)/i)*x*poly[i-1] - ((i-1)/i)*poly[i-2]
      poly_deriv[i] = i*(x*poly[i]-poly[i-1])/(x^2-1)
    end
  end

  return poly_deriv

end

"""
Construct the derivative of a Legendre polynomial at each element of 'x'.
"""
function legendre_polynomial_deriv(order::S, x::Array{R,1}) where {S <: Integer, R <: Number}

  n = length(x)

  poly       = Array{R}(undef, n, order + 1)
  poly_deriv = Array{R}(undef, n, order + 1)
  poly[:,1]  .= one(R)
  poly_deriv[:,1] .= zero(R)

  @inbounds for j = 1:n
    for i = 2:order+1
      if i == 2
        poly[j,i] = x[j]
        poly_deriv[j,i] = one(R)
      else
        poly[j,i] = ((2*i-1)/i)*x[j]*poly[j,i-1] - ((i-1)/i)*poly[j,i-2]
        poly_deriv[j,i] = i*(x[j]*poly[j,i]-poly[j,i-1])/(x[j]^2-1)
      end
    end
  end

  return poly_deriv

end

"""
Compute the weights for a Chebyshev polynomial.
"""
function chebyshev_weights(y::Array{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer}

  N = length(nodes)
  normalized_nodes = normalize_node(nodes,domain)
  P = chebyshev_polynomial(order,normalized_nodes)

  weights = Array{T,1}(undef,order+1)
  @inbounds for i in CartesianIndices(weights)

    numerator = zero(T)
    denominator = zero(T)

    @inbounds for s in CartesianIndices(y)

      numerator   += y[s] * P[s,i]
      denominator += P[s,i]^2

    end

    weights[i] = numerator / denominator

  end

  return weights

end

"""
Compute the weights for a Legendre polynomial.
"""
function legendre_weights(y::Array{T,1},nodes::Array{T,1},order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer}

  normalized_nodes = normalize_node(nodes,domain)
  P = legendre_polynomial(order,normalized_nodes)

  weights = Array{T,1}(undef,order+1)
  @inbounds for i in CartesianIndices(weights)

    numerator = zero(T)
    denominator = zero(T)

    @inbounds for s in CartesianIndices(y)

      numerator   += y[s] * P[s,i]
      denominator += P[s,i]^2

    end

    weights[i] = numerator / denominator

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
    yhat += w[i] * P[i]
  end

  return yhat

end

"""
Evaluate a Legendre polynomial at 'point'.
"""
function legendre_evaluate(w::Array{T,1},point::Q,order::S,domain::Array{R,1}) where {T <: AbstractFloat, R <: AbstractFloat, Q <: Number, S <: Integer}

  normalized_point = normalize_node(point,domain)
  P = legendre_polynomial(order,normalized_point)

  yhat = zero(T)
  @inbounds for i in CartesianIndices(w)
    yhat += w[i] * P[i]
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
Create a random tensor train with spacial dimensions 'n' and tensor ranks 'r'.
"""
function randomTT(n::NTuple{d,S},r::S,T::DataType=Float64) where {S <: Integer, d}

  r = fill(r,d+1)
  r[1] = 1
  r[d+1] = 1
  g = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    g[i] = rand(T,r[i],n[i],r[i+1])
  end

  return TensorTrain(g,r,0)

end

"""
Create a random tensor train with spacial dimensions 'n' and tensor ranks 'r'.
"""
function randomTT(n::NTuple{d,S},r::Array{S,1},T::DataType=Float64) where {S <: Integer, d}

  g = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    g[i] = rand(T,r[i],n[i],r[i+1])
  end

  return TensorTrain(g,r,0)

end

"""
Create a tensor train of a constant 'value' with spacial dimensions 'n' and tensor ranks 'r'.
"""
function constantTT(value::T,n::NTuple{d,S},r::S) where {T <: AbstractFloat, S <: Integer, d}

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

  return TensorTrain(g,r,0)

end

"""
Create a tensor train of a constant 'value' with spacial dimensions 'n' and tensor ranks 'r'.
"""
function constantTT(value::T,n::NTuple{d,S},r::Array{S,1}) where {T <: AbstractFloat,S<:Integer,d}

  s = sign(value)
  value = abs(value)

  g = Array{Array{T,3},1}(undef,d)
  for i = 1:d
    g[i] = fill(value^(1/d)/r[i+1],r[i],n[i],r[i+1])
  end
  g[1] = s*g[1]

  return TensorTrain(g,r,0)

end

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
Compute the truncated SVD decomposition of a matrix.
"""
function tsvd(M::Array{T,2},δ::T) where {T <: AbstractFloat} # Looks at norm of singular values

  u, s, v = svd(M)

  r = 0
  len = zero(T)
  for i = length(s):-1:1
    len += s[i]^2
    if sqrt(len) >= δ
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

  if ndims(a) == 1
    return TensorTrain([reshape(a,1,length(a),1)],[1,1],0)
  end

  if d == 1
    δ = tol*frobenius(a)
  else
    δ = tol/(sqrt(d-1))*frobenius(a)
  end

  n = size(a)
  r = ones(Int,d+1)
  g = Array{Array{T,3},1}(undef,d)

  Nl = n[1]
  Nr = prod(n[2:d])
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

  return TensorTrain(g,r,1)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T;maxsweeps::S = 50) where {T <: AbstractFloat, S <: Integer, d}

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
  #rinit = minimum((n...,rinit))

  r = ones(Int,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{Int,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{Int,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  
  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i]+1-(r[i+1]-1),2)
    #left_to_right_indices[i] = [1:r[i+1];]
    left_to_right_indices[i] = [p:p+r[i+1];]
    #right_to_left_indices[i] = [1:r[i+1];]
    right_to_left_indices[i] = [p:p+r[i+1];]

    left_to_right_subs[i] = [tuple(j,fill(1,i-1)...) for j in 1:r[i+1]]
    right_to_left_subs[i] = [tuple(j,fill(1,d-1-i)...) for j in 1:r[i+1]]

  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{Int64,1}(undef,d)
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
    δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

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
      δ = (tol/sqrt(d-1))*frobenius(a)
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
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

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
    δ = (tol/sqrt(d-1))*frobenius(a)
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

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      break
    end

  end

  return TensorTrain(g,r,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic(f::Function,nodes::NTuple{d,Array{T,1}},μ::R,tol::R;maxsweeps::S = 50) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer, d}

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
  #rinit = minimum((n...,rinit))

  r = ones(Int,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{Int,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{Int,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  
  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i]+1-(r[i+1]-1),2)
    #left_to_right_indices[i] = [1:r[i+1];]
    left_to_right_indices[i] = [p:p+r[i+1];]
    #right_to_left_indices[i] = [1:r[i+1];]
    right_to_left_indices[i] = [p:p+r[i+1];]

    left_to_right_subs[i] = [tuple(j,fill(1,i-1)...) for j in 1:r[i+1]]
    right_to_left_subs[i] = [tuple(j,fill(1,d-1-i)...) for j in 1:r[i+1]]

  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{Int64,1}(undef,d)
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
    δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

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
      δ = (tol/sqrt(d-1))*frobenius(a)
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
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

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
    δ = (tol/sqrt(d-1))*frobenius(a)
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

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      break
    end

  end

  return TensorTrain(g,r,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array based on the function that populates the array.
DMRG function approximation based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_threaded(f::Function,nodes::NTuple{d,Array{T,1}},μ::T,tol::T;maxsweeps::S = 50) where {T <: AbstractFloat, S <: Integer, d}

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
  #rinit = minimum((n...,rinit))

  r = ones(Int,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{Int,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{Int,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  
  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i]+1-(r[i+1]-1),2)
    #left_to_right_indices[i] = [1:r[i+1];]
    left_to_right_indices[i] = [p:p+r[i+1];]
    #right_to_left_indices[i] = [1:r[i+1];]
    right_to_left_indices[i] = [p:p+r[i+1];]

    left_to_right_subs[i] = [tuple(j,fill(1,i-1)...) for j in 1:r[i+1]]
    right_to_left_subs[i] = [tuple(j,fill(1,d-1-i)...) for j in 1:r[i+1]]

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
    δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    # Update right_to_left_subs, keeping the indices nested
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

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
      δ = (tol/sqrt(d-1))*frobenius(a)
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
  
      # Update right_to_left_subs, keeping the indices nested
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

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
    δ = (tol/sqrt(d-1))*frobenius(a)
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

    left_to_right_indices .= left_to_right_indices_new
    right_to_left_indices .= right_to_left_indices_new

    if sweeps >= maxsweeps
      break
    end

  end

  return TensorTrain(g,r,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array.
DMRG tensor compression based on Dolgov and Savostyanov (2020).
"""
function DMRGcross(b::Array{T,d},μ::T,tol::T;maxsweeps::S = 50) where {T <: AbstractFloat, S <: Integer, d}

  n = size(b)

  if d <= 2
    return TTsvd(b,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2
  #rinit = minimum((n...,rinit))

  r = ones(Int,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{Int,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{Int,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)

  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i]+1-(r[i+1]-1),2)
    #left_to_right_indices[i] = [1:r[i+1];]
    left_to_right_indices[i] = [p:p+r[i+1];]
    #right_to_left_indices[i] = [1:r[i+1];]
    right_to_left_indices[i] = [p:p+r[i+1];]

    left_to_right_subs[i] = [tuple(j,fill(1,i-1)...) for j in 1:r[i+1]]
    right_to_left_subs[i] = [tuple(j,fill(1,d-1-i)...) for j in 1:r[i+1]]

  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{Int64,1}(undef,d)

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
    δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

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
      δ = (tol/sqrt(d-1))*frobenius(a)
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

      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

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
    δ = (tol/sqrt(d-1))*frobenius(a)
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

  end

  return TensorTrain(g,r,sweeps)

end

"""
Compute a tensor train approximation of a dense d-dimensional array.
DMRG tensor compression based on Dolgov and Savostyanov (2020).
"""
function DMRGcross_generic(b::Array{T,d},μ::R,tol::R;maxsweeps::S = 50) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer, d}

  n = size(b)

  if d <= 2
    return TTsvd(b,tol)
  end

  # The following is used when d ≥ 3

  rinit = 2
  #rinit = minimum((n...,rinit))

  r = ones(Int,d+1)
  for i = 2:d
    r[i] = min(n[i-1],n[i],rinit)
  end

  g = Array{Array{T,3},1}(undef,d)

  left_to_right_indices = Array{Array{Int,1},1}(undef,d-1)
  right_to_left_indices = Array{Array{Int,1},1}(undef,d-1)

  left_to_right_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)
  right_to_left_subs = Array{Array{Tuple{Int,Vararg{Int}}},1}(undef,d-1)

  # The initial linear indices are arbitrary
  # The initial sub-indices are constructed from the linear indices.

  for i = 1:d-1

    p = div(n[i]+1-(r[i+1]-1),2)
    #left_to_right_indices[i] = [1:r[i+1];]
    left_to_right_indices[i] = [p:p+r[i+1];]
    #right_to_left_indices[i] = [1:r[i+1];]
    right_to_left_indices[i] = [p:p+r[i+1];]

    left_to_right_subs[i] = [tuple(j,fill(1,i-1)...) for j in 1:r[i+1]]
    right_to_left_subs[i] = [tuple(j,fill(1,d-1-i)...) for j in 1:r[i+1]]

  end

  left_to_right_indices_new = similar(left_to_right_indices)
  right_to_left_indices_new = similar(right_to_left_indices)

  point_index = Array{Int64,1}(undef,d)

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
    δ = (tol/sqrt(d-1))*frobenius(a)
    u,s,v,r[2] = tsvd(a,δ)

    if r[2] == 1
      u = reshape(u,length(u),1)
      v = reshape(v,length(v),1)
    end

    left_to_right_indices_new[1], sweep = maxvol_generic!(u,μ,100)
    right_to_left_indices_new[1], sweep = maxvol_generic!(v,μ,100)

    # Update left_to_right_subs, keeping the indices nested
    left_to_right_subs[1] = [tuple(ind2sub(x,(r[1],n[1]))[2]) for x in left_to_right_indices_new[1]]

    # Update right_to_left_subs, nesting is not kept
    temp = [ind2sub(x,(n[2],r[3])) for x in right_to_left_indices_new[1]]
    right_to_left_subs[1] = [tuple(temp[j][1]...,right_to_left_subs[2][temp[j][2]]...) for j in eachindex(temp)]

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
      δ = (tol/sqrt(d-1))*frobenius(a)
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

      # Update right_to_left_subs, nesting is not kept
      temp = [ind2sub(x,(n[i+1],r[i+2])) for x in right_to_left_indices_new[i]]
      right_to_left_subs[i] = [tuple(temp[j][1]...,right_to_left_subs[i+1][temp[j][2]]...) for j in eachindex(temp)]

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
    δ = (tol/sqrt(d-1))*frobenius(a)
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

  end

  return TensorTrain(g,r,sweeps)

end

"""
Recreates the dense array from a tensor train.
"""
function decompress(train::TensorTrain)

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
Recreates the dense array from a tensor train.
"""
function decompress_threaded(train::TensorTrain)

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
Evaluate a tensor train at an index point of the dense array.
"""
function TTevaluate(train::TensorTrain,index::Array{S,1}) where { S <: Integer}

  d = length(train.cores)

  a = train.cores[1][:,index[1],:]
  for j = 2:d
    a *= train.cores[j][:,index[j],:]
  end

  return a[1]

end

"""
Evaluate a tensor train at a vector of index points of the dense array.
"""
function TTevaluate(train::TensorTrain,index::Array{Array{S,1},1}) where {S <: Integer}

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

"""
Create a Chebyshev tensor train from a tensor train.
"""
function createCTT(g::TensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    order = size(g.cores[i],2)-1

    f[i] = [chebyshev_weights(g.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return ChebyshevTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a Chebyshev tensor train from a tensor train.
"""
function createCTT(g::TensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    f[i] = [chebyshev_weights(g.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return ChebyshevTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a Legendre tensor train from a tensor train.
"""
function createLTT(g::TensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    order = size(g.cores[i],2)-1

    f[i] = [legendre_weights(g.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return LegendreTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a Legendre tensor train from a tensor train.
"""
function createLTT(g::TensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, S <: Integer, d}

  f = Array{Array{Array{T,1},2},1}(undef,d)

  for i = 1:d

    f[i] = [legendre_weights(g.cores[i][j,:,k],nodes[i],order[i],domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return LegendreTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a piecewise linear tensor train from a tensor train.
"""
function createPTT(g::TensorTrain) # Creates a piecewise linear tensor train from a tensor train

  return PiecewiseTensorTrain(g.cores,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a tensor train, assuming the nodes are Chebyshev roots.
"""
function createFTT(g::TensorTrain,nodes::NTuple{d,Array{T,1}},domain::Array{R,2}) where {T <: AbstractFloat, R <: AbstractFloat, d}

  f = Array{Array{Function,2},1}(undef,d)

  for i = 1:d

    order = size(g.cores[i],2)-1

    f[i] = [chebyshev_interp(g.cores[i][j,:,k],nodes[i],order,domain[:,i]) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

  end

  return FunctionalTensorTrain(f,g.ranks,g.sweeps)

end

"""
Create a functional tensor train from a tensor train, assuming the nodes are Chebyshev roots.
"""
function createFTT(g::TensorTrain,order::NTuple{d,S},nodes::NTuple{d,Array{T,1}},domain::Array{T,2}) where {T <: AbstractFloat, S <: Integer, d}

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

    f[i] = [piecewise_linear_evaluate(g.cores[i][j,:,k],(nodes[i],)) for j = 1:g.ranks[i], k = 1:g.ranks[i+1]]

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
function ind2sub(i::S,dims::Array{S,1}) where {S <: Integer}

  if i < 1 || i > prod(dims)
    error("index is out of bounds.")
  end

  subs = Tuple(CartesianIndices(Tuple(dims))[i])

  return subs

end

"""
Integrate a tensor train using Gauss-Chebyshev quadrature.
"""
function TTintegrate_GC(train::TensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R <: AbstractFloat} # Integrates over all dimensions

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
Integrate a tensor train using Gauss-Legendre quadrature.
"""
function TTintegrate_GL(train::TensorTrain,domain::Union{Array{R,2},Array{R,1}}) where {R <: AbstractFloat} # Integrates over all dimensions

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
Integrate a tensor train using Gauss-Hermite quadrature.
"""
function TTintegrate_GH(train::TensorTrain) # Integrates over all dimensions

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