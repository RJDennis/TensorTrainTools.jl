
using TensorTrainTools
using Test

@testset "TensorTrainTools.jl" begin
    
    constant_tensor_train_a = TTconstant(2.0,(5,6,7),3)
    constant_tensor_train_b = TTconstant(2.0,(5,6,7),[1,3,3,1])

    random_tensor_train_a = TTrand((5,6,7),3)
    random_tensor_train_b = TTrand((5,6,7),[1,3,3,1])

    # Test on exponential density

    test(x) = exp(-0.5*sum(x.^2))

    n1 = cheb_nodes(51,[2.0,-2.0])
    n2 = cheb_nodes(51,[2.0,-2.0])
    n3 = cheb_nodes(51,[2.0,-2.0])
      
    A = [test([n1[i],n2[j],n3[k]]) for i in 1:51, j in 1:51, k in 1:51]

    A_TTsvd  = TTsvd(A,1e-10)
    A_DMRG_a = DMRGcross(A,1.05,1e-10)
    A_DMRG_b = DMRGcross(test,(n1,n2,n3),1.05,1e-10)

    B_TTsvd  = TTdecompress(A_TTsvd)
    B_DMRG_a = TTdecompress(A_DMRG_a)
    B_DMRG_b = TTdecompress(A_DMRG_b)

    test_one   = maximum(abs,A-B_TTsvd) < 1e-10
    test_two   = maximum(abs,A-B_DMRG_a) < 1e-10
    test_three = maximum(abs,A-B_DMRG_b) < 1e-10
      
    # Test on Hilbert function
      
    hilbert(x) = 1/sum(x)
      
    nodes = cheb_nodes(21,[1.0,0.0])
      
    node_vec = (nodes,nodes,nodes,nodes)
      
    hilbert_tt = DMRGcross(hilbert,node_vec,1.05,1e-9)
    H = TTdecompress(hilbert_tt)
      
    n = length.(node_vec)
    J = [hilbert([nodes[i],nodes[j],nodes[k],nodes[l]]) for i in 1:21, j in 1:21, k in 1:21, l in 1:21]
    
    test_four = maximum(abs,H-J) < 1e-7
      
    hilbert_ftt = TTcreateFTT(hilbert_tt,node_vec,[1.0 1.0 1.0 1.0; 0.0 0.0 0.0 0.0])
    TTevaluate(hilbert_ftt,[0.5,0.5,0.5,0.5])
      
    hilbertfn = TTinterp(hilbert_ftt)

    test_five = abs(hilbertfn([0.5,0.5,0.5,0.5]) - hilbert([0.5,0.5,0.5,0.5])) < 1e-7
      
    # Test integrateTT by integrating a normal density function 
      
    normal_density(x) = (1/sqrt(2π))^length(x)*exp((-1/(2))*sum(x.^2))
      
    x_nodes = cheb_nodes(1001,[6.0,-6.0])
      
    normTT = DMRGcross(normal_density,(x_nodes,x_nodes,x_nodes),1.05,1e-13)
      
    integral = TTintegrate_GC(normTT,[6.0 6.0 6.0;-6.0 -6.0 -6.0])

    test_six = abs(integral-1.0) < 1e-8

end
