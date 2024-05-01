import LinearAlgebra as LA
import Base
include("./primitives_utils.jl")


# Bloch basis
Imat = [1. 0.; 0. 1.]
Xmat = [0. 1.; 1. 0.]
Ymat = [0. -1.0im; 1.0im 0.]
Zmat = [1. 0.; 0. -1.]
Hmat = [1. 1.; 1. -1.] / sqrt(2)

# function get_n_idc(d_B; d_n=nothing, n_max=nothing)
#     @assert !isnothing(d_n) || !isnothing(n_max)
#     if !isnothing(d_n)
#         @assert d_n % 2 == 1
#         n_max = div(d_n, 2)
#     end
#     @assert d_B >= 2*n_max + 1    
#     n_range = -n_max:n_max
#     @assert d_B % 2 == 1
#     mid = div(d_B, 2) + 1
#     return n_range .+ mid
# end
   

function BlochPairGate(
    basis::BlochBasis,
    pair_idx::Int,
    qb_mat;
    name::String="B"
)
    # Bloch pair index: pair_idx: 1 = (1,2), 2 = (3,4), 3 = (5,6), ...
    # number of Bloch pairs in trunction: n_Bpairs
    i = pair_idx
    @assert basis.N_pairs >= i
    mat = collect(LA.I(basis.N)) * (1.0 + 0.0im)
    mat[2*i:2*i+1,2*i:2*i+1] = qb_mat
    return OpGate(QO.Operator(basis, mat); name="$name($(2i-1),$(2i))")
end

X(args...) = BlochPairGate(args..., Xmat; name="X")
Y(args...) = BlochPairGate(args..., Ymat; name="Y")
Z(args...) = BlochPairGate(args..., Zmat; name="Z")
H(args...) = BlochPairGate(args..., Hmat; name="H")

RXY(basis, pair_idx, theta, phi) = BlochPairGate(
                                                basis, 
                                                pair_idx, 
                                                cos(theta/2)*Imat - 1im*sin(theta/2)*(cos(phi)*Ymat + sin(phi)*Xmat);
                                                name="RXY($(round(theta, digits=2)), $(round(phi, digits=2))"
                                                )
RX(basis, pair_idx, theta) = BlochPairGate(
                                        basis, 
                                        pair_idx, 
                                        cos(theta/2)*Imat - 1im*sin(theta/2)*Xmat; 
                                        name="RX($(round(theta, digits=2)))"
                                        )
RY(basis, pair_idx, theta) = BlochPairGate(
                                        basis, 
                                        pair_idx, 
                                        cos(theta/2)*Imat - 1im*sin(theta/2)*Ymat;
                                        name="RY($(round(theta, digits=2)))"
                                        )
RZ(basis, pair_idx, phi) = BlochPairGate(
                                        basis, 
                                        pair_idx, 
                                        [exp(-1im*phi/2) 0.; 0. exp(1im*phi/2)];
                                        name="RZ($(round(phi, digits=2)))"
                                        )


function free_propagator_B(
    t::Float64,
    bloch_energies::Vector{Float64}
)
    U = QO.Operator(BlochBasis(; N=length(bloch_energies)), LA.diagm(exp.(-1im*bloch_energies*t)))
    return OpGate(U, "free_B", t)
end

function free_hamiltonian(
    p_basis::pBasis,
    V::Float64
)
    w = p_basis.w
    p = QO.momentum(QO.MomentumBasis(p_basis))
    H_kin_p = p^2
    H_pot_p = -V/4 * QO.Operator(QO.MomentumBasis(p_basis), LA.diagm(w => ones(p_basis.N-w), -w => ones(p_basis.N-w)))
    H_p = H_kin_p + H_pot_p
    return H_p
end

function free_propagator_p(
    p_basis::pBasis,
    ts::Vector{Float64},
    V::Float64
)
    H = free_hamiltonian(p_basis, V)
    _, Ut = QO.timeevolution.schroedinger(ts, collect(QO.identityoperator(QO.MomentumBasis(p_basis))), H)
    return [OpGate(QO.Operator(p_basis, Ut[i+1].data), "free_p", ts[i+1]-ts[i]) for i=1:(length(Ut)-1)]
end
free_propagator_p(p_basis::pBasis, t::Float64, V::Float64) = free_propagator_p(p_basis, [0., t], V)[end]

function free_propagator_action_p(
    p_basis::pBasis,
    ts::Vector{Float64},
    V::Float64
)
    H = free_hamiltonian(p_basis, V)
    function action(psi::QO.Ket)
        # psi = ket_to_basis(p_basis, psi)
        _, psis = QO.timeevolution.schroedinger(ts, QO.Ket(QO.MomentumBasis(p_basis), psi.data), H)
        psis = [QO.Ket(p_basis, psi.data) for psi in psis]
        return ts, psis
    end
    return ActionGate(action, "free_p", ts)
end


abstract type AbstractCircuit end

struct CircuitBlock <: AbstractCircuit
    circ::AbstractCircuit
    name::String
end
CircuitBlock(circ::AbstractCircuit) = CircuitBlock(circ, "CB")
show_name_(block::CircuitBlock) = "[$(block.name)]"
function Base.show(io::IO, block::CircuitBlock)
    print(io, show_name_(block))
end

CircuitElement = Union{AbstractGate, CircuitBlock}
struct Circuit <: AbstractCircuit
    elements::Vector{CircuitElement}
end
Circuit() = Circuit(CircuitElement[])
function Base.show(io::IO, circ::Circuit)
    s = " - "
    for el in circ.elements
        s = s * show_name_(el) * " - "
    end
    print(io, s)
end

Base.push!(circ::Circuit, element::CircuitElement) = push!(circ.elements, deepcopy(element))
Base.append!(circ::Circuit, elements::Vector{<:CircuitElement}) = append!(circ.elements, deepcopy(elements))
# Base.append!(circ::Circuit, block::CircuitBlock) = append!(circ.elements, elements.elements)
# Base.append!(circ::Circuit, element::CircuitElement) = push!(circ, element)
# function Base.append!(circ::Circuit, elements::Vector{CircuitElement})
#     for element in elements
#         append!(circ, element)
#     end
# end
function concatenate(circs...)
    new_circ = Circuit()
    for circ in circs
        @assert circ isa Circuit
        append!(new_circ, circ.elements)
    end
    return new_circ
end
function unroll(circ::Circuit)
    new_circ = Circuit()
    for el in circ.elements
        if el isa AbstractGate
            push!(new_circ, el)
        else
            append!(new_circ, el.circ.elements)
        end
    end
    return new_circ
end
function unroll_full(circ::Circuit)
    new_circ = Circuit()
    for el in circ.elements
        if el isa AbstractGate
            push!(new_circ, el)
        else
            append!(new_circ, unroll_full(el.circ).elements)
        end
    end
    return new_circ
end

function rollout(circ::Circuit, psi0::QO.Ket, basis; TnB=TnB)
    unrolled_circ = unroll_full(circ)
    # if isnothing(basis)
    #     # find largest n_max and create BlochBasis
    #     N = 0
    #     for gate in unrolled_circ.elements
    #         N = max(N, nBasis(gate.basis_l).N)
    #     end
    #     basis = BlochBasis(; N=N)
    # end
    #psi = ket_to_basis(basis, psi0; TnB=TnB)
    psis = [psi0]
    ts = [0.]
    print(" - ")
    for gate in unrolled_circ.elements
        if gate isa OpGate
            op = op_to_basis(basis, gate.op; TnB=TnB)
            push!(psis, op * psis[end])
            push!(ts, ts[end] + gate.t)
        else
            # gate isa ActionGate
            ts_, psis_ = gate.action(psis[end])
            append!(psis, psis_)
            append!(ts, ts_ .+ ts[end])
        end
        print("$(gate.name) - ")
    end
    return ts, psis
end