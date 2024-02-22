import LinearAlgebra as LA
import Base
include("./primitives_utils.jl")


# Bloch basis
Imat = [1. 0.; 0. 1.]
Xmat = [0. 1.; 1. 0.]
Ymat = [0. -1.0im; 1.0im 0.]
Zmat = [1. 0.; 0. -1.]
Hmat = [1. 1.; 1. -1.] / sqrt(2)

function get_n_idc(d_B; d_n=nothing, n_max=nothing)
    @assert !isnothing(d_n) || !isnothing(n_max)
    if !isnothing(d_n)
        @assert d_n % 2 == 1
        n_max = div(d_n, 2)
    end
    @assert d_B >= 2*n_max + 1    
    n_range = -n_max:n_max
    @assert d_B % 2 == 1
    mid = div(d_B, 2) + 1
    return n_range .+ mid
end
   

function BlochPairGate(
    basis::BlochBasis,
    pair_idx::Int,
    qb_mat
)
    # Bloch pair index: pair_idx: 1 = (1,2), 2 = (3,4), 3 = (5,6), ...
    # number of Bloch pairs in trunction: n_Bpairs
    @assert basis.N_pairs >= pair_idx
    mat = collect(LA.I(basis.N)) * (1.0 + 0.0im)
    mat[2*pair_idx:2*pair_idx+1,2*pair_idx:2*pair_idx+1] = qb_mat
    return QO.Operator(basis, mat)
end

X(args...) = BlochPairGate(args..., Xmat)
Y(args...) = BlochPairGate(args..., Ymat)
Z(args...) = BlochPairGate(args..., Zmat)
H(args...) = BlochPairGate(args..., Hmat)

RXY(basis, pair_idx, theta, phi) = BlochPairGate(basis, pair_idx, cos(theta/2)*Imat - 1im*sin(theta/2)*(cos(phi)*Ymat + sin(phi)*Xmat))
RX(basis, pair_idx, theta) = RXY(basis, pair_idx, theta, pi/2)
RY(basis, pair_idx, theta) = RXY(basis, pair_idx, theta, 0.)
RZ(basis, pair_idx, phi) = BlochPairGate(basis, pair_idx, [exp(-1im*phi/2) 0.; 0. exp(1im*phi/2)])


function free_propagator(
    t::Float64,
    bloch_energies::Vector{Float64}
)
    U = QO.Operator(BlochBasisN(length(bloch_energies)), LA.diagm(exp.(-1im*bloch_energies*t)))
    return U
end

function free_hamiltonian(
    p_basis::QO.MomentumBasis,
    V::Float64
)
    w = get_w(p_basis)
    p = QO.momentum(p_basis)
    H_kin_p = p^2
    H_pot_p = -V/4 * QO.Operator(p_basis, LA.diagm(w => ones(p_basis.N-w), -w => ones(p_basis.N-w)))
    H_p = H_kin_p + H_pot_p
    return H_p
end

function free_propagator_QO(
    p_basis::QO.MomentumBasis,
    ts::Vector{Float64},
    V::Float64
)
    H = free_hamiltonian(p_basis, V)
    _, Ut = QO.timeevolution.schroedinger(ts, collect(QO.identityoperator(p_basis)), H)
    return Ut
end
free_propagator_QO(p_basis::QO.MomentumBasis, t::Float64, V::Float64) = free_propagator_QO(p_basis, [0., t], V)[end]


struct GateSeries
    gates::Vector{QO.Operator}
    # function GateSeries(gates::Vector{QO.Operator}=Vector{QO.Operator}[])
    #     return new(gates)
    # end
end

CircuitElement = Union{QO.Operator, GateSeries}
struct Circuit
    gates::Vector{CircuitElement}
    function Circuit()
        return new(CircuitElement[])
    end
    # function GateSeries(gates::T=T[]) where T=Vector{Union{QO.Operator, GateSeries}}
    #     return new{gates}
    # end
end
Base.push!(circ::Circuit, element::CircuitElement) = push!(circ.gates, element)
Base.append!(circ::Circuit, elements::Vector{<:QO.Operator}) = append!(circ.gates, elements)
Base.append!(circ::Circuit, elements::GateSeries) = append!(circ.gates, elements.gates)
Base.append!(circ::Circuit, element::CircuitElement) = push!(circ, element)
function Base.append!(circ::Circuit, elements::Vector{CircuitElement})
    for element in elements
        append!(circ, element)
    end
end
function unroll(circ::Circuit)
    new_circ = Circuit()
    for el in circ.gates
        if el isa GateSeries
            append!(new_circ, el)
        else
            push!(new_circ, el)
        end
    end
    return new_circ
end
function concatenate(circs...)
    new_circ = Circuit()
    for circ in circs
        @assert circ isa Circuit
        append!(new_circ, circ.gates)
    end
    return new_circ
end
