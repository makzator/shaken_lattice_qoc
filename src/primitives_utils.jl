import QuantumOptics as QO
import Base
include("./system.jl")

struct BlochBasis{T} <: QO.Basis
    shape::Vector{T}
    N_pairs::T
    N::T
    function BlochBasis(N_pairs::T) where T<:Int
        N = 2*N_pairs + 1 # +1 bc of unpaired |0>
        return new{T}([N], N_pairs, N)
    end
end
function BlochBasisN(N::T) where T<:Int
    @assert N % 2 == 1
    return BlochBasis(div(N, 2))
end
function BlochBasis(p_basis::QO.MomentumBasis)
    return BlochBasis(nBasis(p_basis))
end
Base.:(==)(b1::BlochBasis, b2::BlochBasis) = b1.N==b2.N


struct nBasis{T} <: QO.Basis
    shape::Vector{T}
    n_max::T
    N::T
    function nBasis(n_max::T) where T<:Int
        N = 2*n_max + 1
        return new{T}([N], n_max, N)
    end
end
function nBasisN(N::T) where T<:Int
    @assert N % 2 == 1
    return nBasis(div(N, 2))
end
function nBasis(p_basis::QO.MomentumBasis)
    p_max = p_basis.pmax
    @assert p_max == -p_basis.pmin
    return nBasis(Int(round((p_max - 1.)/2)))
end
Base.:(==)(b1::nBasis, b2::nBasis) = b1.N==b2.N


BlochBasis(n_basis::nBasis) = BlochBasisN(n_basis.N)
nBasis(bloch_basis::BlochBasis) = nBasisN(bloch_basis.N)


function MomentumBasis(n_basis::nBasis, w::Int)
    p_max = 2*n_basis.n_max + 1
    return QO.MomentumBasis(-p_max, p_max, p_max*w)
end
function MomentumBasis(bloch_basis::BlochBasis, w::Int)
    return MomentumBasis(nBasis(bloch_basis), w)
end
function get_w(p_basis::QO.MomentumBasis)
    return Int(round(p_basis.N/p_basis.pmax))
end

transform_nB(bloch_states) = QO.Operator(
                                nBasisN(size(bloch_states)[1]),
                                BlochBasisN(size(bloch_states)[2]),
                                bloch_states
                                )   
transform_Bn(bloch_states) = transform_nB(bloch_states)'


function coarse_setup(system::QC.QuantumSystem)
    TnB = transform_nB(system.params[:bloch_states])
    return TnB.basis_l, TnB.basis_r, TnB
end



function gate_to_same_basis_type(
    out_basis::Union{BlochBasis, nBasis, QO.MomentumBasis},
    gate::QO.Operator 
)
    @assert gate.basis_l == gate.basis_r && typeof(gate.basis_l) == typeof(out_basis)
    in_basis = gate.basis_l
    T = eltype(gate.data)
    if out_basis isa BlochBasis
        if out_basis.N_pairs > in_basis.N_pairs
            mat = Matrix{T}(LA.I, out_basis.N, out_basis.N)
            mat[1:in_basis.N,1:in_basis.N] = gate.data
            out_gate = QO.Operator(out_basis, mat)
        else
            TBB = QO.Operator(out_basis, in_basis, Matrix{T}(LA.I, out_basis.N, in_basis.N))
            out_gate = TBB * gate * TBB'
        end
    elseif out_basis isa nBasis
        if out_basis.n_max > in_basis.n_max
            mat = Matrix{T}(LA.I, out_basis.N, out_basis.N)
            in_mid = in_basis.N - in_basis.n_max
            out_mid = out_basis.N - out_basis.n_max
            idc = -in_basis.n_max:in_basis.n_max .+ out_mid
            mat[idc,idc] = gate.data
            out_gate = QO.Operator(out_basis, mat)
        else
            d = in_basis.n_max - out_basis.n_max
            TBB_mat = hcat(zeros(out_basis.N,d), Matrix{T}(LA.I, out_basis.N, out_basis.N), zeros(out_basis.N,d))
            TBB = QO.Operator(out_basis, in_basis, TBB_mat)
            out_gate = TBB * gate * TBB'
        end
    else
        # MomentumBasis -> nBasis -> MomentumBasis
        gate_n_basis = gate_to_basis(gate, nBasis(in_basis))
        gate_n_basis = gate_to_same_basis_type(gate_n_basis, nBasis(out_basis))
        out_gate = gate_to_basis(gate_n_basis, out_basis)
    end
    return out_gate
end


function gate_to_basis(
    out_basis::Union{BlochBasis, nBasis, QO.MomentumBasis},
    gate::QO.Operator;
    TnB::Union{Nothing, QO.Operator}=nothing,
    sparse::Bool=false
)
    in_basis = gate.basis_l
    @assert in_basis == gate.basis_r
    if typeof(in_basis) == typeof(out_basis)
        return gate_to_same_basis_type(gate, out_basis)
    end
    if out_basis isa BlochBasis
        if in_basis isa nBasis
            @assert !isnothing(TnB) && (TnB.basis_l == in_basis || TnB.basis_r == out_basis)
            if TnB.basis_l == in_basis
                out_gate = TnB' * gate * TnB
                out_gate = gate_to_same_basis_type(out_gate, out_basis)
            else
                out_gate = gate_to_same_basis_type(gate, TnB.basis_l)
                out_gate = TnB' * gate * TnB
            end
        else
            # MomentumBasis
            out_gate = gate_to_basis(gate, nBasis(in_basis), TnB=TnB)
        end
    elseif out_basis isa nBasis
        if in_basis isa BlochBasis
            @assert !isnothing(TnB) && (TnB.basis_l == out_basis || TnB.basis_r == in_basis)
            if TnB.basis_r == in_basis
                out_gate = TnB * gate * TnB'
                out_gate = gate_to_same_basis_type(out_gate, out_basis)
            else
                out_gate = gate_to_same_basis_type(gate, TnB.basis_r)
                out_gate = TnB * gate * TnB'
            end
        else
            # MomentumBasis
            w = get_w(in_basis)
            in_n_basis = nBasis(in_basis)
            idc = (0:(in_n_basis.N-1)) * w .+ div(w, 2)
            out_gate = QO.Operator(in_n_basis, gate.data[idc,idc])
            out_gate = gate_to_same_basis_type(gate, out_basis)
        end
    else
        # out_basis isa MomentumBasis
        if in_basis isa BlochBasis
            gate = gate_to_basis(gate, nBasis(in_basis); TnB=TnB)
        end
        # in_basis isa nBasis
        out_gate = gate_to_same_basis_type(gate, nBasis(out_basis))
        w = get_w(out_basis)
        I_w = Matrix{eltype(out_gate.data)}(LA.I, w, w)
        out_gate = QO.Operator(out_basis, kron(out_gate.data, I_w))
    end
    if sparse
        out_gate = QO.sparse(out_gate)
    end    
    return out_gate
end
