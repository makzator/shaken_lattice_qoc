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
Base.:(==)(b1::BlochBasis, b2::BlochBasis) = b1.N==b2.N
function BlochBasis(; N::T=11) where T<:Int
    @assert N % 2 == 1
    return BlochBasis(div(N, 2))
end


struct nBasis{T} <: QO.Basis
    shape::Vector{T}
    n_max::T
    N::T
    function nBasis(n_max::T) where T<:Int
        N = 2*n_max + 1
        return new{T}([N], n_max, N)
    end
end
Base.:(==)(b1::nBasis, b2::nBasis) = b1.N==b2.N
function nBasis(; N::T=11) where T<:Int
    @assert N % 2 == 1
    return nBasis(div(N, 2))
end


struct pBasis{T} <: QO.Basis
    shape::Vector{T}
    p_max::T
    N::T
    w::T
    function pBasis(n_max::T, w::T) where T<:Int
        p_max = 2*n_max + 1
        N = p_max * w
        return new{T}([N], p_max, N, w)
    end
end
Base.:(==)(b1::pBasis, b2::pBasis) = b1.N==b2.N && b1.w==b2.w
QO.MomentumBasis(p_basis::pBasis) = QO.MomentumBasis(-p_basis.p_max, p_basis.p_max, p_basis.N)
QO.samplepoints(p_basis::pBasis) = QO.samplepoints(QO.MomentumBasis(p_basis))


nBasis(bloch_basis::BlochBasis) = nBasis(; N=bloch_basis.N)
nBasis(p_basis::pBasis) = nBasis(div(p_basis.p_max-1,2))
BlochBasis(n_basis::nBasis) = BlochBasis(; N=n_basis.N)
BlochBasis(p_basis::pBasis) = BlochBasis(nBasis(p_basis))
pBasis(n_basis::nBasis, w::Int) = pBasis(n_basis.n_max, w)
pBasis(bloch_basis::BlochBasis, w::Int) = pBasis(nBasis(bloch_basis), w)


transform_nB(bloch_states) = QO.Operator(
                                nBasis(; N=size(bloch_states)[1]),
                                BlochBasis(; N=size(bloch_states)[2]),
                                bloch_states
                                )   
transform_Bn(bloch_states) = transform_nB(bloch_states)'


function coarse_setup(system::QC.QuantumSystem)
    TnB = transform_nB(system.params[:bloch_states])
    return TnB.basis_l, TnB.basis_r, TnB
end


struct Gate
    op::QO.Operator
    name::String 
end
function Gate(op::QO.Operator; name::String="U")
    return Gate(op, name)
end
function Gate(gate::Gate, new_op::QO.Operator)
    fields = [getfield(gate, key) for key in fieldnames(Gate)]
    fields[1] = new_op
    return Gate(fields...)
end
show_name_(gate::Gate) = gate.name
Base.:(==)(g1::Gate, g2::Gate) = all([getfield(g1, key) == getfield(g2, key) for key in fieldnames(Gate)])


function op_to_same_basis_type(
    out_basis::Union{BlochBasis, nBasis, pBasis},
    op::QO.Operator
)   
    @assert op.basis_l == op.basis_r && typeof(op.basis_l) == typeof(out_basis)
    in_basis = op.basis_l
    T = eltype(op.data)
    if out_basis isa BlochBasis
        if out_basis.N_pairs > in_basis.N_pairs
            mat = Matrix{T}(LA.I, out_basis.N, out_basis.N)
            mat[1:in_basis.N,1:in_basis.N] = op.data
            out_op = QO.Operator(out_basis, mat)
        else
            TBB = QO.Operator(out_basis, in_basis, Matrix{T}(LA.I, out_basis.N, in_basis.N))
            out_op = TBB * op * TBB'
        end
    elseif out_basis isa nBasis
        if out_basis.n_max > in_basis.n_max
            mat = Matrix{T}(LA.I, out_basis.N, out_basis.N)
            in_mid = in_basis.N - in_basis.n_max
            out_mid = out_basis.N - out_basis.n_max
            idc = -in_basis.n_max:in_basis.n_max .+ out_mid
            mat[idc,idc] = op.data
            out_op = QO.Operator(out_basis, mat)
        else
            d = in_basis.n_max - out_basis.n_max
            TBB_mat = hcat(zeros(out_basis.N,d), Matrix{T}(LA.I, out_basis.N, out_basis.N), zeros(out_basis.N,d))
            TBB = QO.Operator(out_basis, in_basis, TBB_mat)
            out_op = TBB * op * TBB'
        end
    else
        # MomentumBasis -> nBasis -> MomentumBasis
        op_n_basis = op_to_basis(nBasis(in_basis), op)
        op_n_basis = op_to_same_basis_type(nBasis(out_basis), op_n_basis)
        out_op = op_to_basis(out_basis, op_n_basis)
    end
    return out_op
end

gate_to_same_basis_type(
    out_basis::Union{BlochBasis, nBasis, pBasis},
    gate::Gate
) = Gate(gate, op_to_same_basis_type(out_basis, gate.op))


function op_to_basis(
    out_basis::Union{BlochBasis, nBasis, pBasis},
    op::QO.Operator;
    TnB::Union{Nothing, QO.Operator}=nothing,
    sparse::Bool=false
)
    in_basis = op.basis_l
    @assert in_basis == op.basis_r
    if typeof(in_basis) == typeof(out_basis)
        return op_to_same_basis_type(out_basis, op)
    end
    if out_basis isa BlochBasis
        if in_basis isa nBasis
            @assert !isnothing(TnB) && (TnB.basis_l == in_basis || TnB.basis_r == out_basis)
            if TnB.basis_l == in_basis
                out_op = TnB' * op * TnB
                out_op = op_to_same_basis_type(out_basis, out_op)
            else
                out_op = op_to_same_basis_type(TnB.basis_l, op)
                out_op = TnB' * op * TnB
            end
        else
            # MomentumBasis
            out_op = op_to_basis(nBasis(in_basis), op; TnB=TnB)
        end
    elseif out_basis isa nBasis
        if in_basis isa BlochBasis
            @assert !isnothing(TnB) && (TnB.basis_l == out_basis || TnB.basis_r == in_basis)
            if TnB.basis_r == in_basis
                out_op = TnB * op * TnB'
                out_op = op_to_same_basis_type(out_basis, out_op)
            else
                out_op = op_to_same_basis_type(TnB.basis_r, op)
                out_op = TnB * op * TnB'
            end
        else
            # MomentumBasis
            w = in_basis.w
            in_n_basis = nBasis(in_basis)
            idc = (0:(in_n_basis.N-1)) * w .+ div(w, 2)
            out_op = QO.Operator(in_n_basis, op.data[idc,idc])
            out_op = op_to_same_basis_type(out_basis, out_op)
        end
    else
        # out_basis isa MomentumBasis
        if in_basis isa BlochBasis
            op = op_to_basis(nBasis(in_basis), op; TnB=TnB)
        end
        # in_basis isa nBasis
        out_op = op_to_same_basis_type(nBasis(out_basis), op)
        w = out_basis.w
        I_w = Matrix{eltype(out_op.data)}(LA.I, w, w)
        out_op = QO.Operator(out_basis, kron(out_op.data, I_w))
    end
    if sparse
        out_op = QO.sparse(out_op)
    end    
    return out_op
end

gate_to_basis(
    out_basis::Union{BlochBasis, nBasis, pBasis},
    gate::Gate;
    TnB::Union{Nothing, QO.Operator}=nothing,
    sparse::Bool=false
) = Gate(gate, op_to_basis(out_basis, gate.op; TnB=TnB, sparse=sparse))