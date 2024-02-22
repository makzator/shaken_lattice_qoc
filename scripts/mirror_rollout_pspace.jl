import QuantumCollocation as QC
import NamedTrajectories as NT
import TrajectoryIndexingUtils as NTidx
import LinearAlgebra as LA
import SparseArrays as SA
import Interpolations as IP
using LaTeXStrings
import QuantumOptics as QO

import JLD2

include("utils.jl")
include("system.jl")
include("constraints.jl")
include("objectives.jl")


V = 10.
p_max = 10
system_momentum = ShakenLatticeSystem1D(V, p_max; bloch_basis=false)
mid = system_momentum.params[:mid]
dim = system_momentum.params[:dim]

x_max = 70.
Nx = 500
b_position = QO.PositionBasis(-x_max, x_max, Nx)
b_momentum = QO.MomentumBasis(b_position)
xs = QO.samplepoints(b_position)
ps = QO.samplepoints(b_momentum)

Txp = QO.transform(b_position, b_momentum)
Tpx = QO.transform(b_momentum, b_position)
p_op = QO.momentum(b_momentum)
H_kin_p = p_op^2
H_pot_I_x = QO.potentialoperator(b_position, x -> -V/2 * cos(2x))
H_pot_Q_x = QO.potentialoperator(b_position, x -> V/2 * sin(2x))
H_pot_I_p = QO.LazyProduct(Tpx, H_pot_I_x, Txp)
H_pot_Q_p = QO.LazyProduct(Tpx, H_pot_Q_x, Txp)


Z_mirror = NT.load_traj("interferometer/mirror_bloch78_Z.jld2")
flight_time = 2pi * 0.2
T_flight = Int(round(flight_time/2pi * 1000; digits=0))
dts_flight = fill(flight_time/(T_flight-1), T_flight)
a, dts = Z_mirror.a, vec(Z_mirror.dts)

T = size(a, 2)
times = cumsum(dts) - dts

I_itp = IP.interpolate(a[1,:], IP.BSpline(IP.Cubic(IP.Free(IP.OnCell()))))
Q_itp = IP.interpolate(a[2,:], IP.BSpline(IP.Cubic(IP.Free(IP.OnCell()))))

function I(t)
    if t < times[end]
        return I_itp(t/times[end]*(T-1) + 1)
    else
        return 1.0
    end
end 
function Q(t)
    if t < times[end]
        return Q_itp(t/times[end]*(T-1) + 1)
    else
        return 0.0
    end
end 

H_p = QO.TimeDependentSum(1.0 => H_kin_p, I => H_pot_I_p, Q => H_pot_Q_p)

function gaussian_momentum(p, sigma)
    return (sigma^2*pi)^(-1/4) * exp.(-p.^2/(2*sigma^2))
end
function momentum_comb(p, v, sigma)
    l = div(length(v),2)
    ns = -l:l
    return gaussian_momentum(p*ones(1,length(ns)) - 2*ones(length(p))*ns', sigma) * v * (1. + 0. * 1im)
end

v = system_momentum.params[:bloch_states][:,8]
sigma = 0.2
psip0 = momentum_comb(ps, v, sigma)
psip0_ket = QO.Ket(b_momentum, psip0)

t_max = 20.0
dt = 0.007
times = collect(0.0:dt:t_max)
times_slice = 1:length(times)

Up0_op = QO.identityoperator(b_momentum) * 1.0
tout, Upt_op = QO.timeevolution.schroedinger_dynamic(times[times_slice], Up0_op, H_p)

JLD2.save("mirror_rollout.jld2", Dict("tout" => tout, "Upt_op" => Upt_op))