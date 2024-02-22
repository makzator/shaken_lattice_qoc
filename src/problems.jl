import QuantumCollocation as QC
import NamedTrajectories as NT
import TrajectoryIndexingUtils as NTidx

include("system.jl")
include("utils.jl")

function SLProblemSetup(
    psi0s::Vector{<:AbstractVector{<:ComplexF64}},
    psiTs::Vector{<:AbstractVector{<:ComplexF64}},
    duration::Float64;
    V::Float64=10.,
    trunc::Int=13,
    bloch_basis::Bool=true,
    dt::Float64=2*pi*1e-3,
    T::Int=Int(round(duration/dt + 1.)),
    modulation=:phase,
    controls::Vector{AbstractMatrix{Float64}}=Vector{AbstractMatrix{Float64}}(),
)
    # SL system
    system = ShakenLatticeSystem1D(V, trunc; bloch_basis=bloch_basis, bloch_transformation_trunc=3*trunc)
    dim = system.params[:dim]

    # set up fixed time discretization, prioritize setting T
    dt = duration / (T-1)
    dts = zeros(T) .+ dt
    dt_bound = (dt, dt)
    times = cumsum(dts) - dts

    @assert modulation in [:phase, :amp, :phaseamp]
    
    comp_dict = Dict{Symbol, Union{AbstractVector{Float64}, AbstractMatrix{Float64}}}()
    comp_dict[:dts] = dts
    initial_dict = Dict{Symbol, AbstractVector{Float64}}()
    final_dict = Dict{Symbol, AbstractVector{Float64}}()
    bounds_dict = Dict()
    bounds_dict[:dts] = (dt, dt)
    constraints = QC.AbstractConstraint[]
    
    if modulation == :phase
        @assert length(controls) <= 1
        if length(controls) == 0
            comp_dict[:phi] = (2*rand(1, T) .- 1.) * pi
        else
            comp_dict[:phi] = controls[1]
        end
        comp_dict[:a] = rphi_to_IQ(ones(1, T), comp_dict[:phi])
        initial_dict[:phi] = [0.]
        final_dict[:phi] = [0.]
        bounds_dict[:phi] = [1.0*pi]
        control_syms = (:phi)
    elseif modulation == :amp
        @assert length(controls) <= 1
        if length(controls) == 0
            comp_dict[:r] = 2*rand(1, T) .- 1.
        else
            comp_dict[:r] = controls[1]
        end
        comp_dict[:a] = rphi_to_IQ(comp_dict[:r], zeros(1, T))
        initial_dict[:r] = [1.]
        final_dict[:r] = [1.]
        bounds_dict[:r] = [1.]
        control_syms = (:r)
    else
        @assert length(controls) in [0, 2]
        if length(controls) == 0
            comp_dict[:r] = 2*rand(1, T) .- 1.
            comp_dict[:phi] = (2*rand(1, T) .- 1.) * pi
        else
            comp_dict[:r] = controls[1]
            comp_dict[:phi] = controls[2]
        end
        comp_dict[:a] = rphi_to_IQ(comp_dict[:r], comp_dict[:phi])
        initial_dict[:phi] = [0.]
        initial_dict[:r] = [1.]
        final_dict[:phi] = [0.]
        final_dict[:r] = [1.]
        bounds_dict[:phi] = [1.0*pi]
        bounds_dict[:r] = [1.]
        control_syms = (:r, :phi)
    end

    N = length(psi0s)
    @assert all([length(psi0) == dim for psi0 in psi0s])
    @assert all([length(psiT) == dim for psiT in psiTs])
    psi0s = [QC.ket_to_iso(psi0) for psi0 in psi0s]
    psiTs = [QC.ket_to_iso(psiT) for psiT in psiTs]

    psi_syms = [Symbol("psi$(i-1)_iso") for i=1:N]
    psits = [QC.rollout(psi0, comp_dict[:a], dts, system; integrator=exp) for psi0 in psi0s]
    for i=1:N
        initial_dict[psi_syms[i]] = psi0s[i]
        comp_dict[psi_syms[i]] = psits[i]        
    end

    comps = NamedTuple(comp_dict)
    initial = NamedTuple(initial_dict)
    final = NamedTuple(final_dict)
    bounds = NamedTuple(bounds_dict)

    Z_guess = NT.NamedTrajectory(
        comps;
        controls=control_syms,
        timestep=:dts,
        bounds=bounds,
        initial=initial,
        final=final,
        goal=(;)
    )

    J = QC.QuantumObjective(name=:psi0_iso, goals=psiTs[1], loss=:InfidelityLoss, Q=100.0/N)
    for i=2:N
        J += QC.QuantumObjective(name=psi_syms[i], goals=psiTs[i], loss=:InfidelityLoss, Q=100.0/N)
    end

    integrators = [
        QC.QuantumStatePadeIntegrator(
            system,
            psi_syms[i],
            :a,
            :dts;
            order=4
        )
        for i=1:N
    ]

    if modulation == :phase
        constraints = [IQPhiConstraint(:a, :phi, Z_guess)]
    elseif modulation == :amp
        throw("not implemented, pure amplitude modulation needs to be redesigned")
    else
        constraints = [IQAPhiConstraint(:a, :r, :phi, Z_guess)]
    end

    # Ipopt options
    options = QC.Options(
        max_iter=200,
    )

    return system, Z_guess, J, integrators, constraints, options
end