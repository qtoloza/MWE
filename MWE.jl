using OrdinaryDiffEq, Octavian, Parameters, DiffEqCallbacks, MAT, LSODA, Sundials

function input_RHS!(du, u, p, t)

    du .= -1.0.*u./10

    nothing

end

@views function forward_RHS!(du, u, p, t)

    p.input_solution(p.n, t)

    matmul!(p.inputs_s, p.U_s, p.n)
    matmul!(p.recurrents_s, p.R_s, u[p.network_size + 1:2*p.network_size])

    @inbounds @simd for i in 1:p.network_size
        du[i] = ((-40*(u[i] - 0.3)^2 + 0.004)*(u[i] - 1)
                    - 20*u[2*p.network_size + i]*(u[i] - 0.3)
                    - 0.5*u[3*p.network_size + i]*(u[i] - 0.2)
                    + p.inputs_s[i] + p.recurrents_s[i] + p.I_s[i])

        du[p.network_size + i] = (2*u[2*p.network_size + i] - u[p.network_size + i])/10

        du[2*p.network_size + i] = (10*(u[i] - 0.3)^3 - u[2*p.network_size + i])/2

        du[3*p.network_size + i] = (16*(u[i] - 0.3)^2 + 0.2 - u[3*p.network_size + i])/5
    end

    nothing

end

@with_kw mutable struct InputParameters
    number_of_inputs::Int

    input_neurons::Vector{Int}
    input_index::Int = 1
end

@with_kw mutable struct ForwardParameters
    number_of_inputs::Int
    network_size::Int
    number_of_targets::Int

    U_s::Matrix{Float64}
    O::Matrix{Float64}
    R_s::Matrix{Float64}
    I_s::Vector{Float64}

    input_solution::ODESolution

    n::Vector{Float64} = zeros(number_of_inputs)

    inputs_s::Vector{Float64} = zeros(network_size)
    recurrents_s::Vector{Float64} = zeros(network_size)
end

function input_spike_time_affect!(integrator)

    neuron = integrator.p.input_neurons[integrator.p.input_index]

    integrator.u[neuron] += 0.5*(2 - integrator.u[neuron])

    integrator.p.input_index += 1

end

#data_file = matread("data.mat")
data_file = matread("nonconvergent_data.mat")
network_weights = matread("weights.mat")

input_spike_times = data_file["input_spike_times"]
input_neurons = data_file["input_neurons"]

input_parameters = InputParameters(number_of_inputs = 64,
    input_neurons = input_neurons)

input_spike_time_cb = PresetTimeCallback(input_spike_times, input_spike_time_affect!, save_positions = (true, true))

input_ODE = ODEProblem(input_RHS!, zeros(100), (0, 200), input_parameters)
input_solution = solve(input_ODE, Tsit5(), callback = input_spike_time_cb, dense = true, reltol = 1e-3, abstol = 1e-6)

println("input solve done.")
flush(stdout)

forward_parameters = ForwardParameters(number_of_inputs = 100, network_size = 64, number_of_targets = 6,
    U_s = network_weights["U_s"], O = network_weights["O"],
    R_s = network_weights["R_s"],
    I_s = network_weights["I_s"],
    input_solution = input_solution)

forward_ODE = ODEProblem(forward_RHS!, zeros(4*64), (0, 200), forward_parameters)
forward_solution = solve(forward_ODE, AutoDP5(Rosenbrock23(autodiff = false)), dense = true, reltol = 1e-4, abstol = 1e-6)
#forward_solution = solve(forward_ODE, Tsit5(), dense = true, reltol = 1e-4, abstol = 1e-6)
#forward_solution = solve(forward_ODE, Rosenbrock23(autodiff = false), dense = true, reltol = 1e-4, abstol = 1e-6)
#forward_solution = solve(forward_ODE, lsoda(), reltol = 1e-4, abstol = 1e-6)
#forward_solution = solve(forward_ODE, CVODE_BDF(), dense = true, reltol = 1e-4, abstol = 1e-6)

println("forward solve done.")
flush(stdout)
