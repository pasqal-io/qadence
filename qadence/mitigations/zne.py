from __future__ import annotations

def expectation_val(reg, obs, parameters, noise_type=None, noise_p=None):
    
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel("ch0", "rydberg_global")
    
    if noise_type == "dephasing":
        config = SimConfig(noise=noise_type,dephasing_prob=noise_p)
    elif noise_type == "depolarizing":
        config = SimConfig(noise=noise_type,depolarizing_prob=noise_p)
    else:
        config = SimConfig()

   
    for x in parameters:                      
        seq.add(Pulse.ConstantPulse(1000 , x, 0.0, 0), "ch0") ## amplitude
        seq.add(Pulse.ConstantPulse(1000 , 0, 1, 0), "ch0") ## detuning

    
    ## sampling rate is used to finely monitor pulse sheduling to apply the evolution
    simul = simulation.QutipEmulator.from_sequence(seq, sampling_rate =0.01, config = config)
    results = simul.run()

    return results.expect([obs])[0][-1]


def zne(prob_range,sample_points,reg,obs,amplitude_list,noise_type):
    prob_min = prob_range[0]
    prob_max = prob_range[1]

    noise_list = np.linspace(prob_min,prob_max, sample_points)
    
    ### storing the expectation values for different noise values
    output = []

    for noise in noise_list:
        output.append(expectation_val(reg,obs,amplitude_list,noise_type=noise_type, noise_p=noise))

    # print(output)  

    ## doing a polynomial fit with 1 less degree to prevent any overfitting, this may not always be a good idea, maybe leave it to the user to decide
    
    degree=sample_points-1
    coefs, res, _, _, _ = np.polyfit(noise_list,output,degree, full = True)
    
    return coefs[-1]




def mitigate(options: dict) -> None:
    ## Lets define our pulses here (this defines the evolution hamiltonian)
    ## using a simple set of pulses
    amplitude_list = np.random.rand(4)


    ## generating a register for qubit location
    n_qubits = 2
    qubits = dict(enumerate([(-2,0),(2,0)]))
    reg = Register(qubits)


    ## taking Z_1 +Z_2 to be the obs
    obs = tensor(sigmaz(),qeye(2)) +  tensor(qeye(2),sigmaz())
    prob_range= (0.1,0.5)
    sample_points = 8
    
    zne = ZNE(prob_range,sample_points,reg,obs,amplitude_list,"depolarizing")
    exact = expectation_val(reg, obs, amplitude_list, )
    return zne, exact
