import pandas as pd
import numpy as np
import cantera as ct

import torch


import warnings
# Suppress only the DataConversionWarning from scikit-learn
warnings.filterwarnings(action='ignore', category=UserWarning)

#-------------------------------------------------------------------
# HOMOGENEOUS REACTOR
#-------------------------------------------------------------------

def compute_nn_cantera_0D_homo(device, model, Xscaler, Yscaler, phi_ini, temperature_ini, dt, dtb_params, A_element):

    log_transform = dtb_params["log_transform"]
    threshold = dtb_params["threshold"]
    fuel = dtb_params["fuel"]
    mech_file = dtb_params["mech_file"]
    p = dtb_params["p"]

    # CANTERA gas object
    gas = ct.Solution(mech_file)

    species_list = gas.species_names
    nb_spec = len(species_list)

    # Setting composition
    fuel_ox_ratio = gas.n_atoms(fuel,'C') + 0.25*gas.n_atoms(fuel,'H') - 0.5*gas.n_atoms(fuel,'O')
    compo_ini = f'{fuel}:{phi_ini:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'

    # Computing equilibrium (to get end of simulation criterion)
    gas_equil = ct.Solution(mech_file)
    gas_equil.TPX = temperature_ini, p, compo_ini
    gas_equil.equilibrate('HP')
    state_equil = np.append(gas_equil.X, gas_equil.T)

    # CANTERA RUN ----------------------------------------------------------------
    # Initialize
    gas.TPX = temperature_ini, p, compo_ini

    # Defining reactor
    r = ct.IdealGasConstPressureReactor(gas)
        
    sim = ct.ReactorNet([r])
    simtime = 0.0
    states = ct.SolutionArray(gas, extra=['t'])

    # Initial state (saved for later use in NN computation)
    states.append(r.thermo.state, t=0.0)
    Y_ini = states.Y

    equil_bool = False
    n_iter = 0
    max_sim_time = 1000 * dt  # to limit in case of issues
    equil_tol = 0.5

    while (equil_bool == False) and (simtime < max_sim_time):

        simtime += dt
        sim.advance(simtime)
        states.append(r.thermo.state, t=simtime)

        # checking if equilibrium is reached
        state_current = np.append(r.thermo.X, r.T)
        residual = 100.0*np.linalg.norm(state_equil - state_current,ord=np.inf)/np.linalg.norm(state_equil,
                                                                                                ord=np.inf)
        
        n_iter +=1
        # max iteration                    
        if residual < equil_tol:
            equil_bool = True



    # NN MODEL RUN ----------------------------------------------------------------
    # Evaluation mode
    model.eval()
    
    # Vectors to store time data
    state_save = np.append(temperature_ini, Y_ini)
    state_current = np.append(temperature_ini, Y_ini)

    # Mass and element conservation
    sum_Yk = np.empty(n_iter+1)
    Ye = np.empty((n_iter+1,4))

    #Initial values
    sum_Yk[0] = Y_ini.sum()
    Ye[0,:] = np.dot(A_element, Y_ini.transpose()).ravel()

    crashed = False

    for i in range(n_iter):  # To compute NN on same range than CVODE

        simtime += dt

        # Gas object modification
        try:
            gas.TPY= state_current[0], p, state_current[1:]
        except ct.CanteraError:   # If crash we set solution to zero
            state_current = np.zeros(len(state_current))
            state_save = np.vstack([state_save,state_current])
            crashed = True
            continue

        T_new, Y_new = advance_ANN(state_current, model, Xscaler, Yscaler, gas, log_transform, threshold, device)

        state_current = np.append(T_new, Y_new)

        # Mass conservation
        sum_Yk[i+1] = Y_new.sum()

        # Elements conservation
        Ye[i+1,:] = np.dot(A_element, Y_new.transpose()).ravel()

        # Saving values
        state_save = np.vstack([state_save,state_current])

    if crashed:
        print("WARNING: this simulation crashed, interpret fitness with caution")

    
    # Generate pandas dataframes with solutions -----------------------------------
        
    # EXACT
    n_rows = len(states.t)
    # time / temperature / species
    n_cols = 2 + nb_spec
    arr = np.empty(shape=(n_rows,n_cols))
    arr[:, 0] = states.t
    arr[:, 1] = states.T
    arr[:, 2:2 + nb_spec] = states.Y


    cols = ['Time'] + ['Temperature'] + gas.species_names
    df_exact = pd.DataFrame(data=arr, columns=cols)


    # ANN
    cols_nn = ['Time'] + ['Temperature'] + gas.species_names + ["SumYk", "Y_C", "Y_H", "Y_O", "Y_N"]
    time_vect = np.asarray([i*dt for i in range(n_iter+1)])
    arr_nn = np.hstack([time_vect.reshape(-1,1), state_save, sum_Yk.reshape(-1,1), Ye])
    df_ann = pd.DataFrame(data=arr_nn, columns=cols_nn)

    fail = 0
    if crashed:
        fail = 1

    return df_exact, df_ann, fail




def advance_ANN(state_current, model, Xscaler, Yscaler, gas, log_transform, threshold, device):

    T_m1 = np.copy(state_current[0])
    Yk_m1 = np.copy(state_current[1:])

    # Log transform
    if log_transform:
        state_current[state_current<threshold] = threshold
        state_current[1:] = np.log(state_current[1:])

    # Scaling
    state_current_scaled = (state_current-Xscaler.mean.values)/(Xscaler.std.values+1.0e-7)
    state_current_scaled = state_current_scaled.reshape(-1, 1).T


    # Apply NN
    with torch.no_grad():
        NN_input = torch.from_numpy(state_current_scaled).to(device)
        Y_new = model(NN_input)
    #Back to cpu numpy
    Y_new = Y_new.cpu().numpy()

    # De-transform
    Y_new = Yscaler.mean.values + Y_new * (Yscaler.std.values+1.0e-7)

    # De-log
    if log_transform:
        Y_new = np.exp(Y_new)


    # Deducing T from energy conservation
    T_new = T_m1 - (1/gas.cp)*np.sum(gas.partial_molar_enthalpies/gas.molecular_weights*(Y_new-Yk_m1))

    return T_new, Y_new




