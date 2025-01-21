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

def compute_nn_cantera_0D_homo(device, kmeans, kmeans_scaler, model_list, Xscaler, Yscaler, phi_ini, temperature_ini, dt, dtb_params, A_element):

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

        # attributing cluster
        i_cluster = attribute_cluster(kmeans, kmeans_scaler, state_current, log_transform, threshold)
        print(f">> Current point in cluster {i_cluster}")

        # Evaluation mode
        model = model_list[i_cluster]
        model.eval()

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



#-------------------------------------------------------------------
# 1D premixed flame
#-------------------------------------------------------------------

def compute_nn_cantera_1D_prem(device, kmeans, kmeans_scaler, model_list, Xscaler, Yscaler, phi_ini, temperature_ini, dt, dtb_params, A_element):

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


    initial_grid = np.linspace(0.0, 0.03, 10)  # m
    tol_ss = [1.0e-4, 1.0e-9]  # [rtol atol] for steady-state problem
    tol_ts = [1.0e-5, 1.0e-5]  # [rtol atol] for time stepping
    loglevel = 0  # amount of diagnostic output (0 to 8)

    # COMPUTE 1D PREMIXED FLAME -------------------------------------------------------
    # Initialize
    gas.TPX = temperature_ini, p, compo_ini

    # Flame
    f = ct.FreeFlame(gas, initial_grid)

    f.flame.set_steady_tolerances(default=tol_ss)
    f.flame.set_transient_tolerances(default=tol_ts)
    f.inlet.set_steady_tolerances(default=tol_ss)
    f.inlet.set_transient_tolerances(default=tol_ts)
    f.outlet.set_steady_tolerances(default=tol_ss)
    f.outlet.set_transient_tolerances(default=tol_ts)

    f.transport_model = 'mixture-averaged'
    # f.transport_model = 'multicomponent'
    f.soret_enabled = False

    f.set_max_jac_age(10, 10)
    f.set_time_step(1e-5, [2, 5, 10, 20])

    f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)

    try:
        f.solve(loglevel=loglevel, auto=True)
    except ct.CanteraError:
        print(f" WARNING: Computation crashed for T0={temperature_ini}, phi={phi_ini} => skipped")
        return 1
    
    # x axis
    X_grid = f.grid
    nb_0_reactors = len(X_grid) # number of reactors

    # Mass fractions at time t
    Yt = f.Y

    # Temperature at time t
    Tt = f.T

    # Initializing Y at t+dt
    Yt_dt_exact = np.zeros(Yt.shape)
    Yt_dt_ann = np.zeros(Yt.shape)


    # COMPUTE EXACT RR ----------------------------------------------------------------

    for i_reac in range(nb_0_reactors):
            
        gas.TPY = Tt[i_reac], p, Yt[:,i_reac]

        r = ct.IdealGasConstPressureReactor(gas)
            
        # Initializing reactor
        sim = ct.ReactorNet([r])
        time = 0.0
        states = ct.SolutionArray(gas, extra=['t'])
        
        # We advance solution by dt
        time = dt
        sim.advance(time)
        states.append(r.thermo.state, t=time * 1e3)
        
        Yt_dt_exact[:,i_reac] = states.Y
        
    # Reaction rates
    Omega_exact = (Yt_dt_exact-Yt)/dt
    Omega_exact = Omega_exact.transpose()
    
    # COMPUTE NN RR ----------------------------------------------------------------

    # Mass and element conservation
    Omega_e = np.empty((nb_0_reactors,4))

    for i_reac in range(nb_0_reactors): 
        
        state_current = np.append(Tt[i_reac], Yt[:,i_reac])

        # Gas object modification
        gas.TPY= state_current[0], p, state_current[1:]

        # attributing cluster
        i_cluster = attribute_cluster(kmeans, kmeans_scaler, state_current, log_transform, threshold)
        print(f">> Current point in cluster {i_cluster}")

        # Evaluation mode
        model = model_list[i_cluster]
        model.eval()

        T_new, Y_new = advance_ANN(state_current, model, Xscaler, Yscaler, gas, log_transform, threshold, device)

        Yt_dt_ann[:,i_reac] = np.reshape(Y_new,-1)

        omega_current = (Y_new - Yt[:,i_reac])/dt

        # Elements conservation
        Omega_e[i_reac,:] = np.dot(A_element, omega_current.transpose()).ravel()

    # Reaction rates
    Omega_ann = (Yt_dt_ann-Yt)/dt
    Omega_ann = Omega_ann.transpose()

    # Mass conservation
    sum_Omega_k = Omega_ann.sum(axis=1)
    
    # Generate pandas dataframes with solutions -----------------------------------
        
    # EXACT
    n_rows = nb_0_reactors
    # time / temperature / species
    n_cols = 1 + nb_spec
    arr = np.empty(shape=(n_rows,n_cols))
    arr[:, 0] = X_grid
    arr[:, 1:1 + nb_spec] = Omega_exact


    cols = ['X'] + gas.species_names
    df_exact = pd.DataFrame(data=arr, columns=cols)

    # ANN
    cols_nn = ['X'] + gas.species_names + ["SumOmegak", "Omega_C", "Omega_H", "Omega_O", "Omega_N"]
    arr_nn = np.hstack([X_grid.reshape(-1,1), Omega_ann, sum_Omega_k.reshape(-1,1), Omega_e])
    df_ann = pd.DataFrame(data=arr_nn, columns=cols_nn)

    return df_exact, df_ann




#-------------------------------------------------------------------
# 1D diffusion flame
#-------------------------------------------------------------------

def compute_nn_cantera_1D_diff(device, kmeans, kmeans_scaler, model_list, Xscaler, Yscaler, strain, T0, width, dt, dtb_params, A_element):

    log_transform = dtb_params["log_transform"]
    threshold = dtb_params["threshold"]
    fuel = dtb_params["fuel"]
    mech_file = dtb_params["mech_file"]
    p = dtb_params["p"]

    # CANTERA gas object
    gas = ct.Solution(mech_file)

    species_list = gas.species_names
    nb_spec = len(species_list)

    # COMPUTE 1D PREMIXED FLAME -------------------------------------------------------

    gas.TP = T0, p

    # Stream compositions
    compo_ini_o = 'O2:0.21, N2:0.79'
    compo_ini_f = f'{fuel}:1'

    gas.TPX = T0, p, compo_ini_o
    density_o = gas.density
    gas.TPX = 300.0, p, compo_ini_f
    density_f = gas.density

    # Stream mass flow rates
    vel = strain * width / 2.0
    mdot_o = density_o * vel
    mdot_f = density_f * vel

    f = ct.CounterflowDiffusionFlame(gas, width=width)
    
    f.transport_model = 'mixture-averaged'
    # f.transport_model = 'multicomponent'
    f.soret_enabled = False

    f.radiation_enabled = False

    f.set_max_jac_age(10, 10)
    f.set_time_step(1e-5, [2, 5, 10, 20])

    # Set the state of the two inlets
    f.fuel_inlet.mdot = mdot_f
    f.fuel_inlet.X = compo_ini_f
    f.fuel_inlet.T = 300.0

    f.oxidizer_inlet.mdot = mdot_o
    f.oxidizer_inlet.X = compo_ini_o
    f.oxidizer_inlet.T = T0

    f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)

    try:
        loglevel = 0  # amount of diagnostic output (0 to 8)
        f.solve(loglevel=loglevel, auto=True)
    except ct.CanteraError:
        print(f" WARNING: Computation crashed for T0_ox={T0}, a={strain} => skipped")
        return 1
    
    # x axis
    X_grid = f.grid
    nb_0_reactors = len(X_grid) # number of reactors

    # Mass fractions at time t
    Yt = f.Y

    # Temperature at time t
    Tt = f.T

    # Initializing Y at t+dt
    Yt_dt_exact = np.zeros(Yt.shape)
    Yt_dt_ann = np.zeros(Yt.shape)


    # COMPUTE EXACT RR ----------------------------------------------------------------

    for i_reac in range(nb_0_reactors):
            
        gas.TPY = Tt[i_reac], p, Yt[:,i_reac]

        r = ct.IdealGasConstPressureReactor(gas)
            
        # Initializing reactor
        sim = ct.ReactorNet([r])
        time = 0.0
        states = ct.SolutionArray(gas, extra=['t'])
        
        # We advance solution by dt
        time = dt
        sim.advance(time)
        states.append(r.thermo.state, t=time * 1e3)
        
        Yt_dt_exact[:,i_reac] = states.Y
        
    # Reaction rates
    Omega_exact = (Yt_dt_exact-Yt)/dt
    Omega_exact = Omega_exact.transpose()
    
    # COMPUTE NN RR ----------------------------------------------------------------

    # Mass and element conservation
    Omega_e = np.empty((nb_0_reactors,4))

    for i_reac in range(nb_0_reactors): 
        
        state_current = np.append(Tt[i_reac], Yt[:,i_reac])

        # Gas object modification
        gas.TPY= state_current[0], p, state_current[1:]

        # attributing cluster
        i_cluster = attribute_cluster(kmeans, kmeans_scaler, state_current, log_transform, threshold)
        print(f">> Current point in cluster {i_cluster}")

        # Evaluation mode
        model = model_list[i_cluster]
        model.eval()

        T_new, Y_new = advance_ANN(state_current, model, Xscaler, Yscaler, gas, log_transform, threshold, device)

        Yt_dt_ann[:,i_reac] = np.reshape(Y_new,-1)

        omega_current = (Y_new - Yt[:,i_reac])/dt

        # Elements conservation
        Omega_e[i_reac,:] = np.dot(A_element, omega_current.transpose()).ravel()

    # Reaction rates
    Omega_ann = (Yt_dt_ann-Yt)/dt
    Omega_ann = Omega_ann.transpose()

    # Mass conservation
    sum_Omega_k = Omega_ann.sum(axis=1)
    
    # Generate pandas dataframes with solutions -----------------------------------
        
    # EXACT
    n_rows = nb_0_reactors
    # time / temperature / species
    n_cols = 1 + nb_spec
    arr = np.empty(shape=(n_rows,n_cols))
    arr[:, 0] = X_grid
    arr[:, 1:1 + nb_spec] = Omega_exact


    cols = ['X'] + gas.species_names
    df_exact = pd.DataFrame(data=arr, columns=cols)

    # ANN
    cols_nn = ['X'] + gas.species_names + ["SumOmegak", "Omega_C", "Omega_H", "Omega_O", "Omega_N"]
    arr_nn = np.hstack([X_grid.reshape(-1,1), Omega_ann, sum_Omega_k.reshape(-1,1), Omega_e])
    df_ann = pd.DataFrame(data=arr_nn, columns=cols_nn)

    return df_exact, df_ann



#-------------------------------------------------------------------
# GENERIC FUNCTIONS
#-------------------------------------------------------------------


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


def attribute_cluster(kmeans, kmeans_scaler, state_vector, log_transform, threshold):
        
    log_state = state_vector.copy()
        
    # Transformation
    if log_transform:
        log_state[log_state < threshold] = threshold
        log_state[1:] = np.log(log_state[1:])

    # Scaling vector
    vect_scaled = kmeans_scaler.transform(log_state)
    # Applying k-means
    i_cluster = kmeans.predict(vect_scaled.to_numpy().reshape(1, -1))[0]

    return i_cluster