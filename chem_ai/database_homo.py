import os, sys
from scipy.stats import qmc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct

import seaborn as sns
sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split


class Database_HomoReac(object):

    def __init__(self, mech_file, fuel, folder, p, phi_bounds, T0_bounds, n_samples, dt_cfd, max_sim_time, solve_mode, multi_dt, nb_dt, node_sampling, rdm_sample):

        self.mech_file = mech_file
        self.fuel = fuel
        self.solve_mode = solve_mode
        self.dt_cfd = dt_cfd

        self.folder = folder
        if not os.path.isdir(folder):
            os.mkdir(folder)
        else:
            sys.exit("Folder already exists")

        self.p = p
        self.phi_bounds = phi_bounds
        self.T0_bounds = T0_bounds

        self.n_samples = n_samples
        self.sim_numbers = np.arange(n_samples)

        # Generate DOE
        self._generate_doe()

        # Run 0D reactors
        self._run_0D_reactors(max_sim_time)

        # Multiple time steps
        self.multi_dt = multi_dt
        self.nb_dt = nb_dt
        self.node_sampling = node_sampling
        self.rdm_sample = rdm_sample



    def _generate_doe(self):
        
        # Initially lhs gives numbers between 0 and 1
        seed = 42
        sampler = qmc.LatinHypercube(d=2, seed=seed) 
        samples = sampler.random(n=self.n_samples)
        self.df_ODE = pd.DataFrame(data=samples, columns=['Phi', 'T0'])
        
        self.df_ODE["sim_number"] = self.sim_numbers 
        
        # Rescale 
        self.df_ODE['Phi'] = self.phi_bounds[0] + (self.phi_bounds[1] - self.phi_bounds[0]) * self.df_ODE['Phi']
        self.df_ODE['T0'] = self.T0_bounds[0] + (self.T0_bounds[1] - self.T0_bounds[0]) * self.df_ODE['T0']




    def _run_0D_reactors(self, max_sim_time):

        # Hard coded for the moment: parameter to control simulations stopping criterion
        equil_tol = 0.5

        # Chemical mechanisms
        gas = ct.Solution(self.mech_file)
        gas_equil = ct.Solution(self.mech_file)

        data=[]

        fig, ax = plt.subplots()
        ax.set_xlabel("t [s]")
        ax.set_ylabel("T [K]")

        for i, row in self.df_ODE.iterrows():

            phi_ini = row['Phi']
            temperature_ini = row['T0']

            # Tolerance on temperature slope
            dT_dt_tol = 0.1 / self.dt_cfd

            print(f"T0={temperature_ini}; phi={phi_ini}")

            # Initial gas state
            fuel_ox_ratio = gas.n_atoms(species=self.fuel, element='C') \
                            + 0.25 * gas.n_atoms(species=self.fuel, element='H') \
                            - 0.5 * gas.n_atoms(species=self.fuel, element='O')
            compo_ini = f'{self.fuel}:{phi_ini:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
            gas.TPX = temperature_ini, self.p, compo_ini

            nb_spec = len(gas.X)   # Number of species
            Y0 = gas.Y   # Initial mass fractions

            # 0D reactor
            r = ct.IdealGasConstPressureReactor(gas)

            # Initializing reactor
            sim = ct.ReactorNet([r])
            time = 0.0
            states = ct.SolutionArray(gas, extra=['t'])

            # Computing equilibrium (to get end of simulation criterion)
            gas_equil.TPX = temperature_ini, self.p, compo_ini
            gas_equil.equilibrate('HP')
            state_equil = np.append(gas_equil.X, gas_equil.T)

            equil_bool = False
            n_iter = 0

            while (equil_bool == False) and (time < max_sim_time):

                T_m1 = r.T
                
                if self.solve_mode=="dt_cfd":
                    time += self.dt_cfd
                    sim.advance(time)
                    states.append(r.thermo.state, t=time)
                elif self.solve_mode=="dt_cvode":
                    time_m1 = time
                    t_cvode = sim.step()
                    time = t_cvode
                    if n_iter%5==0:   # dt_cvode gives too many points
                        states.append(r.thermo.state, t=t_cvode)
                else:
                    sys.exit("solve_mode should be dt_cvode or dt_cfd")

                # checking if equilibrium is reached
                
                state_current = np.append(r.thermo.X, r.T)
                residual = 100.0*np.linalg.norm(state_equil - state_current,ord=np.inf)/np.linalg.norm(state_equil,
                                                                                                           ord=np.inf)
                
                # Checking that temperature profile is flat (else if T is not monotonic, we have issues)
                if self.solve_mode=="dt_cfd":
                    dt = self.dt_cfd
                elif self.solve_mode=="dt_cvode":
                    dt = time- time_m1
                
                dT_dt = (r.T-T_m1)/dt

                
                n_iter +=1
                # max iteration                    
                if residual < equil_tol and np.abs(dT_dt)<dT_dt_tol:
                    equil_bool = True
            
            # ============================== Construction of the database =========
            # Get the total number of rows for the current simulation
            n_rows = len(states.t) + 1
            # temperature / pressure / Y / time / sim number
            n_cols = 1 + 1 + nb_spec + 1 + 1  
            # empty array saving
            arr = np.empty(shape=(n_rows,n_cols))
            #  initial conditions
            arr[0,0] = temperature_ini
            arr[0,1] = self.p
            arr[0,2:2+nb_spec] = Y0
            arr[0,-2] = 0
            arr[0,-1] = i
            #  solution for each time step (each couple of S(t) and S(t+dt) for the dtCVODE case)
            arr[1:, 0] = states.T
            arr[1:, 1] = states.P
            arr[1:, 2:2 + nb_spec] = states.Y
            arr[1:, -2] = states.t
            arr[1:, -1] = i

            # Save in pandas dataframe
            cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['Time'] + ['Simulation number']
            df = pd.DataFrame(data=arr, columns=cols)

            # List of dataframes
            data.append(df)

            # Add to plot
            ax.plot(states.t, states.T)

        # Rearrange the whole data list to a single data file with many different simulations which is predefined with ICs
        self.data_simu = pd.concat(data, axis=0).reset_index(drop=True) 

        self.data_simu.to_csv(os.path.join(self.folder,"0d_runs.csv"))

        fig.savefig(os.path.join(self.folder,"0D_trajectories.png"))




    def generate_train_valid_test(self, valid_ratio, test_ratio, dt_min, dt_max, extend_database):

        np.random.seed(2025)

        # In this function we use the 0D reactors runs to build the X_train, Y_train, X_val, Y_val, X_test, Y_test data
        # Train, validation and test are based on simulations numbers, not states

        self.id_sim_train, self.id_sim_val = train_test_split(self.sim_numbers, test_size=valid_ratio, random_state=24)
        self.id_sim_train, self.id_sim_test = train_test_split(self.id_sim_train, test_size=test_ratio, random_state=12)

        # Save ids to a file, including initial conditions (will be useful later for test)
        # create and save the training doe data IC index 
        self.df_ODE_train = self.df_ODE[self.df_ODE["sim_number"].isin(self.id_sim_train)]
        self.df_ODE_val = self.df_ODE[self.df_ODE["sim_number"].isin(self.id_sim_val)]
        self.df_ODE_test = self.df_ODE[self.df_ODE["sim_number"].isin(self.id_sim_test)]
        #
        self.df_ODE_train.to_csv(os.path.join(self.folder,"sim_train.csv"), index=False)
        self.df_ODE_val.to_csv(os.path.join(self.folder,"sim_val.csv"), index=False)
        self.df_ODE_test.to_csv(os.path.join(self.folder,"sim_test.csv"), index=False)


        fig, ax = plt.subplots()
        ax.scatter(self.df_ODE_train['T0'], self.df_ODE_train['Phi'], color="k", label="Train")
        ax.scatter(self.df_ODE_val['T0'], self.df_ODE_val['Phi'], color="b", label="Validation")
        ax.scatter(self.df_ODE_test['T0'], self.df_ODE_test['Phi'], color="r", label="Test")
        fig.legend(ncol=3)
        ax.set_xlabel("T0 [K]")
        ax.set_ylabel("Phi [-]")
        fig.savefig(os.path.join(self.folder,"doe_0D_reactors.png"))

        # DOE for multi_dt
        if self.multi_dt:
            
            # If database extension
            fact_max = extend_database
            fact_min = 1.0 - (extend_database - 1.0)

            #If NODE, we set the same dt's for each sample, to ease the NODE integration and the loss function calculation
            if self.node_sampling:
                self.dt_array = np.empty((self.data_simu.shape[0], self.nb_dt))
                dt_vect = np.linspace(np.log(dt_min), np.log(dt_max), self.nb_dt)
                dt_vect = np.exp(dt_vect)
                for k in range(self.dt_array.shape[0]):
                    self.dt_array[k,:] = dt_vect
            else:
                # Normal version
                # self.dt_array = np.random.uniform(low=0.0, high=dt_min, size=(self.data_simu.shape[0], self.nb_dt))
                # Log version
                # self.dt_array = np.random.uniform(low=np.log(dt_min), high=np.log(dt_max), size=(self.data_simu.shape[0], self.nb_dt))
                # self.dt_array = np.exp(self.dt_array)
                # #
                # # If NODE sampling, we sort the dt's
                # if self.node_sampling:
                #     self.dt_array = np.sort(self.dt_array, axis=1)

                # Version imposing lowest dt   -> This should not be needed
                if self.rdm_sample:
                    # self.dt_array = np.empty(((self.data_simu.shape[0], self.nb_dt)))
                    # self.dt_array[:,0] = dt_min * np.ones(self.data_simu.shape[0])
                    # if self.nb_dt>1:
                    #     self.dt_array[:,1:] = np.random.uniform(low=np.log(dt_min), high=np.log(dt_max), size=(self.data_simu.shape[0], self.nb_dt-1))
                    #     self.dt_array[:,1:] = np.exp(self.dt_array[:,1:])

                    self.dt_array = np.random.uniform(low=np.log(fact_min*dt_min), high=np.log(fact_max*dt_max), size=(self.data_simu.shape[0], self.nb_dt))
                    self.dt_array = np.exp(self.dt_array)
                else:
                    self.dt_array = np.empty((self.data_simu.shape[0], self.nb_dt))
                    dt_vect = np.linspace(np.log(dt_min), np.log(dt_max), self.nb_dt)
                    dt_vect = np.exp(dt_vect)
                    for k in range(self.dt_array.shape[0]):
                        self.dt_array[k,:] = dt_vect


        if self.multi_dt:
            self.dt_array_train, self.X_train, self.Y_train = self._get_X_Y_multi_dt(self.id_sim_train)
            self.dt_array_val, self.X_val, self.Y_val = self._get_X_Y_multi_dt(self.id_sim_val)
        else:
            self.X_train, self.Y_train = self._get_X_Y(self.id_sim_train)
            self.X_val, self.Y_val = self._get_X_Y(self.id_sim_val)

        # Shuffle data
        permutation_train = np.random.permutation(self.X_train.shape[0])
        if self.multi_dt:
            self.X_train = self.X_train[permutation_train]
            self.Y_train = self.Y_train[permutation_train]
            self.dt_array_train = self.dt_array_train[permutation_train]
        else:
            self.X_train = self.X_train.iloc[permutation_train].reset_index(drop=True)
            self.Y_train = self.Y_train.iloc[permutation_train].reset_index(drop=True)
        #
        permutation_val = np.random.permutation(self.X_val.shape[0])
        if self.multi_dt:
            self.X_val = self.X_val[permutation_val]
            self.Y_val = self.Y_val[permutation_val]
            self.dt_array_val = self.dt_array_val[permutation_val]
        else:
            self.X_val = self.X_val.iloc[permutation_val].reset_index(drop=True)
            self.Y_val = self.Y_val.iloc[permutation_val].reset_index(drop=True)

        # Saving data (raw data)
        if self.multi_dt: # storing 3D tensor in npy format (Y is a 3D tensor)
            np.save(os.path.join(self.folder,"X_train_raw.npy"), self.X_train)
            np.save(os.path.join(self.folder,"Y_train_raw.npy"), self.Y_train)
            np.save(os.path.join(self.folder,"X_val_raw.npy"), self.X_val)
            np.save(os.path.join(self.folder,"Y_val_raw.npy"), self.Y_val)

            # Saving dt's
            np.save(os.path.join(self.folder,"dt_array_train.npy"), self.dt_array_train)
            np.save(os.path.join(self.folder,"dt_array_val.npy"), self.dt_array_val)

        else: # csv
            self.X_train.to_csv(os.path.join(self.folder,"X_train_raw.csv"), index=False)
            self.Y_train.to_csv(os.path.join(self.folder,"Y_train_raw.csv"), index=False)
            self.X_val.to_csv(os.path.join(self.folder,"X_val_raw.csv"), index=False)
            self.Y_val.to_csv(os.path.join(self.folder,"Y_val_raw.csv"), index=False)




    def _get_X_Y(self, simu_ids):

        X_list = []
        Y_list = []

        # Cantera gas object
        self.gas = ct.Solution(self.mech_file)
        
        for i in simu_ids:

            print(f">> Building X, Y for simulation {i}")

            df_i = self.data_simu[self.data_simu['Simulation number'] == i].iloc[:, :-1]

            cols = df_i.columns

            if self.solve_mode=="dt_cfd":
                X = df_i.iloc[0:-1, :].reset_index(drop=True)
                # get the progress variable in dataset index if needed
                X.columns = [str(col) + '_X' for col in cols]
                self.X_cols = X.columns.tolist()

                Y = df_i.iloc[1:,:].reset_index(drop=True) # predict Y(t+dt) directl
                Y.columns = [str(col) + '_Y' for col in cols]
                self.Y_cols = Y.columns.tolist()

            elif self.solve_mode=="dt_cvode":
                X = df_i.copy()
                X.columns = [str(col) + '_X' for col in cols]
                self.X_cols = X.columns.tolist()

                Y_np = self._advance_dt_cfd(X, self.dt_cfd)
                Y_columns = [str(col) + '_Y' for col in cols]
                Y = pd.DataFrame(data=Y_np, columns = Y_columns)
                self.Y_cols = Y.columns.tolist()


            X_list.append(X)
            Y_list.append(Y)

        X = pd.concat(X_list,axis=0).reset_index(drop=True)
        Y = pd.concat(Y_list,axis=0).reset_index(drop=True)

        # Remove non needed items
        list_to_remove = ['Pressure_X', 'Time_X']
        [self.X_cols.remove(elt) for elt in list_to_remove]   
        X = X[self.X_cols]

        # Remove non needed lables for Y
        list_to_remove = ['Pressure_Y', 'Time_Y', 'Temperature_Y']   # Temperature computed from enthalpy conservation
        # Removing unwanted items
        [self.Y_cols.remove(elt) for elt in list_to_remove]   
        Y= Y[self.Y_cols]

        return X, Y
    


    def _get_X_Y_multi_dt(self, simu_ids):

        # Cantera gas object
        self.gas = ct.Solution(self.mech_file)

        # Selecting based on simulations ids
        df_ids = self.data_simu[self.data_simu['Simulation number'].isin(simu_ids)].iloc[:, :-1]
        dt_array_red = self.dt_array[self.data_simu['Simulation number'].isin(simu_ids)]

        # Initializing X and Y
        X = np.empty((df_ids.shape[0],df_ids.shape[1]-2)) # -2 because we remove Pressure and Time
        if self.node_sampling:
            Y = np.empty((df_ids.shape[0],df_ids.shape[1]-2,self.nb_dt))
        else:
            Y = np.empty((df_ids.shape[0],df_ids.shape[1]-3,self.nb_dt)) # -2 because we remove Pressure, Time and temperature

        # Filling X
        X[:,0] = df_ids["Temperature"]
        X[:,1:] = df_ids.iloc[:,2:-1]


        for i in range(X.shape[0]):

            if self.node_sampling:  # We compute a single trajectory along all dt's

                T = X[i,0]
                Yk = X[i,1:]

                self.gas.TPY = T, self.p, Yk

                # Constant pressure reactor
                r = ct.IdealGasConstPressureReactor(self.gas)

                # Initializing reactor
                sim = ct.ReactorNet([r])
                
                for i_dt in range(self.nb_dt):

                    dt = dt_array_red[i,i_dt]

                    # Advancing to dt (each time from previous dt value thus advancing on the trajectory)
                    sim.advance(dt)

                    # Updated state
                    Y[i,0,i_dt] = self.gas.T
                    Y[i,1:,i_dt] = self.gas.Y

            else:  #We compute solution at t+dt for each dt, starting from initial state

                for i_dt in range(self.nb_dt):

                    dt = dt_array_red[i,i_dt]

                    T = X[i,0]
                    Yk = X[i,1:]

                    self.gas.TPY = T, self.p, Yk

                    # Constant pressure reactor
                    r = ct.IdealGasConstPressureReactor(self.gas)
                    
                    # Initializing reactor
                    sim = ct.ReactorNet([r])
                    
                    # Advancing to dt
                    sim.advance(dt)

                    # Updated state
                    Y[i,:,i_dt] = self.gas.Y


        return dt_array_red, X, Y
    


    def _advance_dt_cfd(self, X, dt):

        Y = np.empty(X.shape)

        k=0
        for i, row in X.iterrows():
            
            T = row[0]
            p = row[1]
            Yk = row[2:-1].values
            time = row[-1]

            self.gas.TPY = T, p, Yk

            # Constant pressure reactor
            r = ct.IdealGasConstPressureReactor(self.gas)
            
            # Initializing reactor
            sim = ct.ReactorNet([r])
            
            # Advancing to dt
            sim.advance(dt)

            # Updated state
            Y[k,0] = self.gas.T
            Y[k,1] = self.gas.P
            Y[k,2:-1] = self.gas.Y
            Y[k,-1] = time + dt

            k+=1


        return Y










