import os, sys
import pyDOE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct

import seaborn as sns
sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split

import chem_ai.utils as utils


class DatabaseFlamelets(object):


    def __init__(self, mech_file, fuel, folder, p, dt_cfd):

            self.mech_file = mech_file
            self.fuel = fuel
            self.dt_cfd = dt_cfd

            self.is_augmented = False

            self.folder = folder
            if not os.path.isdir(folder):
                os.mkdir(folder)
            else:
                sys.exit("Folder already exists")

            # We work with constant pressure here
            self.p = p

            # Initialize solution array 
            gas = ct.Solution(self.mech_file)
            self.nb_spec = len(gas.species_names)
            cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['reactor_type'] + ['Simulation number']
            self.df = pd.DataFrame(data=[], columns=cols)



    def compute_0d_reactors(self, phi_bounds, T0_bounds, n_samples, max_sim_time, solve_mode = "dt_cfd"):
         
        self.n_samples_0D = n_samples
        self.sim_numbers_0D = np.arange(n_samples)

         # DOE 0D REACTORS
         # Initially lhs gives numbers between 0 and 1
        self.df_ODE_0D = pd.DataFrame(data=pyDOE.lhs(n=2, samples=n_samples, criterion='maximin'), columns=['Phi', 'T0'])
        
        self.df_ODE_0D["sim_number"] = self.sim_numbers_0D 
        
        # Rescale 
        self.df_ODE_0D['Phi'] = phi_bounds[0] + (phi_bounds[1] - phi_bounds[0]) * self.df_ODE_0D['Phi']
        self.df_ODE_0D['T0'] = T0_bounds[0] + (T0_bounds[1] - T0_bounds[0]) * self.df_ODE_0D['T0']

         
        # PERFORMING SIMULATIONS
        # Hard coded for the moment
        equil_tol = 0.5

        # Chemical mechanisms
        gas = ct.Solution(self.mech_file)
        gas_equil = ct.Solution(self.mech_file)

        data = []

        fig, ax = plt.subplots()
        ax.set_xlabel("t [s]")
        ax.set_ylabel("T [K]")

        for i, row in self.df_ODE_0D.iterrows():

            phi_ini = row['Phi']
            temperature_ini = row['T0']

            print(f"T0={temperature_ini}; phi={phi_ini}")

            # Initial gas state
            fuel_ox_ratio = gas.n_atoms(species=self.fuel, element='C') \
                            + 0.25 * gas.n_atoms(species=self.fuel, element='H') \
                            - 0.5 * gas.n_atoms(species=self.fuel, element='O')
            compo_ini = f'{self.fuel}:{phi_ini:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
            gas.TPX = temperature_ini, self.p, compo_ini

            nb_spec = len(gas.X)   # Number of species
            Y0 = gas.Y   # Initial mass fractions
            h0 = gas.HP[0]

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
                
                if solve_mode=="dt_cfd":
                    time += self.dt_cfd
                    sim.advance(time)
                    states.append(r.thermo.state, t=time)
                elif solve_mode=="dt_cvode":
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
                
                n_iter +=1
                # max iteration                    
                if residual < equil_tol:
                    equil_bool = True
            
            # ============================== Construction of the database =========
            # Get the total number of rows for the current simulation
            n_rows = len(states.t) + 1
            # temperature / pressure / Y / sim number
            n_cols = 1 + 1 + nb_spec + 1 + 1 + 1
            # empty array saving
            arr = np.empty(shape=(n_rows,n_cols))
            #  initial conditions
            arr[0,0] = temperature_ini
            arr[0,1] = self.p
            arr[0,2:2+nb_spec] = Y0
            arr[0,-3] = h0
            arr[0,-2] = 0
            arr[0,-1] = i
            #  solution for each time step (each couple of S(t) and S(t+dt) for the dtCVODE case)
            arr[1:, 0] = states.T
            arr[1:, 1] = states.P
            arr[1:, 2:2 + nb_spec] = states.Y
            arr[1:, -3] = states.enthalpy_mass
            arr[1:, -2] = 0
            arr[1:, -1] = i

            # Save in pandas dataframe
            cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['reactor_type'] + ['Simulation number']
            df = pd.DataFrame(data=arr, columns=cols)

            # List of dataframes
            data.append(df)

            # Add to plot
            ax.plot(states.t, states.T)

        #Add to self.df
        df_ODE_0D = pd.concat(data, axis=0).reset_index(drop=True) 
        self.df = pd.concat([self.df, df_ODE_0D], axis=0).reset_index(drop=True) 

        fig.savefig(os.path.join(self.folder,"0D_trajectories.png"))




    def compute_1d_premixed(self, phi_bounds, T0_bounds, n_samples):
         
        self.n_samples_1D_prem = n_samples
        self.sim_numbers_1D_prem = np.arange(n_samples)

        # DOE 0D REACTORS
        # Initially lhs gives numbers between 0 and 1
        self.df_ODE_1D_prem = pd.DataFrame(data=pyDOE.lhs(n=2, samples=n_samples, criterion='maximin'), columns=['Phi', 'T0'])
        
        self.df_ODE_1D_prem["sim_number"] = self.sim_numbers_1D_prem 
        
        # Rescale 
        self.df_ODE_1D_prem['Phi'] = phi_bounds[0] + (phi_bounds[1] - phi_bounds[0]) * self.df_ODE_1D_prem['Phi']
        self.df_ODE_1D_prem['T0'] = T0_bounds[0] + (T0_bounds[1] - T0_bounds[0]) * self.df_ODE_1D_prem['T0']

         
        # PERFORMING SIMULATIONS

        # Chemical mechanism
        gas = ct.Solution(self.mech_file)

        data = []

        fig, ax = plt.subplots()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("T [K]")

        initial_grid = np.linspace(0.0, 0.03, 10)  # m
        tol_ss = [1.0e-4, 1.0e-9]  # [rtol atol] for steady-state problem
        tol_ts = [1.0e-5, 1.0e-5]  # [rtol atol] for time stepping
        loglevel = 0  # amount of diagnostic output (0 to 8)

        for i, row in self.df_ODE_1D_prem.iterrows():

            crashed = False

            phi_ini = row['Phi']
            temperature_ini = row['T0']

            print(f"T0={temperature_ini}; phi={phi_ini}")

            # Initial gas state
            fuel_ox_ratio = gas.n_atoms(species=self.fuel, element='C') \
                            + 0.25 * gas.n_atoms(species=self.fuel, element='H') \
                            - 0.5 * gas.n_atoms(species=self.fuel, element='O')
            compo_ini = f'{self.fuel}:{phi_ini:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
            gas.TPX = temperature_ini, self.p, compo_ini

            nb_spec = len(gas.X)   # Number of species

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
                crashed = True

            if crashed==False:
            
                # ============================== Construction of the database =========
                # Get the total number of rows for the current simulation
                n_rows = len(f.grid)
                # temperature / pressure / Y / sim number
                n_cols = 1 + 1 + nb_spec + 1 + 1 + 1
                # empty array saving
                arr = np.empty(shape=(n_rows,n_cols))
                #  solution for each time step (each couple of S(t) and S(t+dt) for the dtCVODE case)
                arr[:, 0] = f.T
                arr[:, 1] = f.P
                arr[:, 2:2 + nb_spec] = np.transpose(f.Y)
                arr[:, -3] = f.enthalpy_mass
                arr[:, -2] = 1
                arr[:, -1] = i

                # Save in pandas dataframe
                cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['reactor_type'] + ['Simulation number']
                df = pd.DataFrame(data=arr, columns=cols)

                # List of dataframes
                data.append(df)

                # Add to plot
                ax.plot(f.grid, f.T)

        # Add to self.df
        df_ODE_1D_prem = pd.concat(data, axis=0).reset_index(drop=True) 
        self.df = pd.concat([self.df, df_ODE_1D_prem], axis=0).reset_index(drop=True) 

        fig.savefig(os.path.join(self.folder,"1D_prem_trajectories.png"))




    def compute_1d_diffusion(self, strain_bounds, T0_bounds, n_samples, width):
         
        self.n_samples_1D_diff = n_samples
        self.sim_numbers_1D_diff = np.arange(n_samples)

        # DOE 0D REACTORS
        # Initially lhs gives numbers between 0 and 1
        self.df_ODE_1D_diff = pd.DataFrame(data=pyDOE.lhs(n=2, samples=n_samples, criterion='maximin'), columns=['Strain', 'T0'])
        
        self.df_ODE_1D_diff["sim_number"] = self.sim_numbers_1D_diff 
        
        # Rescale 
        self.df_ODE_1D_diff['Strain'] = strain_bounds[0] + (strain_bounds[1] - strain_bounds[0]) * self.df_ODE_1D_diff['Strain']
        self.df_ODE_1D_diff['T0'] = T0_bounds[0] + (T0_bounds[1] - T0_bounds[0]) * self.df_ODE_1D_diff['T0']

         
        # PERFORMING SIMULATIONS

        # Chemical mechanism
        gas = ct.Solution(self.mech_file)

        data = []

        fig, ax = plt.subplots()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("T [K]")

        loglevel = 0  # amount of diagnostic output (0 to 8)

        for i, row in self.df_ODE_1D_diff.iterrows():

            crashed = False

            strain = row['Strain']
            temperature_ini = row['T0']

            print(f"T0_ox={temperature_ini}; strain={strain}")

            gas.TP = gas.T, self.p

            nb_spec = len(gas.X)   # Number of species

            # Stream compositions
            compo_ini_o = 'O2:0.21, N2:0.79'
            compo_ini_f = f'{self.fuel}:1'

            gas.TPX = temperature_ini, self.p, compo_ini_o
            density_o = gas.density
            gas.TPX = 300.0, self.p, compo_ini_f
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
            f.oxidizer_inlet.T = temperature_ini

            f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)

            try:
                f.solve(loglevel=loglevel, auto=True)
            except ct.CanteraError:
                print(f" WARNING: Computation crashed for T0_ox={temperature_ini}, a={strain} => skipped")
                crashed = True

            if crashed==False:
                # ============================== Construction of the database =========
                # Get the total number of rows for the current simulation
                n_rows = len(f.grid)
                # temperature / pressure / Y / sim number
                n_cols = 1 + 1 + nb_spec + 1 + 1 + 1
                # empty array saving
                arr = np.empty(shape=(n_rows,n_cols))
                #  solution for each time step (each couple of S(t) and S(t+dt) for the dtCVODE case)
                arr[:, 0] = f.T
                arr[:, 1] = f.P
                arr[:, 2:2 + nb_spec] = np.transpose(f.Y)
                arr[:, -3] = f.enthalpy_mass
                arr[:, -2] = 2
                arr[:, -1] = i

                # Save in pandas dataframe
                cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['reactor_type'] + ['Simulation number']
                df = pd.DataFrame(data=arr, columns=cols)

                # List of dataframes
                data.append(df)

                # Add to plot
                ax.plot(f.grid, f.T)

        #Add to self.df
        df_ODE_1D_diff = pd.concat(data, axis=0).reset_index(drop=True) 
        self.df = pd.concat([self.df, df_ODE_1D_diff], axis=0).reset_index(drop=True) 

        fig.savefig(os.path.join(self.folder,"1D_diff_trajectories.png"))



    def augment_data(self):

        self.df_flamelet = self.df.copy()

        gas = ct.Solution(self.mech_file)

        cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['reactor_type'] + ['Simulation number']
        self.df_augmented = pd.DataFrame(data=[], columns=cols)



        max_values_0D = self.df[self.df["reactor_type"]==0].max()    # We should probably do it by simulation type !!
        min_values_0D = self.df[self.df["reactor_type"]==0].min()

        max_values_1D_prem = self.df[self.df["reactor_type"]==1].max()    # We should probably do it by simulation type !!
        min_values_1D_prem = self.df[self.df["reactor_type"]==1].min()

        max_values_1D_diff = self.df[self.df["reactor_type"]==2].max()    # We should probably do it by simulation type !!
        min_values_1D_diff = self.df[self.df["reactor_type"]==2].min()

        # Data augmentation based on the work of Ding et al. (HFRD method)
        for i, row in self.df.iterrows():
            
            tries = 0
            accepted = False
            while tries<=10 and accepted==False:
                
                tries += 1
                accepted = True

                T = row.iloc[0]
                p = row.iloc[1]
                Yk = row.iloc[2:2 + self.nb_spec].values
                h = row.iloc[2 + self.nb_spec]
                i_reac = row.iloc[3 + self.nb_spec]


                # Bounds depending on reactor type
                if i_reac==0:
                    hmax = max_values_0D.iloc[2 + self.nb_spec]
                    hmin = min_values_0D.iloc[2 + self.nb_spec]
                elif i_reac==1:
                    hmax = max_values_1D_prem.iloc[2 + self.nb_spec]
                    hmin = min_values_1D_prem.iloc[2 + self.nb_spec]
                elif i_reac==2:
                    hmax = max_values_1D_diff.iloc[2 + self.nb_spec]
                    hmin = min_values_1D_diff.iloc[2 + self.nb_spec]

                
                # Random numbers
                c = np.random.uniform(-1, 1)
                d = np.random.uniform(-1, 1)

                # Parameters
                a = 8
                b = 5
                
                # New state
                hnew = h + (c/a) * (hmax - hmin)
                Yk = np.clip(Yk, 0, 1)
                Yknew = Yk**(1.+d/b)
                Yknew = Yknew / Yknew.sum()

                # New temperature
                try:
                    gas.HPY = hnew, p, Yknew
                    Tnew = gas.T
                except ct.CanteraError:
                    print(f">> T computation crashed for row {i}")
                    accepted = False


                # Skip under some conditions
                if Tnew<300.0:
                    accepted = False

                # Elements (C, H, O, N)
                X_el = utils.compute_X_element(gas.species_names, Yknew)
                O_N_ratio = X_el[2]/X_el[3]
                # if O_N_ratio<0.25 or O_N_ratio > 0.28:
                #     accepted = False


                if accepted:
                    row_new = np.empty(5+self.nb_spec)
                    row_new[0] = Tnew
                    row_new[1] = p
                    row_new[2:2 + self.nb_spec] = Yknew
                    row_new[2 + self.nb_spec] = hnew
                    row_new[3 + self.nb_spec] = row.iloc[3 + self.nb_spec]
                    row_new[4 + self.nb_spec] = row.iloc[4 + self.nb_spec]

                    self.df_augmented.loc[i] = row_new


        self.df = pd.concat([self.df, self.df_augmented], axis=0).reset_index(drop=True) 
        self.is_augmented = True


    def save_database(self):

        self.df.to_csv(os.path.join(self.folder,"database_simus.csv"))



    def generate_train_valid_test(self, valid_ratio, test_ratio):

        # In this function we use the flames runs to build the X_train, Y_train, X_val, Y_val, X_test, Y_test data
        # Train, validation and test are based on simulations numbers, not states
        # 0D reactors
        self.id_sim_train_0D, self.id_sim_val_0D = train_test_split(self.sim_numbers_0D, test_size=valid_ratio, random_state=24)
        self.id_sim_train_0D, self.id_sim_test_0D = train_test_split(self.id_sim_train_0D, test_size=test_ratio, random_state=12)
        #
        self.df_ODE_train_0D = self.df_ODE_0D[self.df_ODE_0D["sim_number"].isin(self.id_sim_train_0D)]
        self.df_ODE_val_0D = self.df_ODE_0D[self.df_ODE_0D["sim_number"].isin(self.id_sim_val_0D)]
        self.df_ODE_test_0D = self.df_ODE_0D[self.df_ODE_0D["sim_number"].isin(self.id_sim_test_0D)]
        #
        self.df_ODE_train_0D.to_csv(os.path.join(self.folder,"sim_train_0D.csv"), index=False)
        self.df_ODE_val_0D.to_csv(os.path.join(self.folder,"sim_val_0D.csv"), index=False)
        self.df_ODE_test_0D.to_csv(os.path.join(self.folder,"sim_test_0D.csv"), index=False)

        fig, ax = plt.subplots()
        ax.scatter(self.df_ODE_train_0D['T0'], self.df_ODE_train_0D['Phi'], color="k", label="Train")
        ax.scatter(self.df_ODE_val_0D['T0'], self.df_ODE_val_0D['Phi'], color="b", label="Validation")
        ax.scatter(self.df_ODE_test_0D['T0'], self.df_ODE_test_0D['Phi'], color="r", label="Test")
        fig.legend(ncol=3)
        ax.set_xlabel("T0 [K]")
        ax.set_ylabel("Phi [-]")
        fig.savefig(os.path.join(self.folder,"doe_0D_reactors.png"))



        # 1D premixed
        self.id_sim_train_1D_prem, self.id_sim_val_1D_prem = train_test_split(self.sim_numbers_1D_prem, test_size=valid_ratio, random_state=24)
        self.id_sim_train_1D_prem, self.id_sim_test_1D_prem = train_test_split(self.id_sim_train_1D_prem, test_size=test_ratio, random_state=12)
        #
        self.df_ODE_train_1D_prem = self.df_ODE_1D_prem[self.df_ODE_1D_prem["sim_number"].isin(self.id_sim_train_1D_prem)]
        self.df_ODE_val_1D_prem = self.df_ODE_1D_prem[self.df_ODE_1D_prem["sim_number"].isin(self.id_sim_val_1D_prem)]
        self.df_ODE_test_1D_prem = self.df_ODE_1D_prem[self.df_ODE_1D_prem["sim_number"].isin(self.id_sim_test_1D_prem)]
        #
        self.df_ODE_train_1D_prem.to_csv(os.path.join(self.folder,"sim_train_1D_prem.csv"), index=False)
        self.df_ODE_val_1D_prem.to_csv(os.path.join(self.folder,"sim_val_1D_prem.csv"), index=False)
        self.df_ODE_test_1D_prem.to_csv(os.path.join(self.folder,"sim_test_1D_prem.csv"), index=False)

        fig, ax = plt.subplots()
        ax.scatter(self.df_ODE_train_1D_prem['T0'], self.df_ODE_train_1D_prem['Phi'], color="k", label="Train")
        ax.scatter(self.df_ODE_val_1D_prem['T0'], self.df_ODE_val_1D_prem['Phi'], color="b", label="Validation")
        ax.scatter(self.df_ODE_test_1D_prem['T0'], self.df_ODE_test_1D_prem['Phi'], color="r", label="Test")
        fig.legend(ncol=3)
        ax.set_xlabel("T0 [K]")
        ax.set_ylabel("Phi [-]")
        fig.savefig(os.path.join(self.folder,"doe_1D_prem_flames.png"))


        #1D diffusion
        self.id_sim_train_1D_diff, self.id_sim_val_1D_diff = train_test_split(self.sim_numbers_1D_diff, test_size=valid_ratio, random_state=24)
        self.id_sim_train_1D_diff, self.id_sim_test_1D_diff = train_test_split(self.id_sim_train_1D_diff, test_size=test_ratio, random_state=12)
        #
        self.df_ODE_train_1D_diff = self.df_ODE_1D_diff[self.df_ODE_1D_diff["sim_number"].isin(self.id_sim_train_1D_diff)]
        self.df_ODE_val_1D_diff = self.df_ODE_1D_diff[self.df_ODE_1D_diff["sim_number"].isin(self.id_sim_val_1D_diff)]
        self.df_ODE_test_1D_diff = self.df_ODE_1D_diff[self.df_ODE_1D_diff["sim_number"].isin(self.id_sim_test_1D_diff)]
        #
        self.df_ODE_train_1D_diff.to_csv(os.path.join(self.folder,"sim_train_1D_diff.csv"), index=False)
        self.df_ODE_val_1D_diff.to_csv(os.path.join(self.folder,"sim_val_1D_diff.csv"), index=False)
        self.df_ODE_test_1D_diff.to_csv(os.path.join(self.folder,"sim_test_1D_diff.csv"), index=False)

        fig, ax = plt.subplots()
        ax.scatter(self.df_ODE_train_1D_diff['T0'], self.df_ODE_train_1D_diff['Strain'], color="k", label="Train")
        ax.scatter(self.df_ODE_val_1D_diff['T0'], self.df_ODE_val_1D_diff['Strain'], color="b", label="Validation")
        ax.scatter(self.df_ODE_test_1D_diff['T0'], self.df_ODE_test_1D_diff['Strain'], color="r", label="Test")
        fig.legend(ncol=3)
        ax.set_xlabel("T0 [K]")
        ax.set_ylabel("$K_s$ [-]")
        fig.savefig(os.path.join(self.folder,"doe_1D_diff_flames.png"))



        self.X_train, self.Y_train = self._get_X_Y(self.id_sim_train_0D, self.id_sim_train_1D_prem, self.id_sim_train_1D_prem)
        self.X_val, self.Y_val = self._get_X_Y(self.id_sim_val_0D, self.id_sim_val_1D_prem, self.id_sim_val_1D_diff)
        # self.X_test, self.Y_test = self._get_X_Y(self.id_sim_test)   -> not needed as we will test on trajectories, not points

        # Shuffle data
        permutation_train = np.random.permutation(self.X_train.shape[0])
        self.X_train = self.X_train.iloc[permutation_train].reset_index(drop=True)
        self.Y_train = self.Y_train.iloc[permutation_train].reset_index(drop=True)
        #
        permutation_val = np.random.permutation(self.X_val.shape[0])
        self.X_val = self.X_val.iloc[permutation_val].reset_index(drop=True)
        self.Y_val = self.Y_val.iloc[permutation_val].reset_index(drop=True)

        # Saving data (raw data)
        self.X_train.to_csv(os.path.join(self.folder,"X_train_raw.csv"), index=False)
        self.Y_train.to_csv(os.path.join(self.folder,"Y_train_raw.csv"), index=False)
        self.X_val.to_csv(os.path.join(self.folder,"X_val_raw.csv"), index=False)
        self.Y_val.to_csv(os.path.join(self.folder,"Y_val_raw.csv"), index=False)




    def _get_X_Y(self, simu_ids_0D, simu_ids_1D_prem, simu_ids_1D_diff):

        X_list = []
        Y_list = []

        # Cantera gas object
        self.gas = ct.Solution(self.mech_file)
        
        # 0D reactors
        for i in simu_ids_0D:

            print(f">> 0D reactors: Building X, Y for simulation {i}")

            df_i = self.df[self.df['reactor_type'] == 0][self.df['Simulation number'] == i].iloc[:, :-2]

            cols = df_i.columns
            
            X = df_i.copy()

            X.columns = [str(col) + '_X' for col in cols]
            self.X_cols = X.columns.tolist()

            Y_np = self._advance_dt_cfd(X)
            Y_columns = [str(col) + '_Y' for col in cols]
            Y = pd.DataFrame(data=Y_np, columns = Y_columns)
            self.Y_cols = Y.columns.tolist()


            X_list.append(X)
            Y_list.append(Y)



        # 1D premixed flames
        for i in simu_ids_1D_prem:

            print(f">> 1D premixed flames: Building X, Y for simulation {i}")

            df_i = self.df[self.df['reactor_type'] == 1][self.df['Simulation number'] == i].iloc[:, :-2]

            cols = df_i.columns

            X = df_i.copy()
            X.columns = [str(col) + '_X' for col in cols]
            self.X_cols = X.columns.tolist()

            Y_np = self._advance_dt_cfd(X)
            Y_columns = [str(col) + '_Y' for col in cols]
            Y = pd.DataFrame(data=Y_np, columns = Y_columns)
            self.Y_cols = Y.columns.tolist()


            X_list.append(X)
            Y_list.append(Y)
            


        # 1D diffusion flames
        for i in simu_ids_1D_diff:

            print(f">> 1D diffusion flames: Building X, Y for simulation {i}")

            df_i = self.df[self.df['reactor_type'] == 2][self.df['Simulation number'] == i].iloc[:, :-2]

            cols = df_i.columns

            X = df_i.copy()
            X.columns = [str(col) + '_X' for col in cols]
            self.X_cols = X.columns.tolist()

            Y_np = self._advance_dt_cfd(X)
            Y_columns = [str(col) + '_Y' for col in cols]
            Y = pd.DataFrame(data=Y_np, columns = Y_columns)
            self.Y_cols = Y.columns.tolist()


            X_list.append(X)
            Y_list.append(Y)


        X = pd.concat(X_list,axis=0).reset_index(drop=True)
        Y = pd.concat(Y_list,axis=0).reset_index(drop=True)

        # Remove non needed items
        list_to_remove = ['Pressure_X', 'enthalpy_X']
        [self.X_cols.remove(elt) for elt in list_to_remove]   
        X = X[self.X_cols]

        # Remove non needed lables for Y
        list_to_remove = ['Pressure_Y', 'enthalpy_Y', 'Temperature_Y']   # Temperature computed from enthalpy conservation
        # Removing unwanted items
        [self.Y_cols.remove(elt) for elt in list_to_remove]
        Y= Y[self.Y_cols]

        return X, Y
    


    def _advance_dt_cfd(self, X):

        Y = np.empty(X.shape)

        k=0
        for i, row in X.iterrows():
            
            T = row[0]
            p = row[1]
            Yk = row[2:2+self.nb_spec].values


            self.gas.TPY = T, p, Yk

            # Constant pressure reactor
            r = ct.IdealGasConstPressureReactor(self.gas)
            
            # Initializing reactor
            sim = ct.ReactorNet([r])
            
            # Advancing to dt
            sim.advance(self.dt_cfd)

            # Updated state
            Y[k,0] = self.gas.T
            Y[k,1] = self.gas.P
            Y[k,2:2+self.nb_spec] = self.gas.Y

            k+=1


        return Y