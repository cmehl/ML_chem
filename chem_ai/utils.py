import sys
import numpy as np
import cantera as ct

#----------------------------------------------
# CHEMISTRY TOOLS
#----------------------------------------------

# Function to get matrix (Wj/Wk)*n_k^j   (Remark: order of atoms is C, H, O, N)
def get_molar_mass_atomic_matrix(species, fuel, with_N2_chemistry):

    nb_species = len(species)

    atomic_array = parse_species_names(species)
    #
    mol_weights = get_molecular_weights(species)
    #
    mass_per_atom = np.array([12.011, 1.008, 15.999, 14.007])
    #
    A_element = np.copy(atomic_array)
    for j in range(4):
        A_element[j,:] *=  mass_per_atom[j]
    for k in range(nb_species):
        A_element[:,k] /=  mol_weights[k]

    # # Carbon not considered if fuel -> To make more general (get which atoms are in the list of species)
    # if fuel=="H2":
    #     A_element = A_element[1:,:]

    if with_N2_chemistry is False:
        A_element = A_element[:-1,:]

    return A_element



def compute_X_element(species, Yk):

    # Xk
    Wk = get_molecular_weights(species)
    W = 1.0 / np.sum(Yk/Wk)
    Xk = (W/Wk)*Yk


    atomic_array = parse_species_names(species)
    X_el = np.dot(atomic_array, Xk)

    return X_el



# Function to parse species names; for example CH4 -=> n_C=1 and n_H = 4
def parse_species_names(species_list):
    
    nb_species = len(species_list)
    atomic_array = np.empty((4,nb_species))   # 4 comes from the fact that we are dealing with 4 elements: C, H, O and N
    i_spec = 0
    for i_spec in range(nb_species):
        species = species_list[i_spec]

        # Checking if species is a fictive species, in which case we remove the "_F" suffix
        if species.endswith("_F"):
            species = species[:-2]

        i_char = 0
        n_C = 0
        n_H = 0
        n_O = 0
        n_N = 0
        for i_char in range(len(species)):
            
            if species[i_char] in ["C","c"]:
                try:
                    n_C += float(species[i_char+1])
                except ValueError:
                    n_C += 1
                except IndexError:
                    n_C += 1
                    
            elif species[i_char] in ["H","h"]:
                try:
                    n_H += float(species[i_char+1])
                except ValueError:
                    n_H += 1
                except IndexError:
                    n_H += 1
                        
            elif species[i_char] in ["O","o"]:
                try:
                    n_O += float(species[i_char+1])
                except ValueError:
                    n_O += 1
                except IndexError:
                    n_O += 1
                
            elif species[i_char] in ["N","n"]:
                try:
                    n_N += float(species[i_char+1])
                except ValueError:
                    n_N += 1
                except IndexError:
                    n_N += 1
      
        atomic_array[:,i_spec] = np.array([n_C,n_H,n_O,n_N])
        i_spec+=1
        
    return atomic_array



# Species molecular weights
def get_molecular_weights(spec_names):
    
    # Atomic masses (order same as atomic_array: C, H, O, N )
    mass_per_atom = np.array([12.011, 1.008, 15.999, 14.007])
    mass_per_atom = mass_per_atom.reshape((4,1))
    
    # Compute atomic array using the above function
    atomic_array = parse_species_names(spec_names)

    # Multiply array composition by masses
    mass_array = np.multiply(mass_per_atom, atomic_array)
    
    # Molecular weights obtained by squashing the array
    mol_weights = np.sum(mass_array, axis=0)
    
    return mol_weights


#----------------------------------------------
# PYTORCH TOOLS
#----------------------------------------------


class StandardScaler():
  
  def fit(self, x):
    self.mean = x.mean(0)
    self.std = x.std(0)

  def transform(self, x):
    x -= self.mean
    x /= (self.std + 1e-7)
    return x
  
  def inverse_transform(self, x):
    x = self.mean + (self.std + 1e-7)*x
    return x


