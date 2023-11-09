# Quick Introduction to TBLite - a Practical Guide

## Installation
Under a conda environment, install tblite and tblite-python through the conda-forge channel.

`conda install -c conda-forge tblite`
`conda install -c conda-forge tblite-python`

Make sure you also have ASE installed as well as the basic scientific programming packages.

## Python API and Usage in ASE
### Single-Point Calculations
The skeleton of the script is very simple: the calculator is initialised, and attached to your atom object. Then, most functionality of ASE is readily available.
```python
from tblite.ase import TBLite
from ase.build import fcc111, molecule, add_adsorbate

# Create or import your atoms object
slab = fcc111('Ni', size=(2,2,3))
add_adsorbate(slab, 'H', height=1.5, 'ontop')
slab.center(vacuum=10.0, axis=2)

# Set up calculator
calc = TBLite(method="GFN1-xTB")

# Attach to atoms object
slab.calc = calc
slab.get_potential_energy()
```

### Structure Relaxation
In order to perform a relaxation, we need an external optimizer that allows for convergence. 
ASE has several optimizers that can be used: https://wiki.fysik.dtu.dk/ase/ase/optimize.html 
However, the following are recommended:
- Fast Internal Relaxation Engine: FIRE 
- Non-linear (Polak-Ribiere) conjugate gradient algorithm: SciPyFminCG
- Broyden–Fletcher–Goldfarb–Shanno algorithm: BFGS

Let's carry on with the previous example:
```python
# Import optimizer from ASE
from ase.optimize.sciopt import SciPyFminCG 

# Set up optimizer and run
optimizer = SciPyFminCG(slab, trajectory='path/to/directory/system_relaxation.traj')
optimizer.run(fmax=0.05)
```

#### Trajectory Files
A trajectory file will be created in the specifier directory. This file contains all the ionic steps and can be visualised with ase gui. Each frame of this "movie" is an atoms object that can be imported as a vasp file. 

First, let's start by storing the optimized structure:
```python
from ase.io import write, read

# Write the relaxed CO2 structure to a VASP file
write('path/to/directory/system_relaxed.vasp', slab, format='vasp')
```
We can also read/write specific frames:
```python
# Extract specific frames
frames = read('path/to/directory/system_relaxation.traj', index='0,3,5')

# Write extracted frames as individual VASP files
for i, atoms in enumerate(frames):
    write(f'path/to/directory/frame_{i}.vasp', atoms, format='vasp')
```
In the command line:
- Specify images to load: ase gui filename.traj -i 0,3,5
- A continuous range of frames: ase gui filename.traj -i 1:5

### Calculation of Vibrational Modes - Entropic Correction
After the `optimizer.run(fmax=0.05)` line has completed, the ASE atoms object will be in its optimized state. Calling `get_potential_energy()` on it will return the total energy of this optimized state.
```python
latest_total_energy = slab.get_potential_energy()
print("Latest total energy:", latest_total_energy, "eV")
```
TBLite provides the electronic free energy (that is why we set an electronic temperature). We can compute the nuclear contribution associated with the vibration degrees of freedom with TBlite. For a better explanation of such entropic contribution, check out the Notion page. 
```python
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo

vib = Vibrations(slab, name='path/to/directory/CO2_vibrations')  # Specify the name or path where you want the data saved

# Calculate vibrations
vib.run()

# Once the calculations are done, you can get the vibrational frequencies/energies, etc
frequencies = vib.get_frequencies()
energies = vib.get_energies()

# Calculate the Helmholtz energy for each hydrogen and store it in the list
thermo_properties = HarmonicThermo(vib_energies=energies)
F = thermo_properties.get_helmholtz_energy(temperature=298.0)
print("Helmholtz Energy:", F, "eV")

# Total final energy:
print("Total final energy:", latest_total_energy - F, "eV")
```
For more information, check https://wiki.fysik.dtu.dk/ase/ase/thermochemistry/thermochemistry.html#id3

## Parallelizing the script
So far, we have a script that performs various calculations on a H/Ni(111) system, namely geometry optimization followed by vibrational frequency calculations, and then computes thermodynamic properties based on those frequencies.

To parallelize this script we are going to use `joblib`. Let’s consider a case where we want to run calculations on different molecules in parallel. Let's assume we have a list of molecules for which you want to calculate the "Total final energy."
#### Dummy Example:
```python
from joblib import Parallel, delayed

def compute(x):
    return result

slow_results = [compute(x) for x in data] # Run in series
faster_results = Parallel(n_jobs=4)(delayed(compute(x) for x in data) # Run in parallel with joblib
```
#### Practical Example: CO2, H2O, NH3 - Full Relaxation
```python
from tblite.ase import TBLite
from ase.build import molecule
from ase.optimize import FIRE
from joblib import Parallel, delayed

def calculate_final_energy(molecule_name):
    atoms = molecule(molecule_name)

    # Set up calculator
    calc = TBLite(
        method="GFN1-xTB",
        accuracy=1.0,
        max_iterations=250,
        electronic_temperature=300,
    )

    # Attach to atoms object
    atoms.calc = calc
    atoms.get_potential_energy()
    
    # Set up optimizer and run
    optimizer = FIRE(atoms, trajectory=f'path/to/directory/{molecule_name}_relaxation.traj')
    optimizer.run(fmax=0.2)
    latest_total_energy = atoms.get_potential_energy()
    
    return molecule_name, latest_total_energy

# Let's assume a list of molecules
molecules = ['CO2', 'H2O', 'NH3'] 

results = Parallel(n_jobs=-1)(delayed(calculate_final_energy)(molecule_name) for molecule_name in molecules)

for molecule_name, energy in results:
    print(f"Total final energy for {molecule_name}:", energy, "eV")
```
1. We've encapsulated the calculations inside a function `calculate_final_energy` which takes a `molecule_name` as an argument.
2. We use `Parallel` and `delayed` from `joblib` to parallelize the calculations across the list of molecules.
3. This script will run the calculations in parallel using all available CPU cores (`n_jobs=-1`). This is not recommended if you try it for the first time. 
