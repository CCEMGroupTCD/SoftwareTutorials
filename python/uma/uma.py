"""
This module provides functions to relax molecules using the UMA (Universal Molecular Architecture) model from FAIRChem.
Github: https://github.com/facebookresearch/fairchem  -> UMA installation and first steps
Other resources:
Demo: https://facebook-fairchem-uma-demo.hf.space/
Paper: https://arxiv.org/abs/2506.23971
Huggingface: https://huggingface.co/facebook/UMA
"""
import warnings
from datetime import datetime
from typing import Union
from copy import deepcopy
from collections import defaultdict
import tempfile
import numpy as np
import torch
from pathlib import Path
import ase
from ase.io import read
from ase.optimize import LBFGS
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from ase.build import molecule

synonyms = {'umas': 'uma-s-1p1', 'umam': 'uma-m-1p1'}

# Custom inference settings designed to speed up inference and minimise VRAM usage. The only difference to the default settings is tf32=True.
def inference_settings_speedy():
    return InferenceSettings(
        tf32=True,
        activation_checkpointing=True,
        merge_mole=False,
        compile=False,
        wigner_cuda=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )

cached_mlips = defaultdict(dict)
def load_mlip(method: str, device: str):
    print(f'Loading MLIP `{method}` on device `{device}`...')
    try:
        predict_unit = pretrained_mlip.get_predict_unit(method, device=device, inference_settings=inference_settings_speedy())
    except KeyError:
        try:
            predict_unit = pretrained_mlip.get_predict_unit(synonyms[method], device=device, inference_settings=inference_settings_speedy())
        except KeyError:
            raise KeyError(f'Predictor `{method}` not found.')
    cached_mlips[method][device] = predict_unit

    return predict_unit

def get_mlip(method, device):
    try:
        return cached_mlips[method][device]
    except KeyError:
        load_mlip(method, device)
        return cached_mlips[method][device]

def uma_get_hessian(calc: FAIRChemCalculator, atoms, vmap: bool=False):
    """
    Get the Hessian matrix for the given atomic structure.
    Args:
        atoms (Atoms): The atomic structure to calculate the Hessian for.
        vmap (bool): Whether to use vectorized mapping for Hessian calculation. This can speed up the calculation but for medium-sized systems (TMCs) it uses way too much RAM. Calculating the Hessian without vmap for a TMC with the small UMA model took around 6 minutes on Mac (with possible background parallelization).
    Returns:
        np.ndarray: The Hessian matrix.
    """
    from fairchem.core.datasets import data_list_collater
    import torch
    from torch.autograd import grad
    # Turn on create_graph for the first derivative
    calc.predictor.model.module.output_heads['energyandforcehead'].head.training = True

    # Convert using the current a2g object
    data_list = [calc.a2g(atoms)]# for atoms in atoms_list]

    # Batch and predict
    batch = data_list_collater(data_list, otf_graph=True)
    pred = calc.predictor.predict(batch)

    # Get the forces and positions
    positions = batch.pos
    forces = pred["forces"].flatten()

    # Calculate the Hessian using autograd
    if vmap:
        hessian = torch.vmap(
            lambda vec: grad(
                -forces,
                positions,
                grad_outputs=vec,
                retain_graph=True,
                )[0],
            )(torch.eye(forces.numel(), device=forces.device)).detach().cpu().numpy()
    else:
        hessian = np.zeros((len(forces), len(forces)))
        for i in range(len(forces)):
            hessian[:, i] = grad(
                -forces[i],
                positions,
                retain_graph=True,
            )[0].flatten().detach().cpu().numpy()

    # Turn off create_graph for the first derivative
    calc.predictor.model.module.output_heads['energyandforcehead'].head.training = False

    return hessian

def _ensure_writable_arrays_inplace(atoms):
    """
    Ensure that all arrays in the atoms object are writable. Modifies the atoms object in place.
    """
    for _key, _arr in list(atoms.arrays.items()):
        try:
            if hasattr(_arr, "flags") and not _arr.flags.writeable:
                atoms.arrays[_key] = np.array(_arr, copy=True)
        except Exception:
            atoms.arrays[_key] = np.array(_arr, copy=True)


def _run_uma(atoms: ase.Atoms, charge: int, n_unpaired: int, device: str, fmax: float, logfile: Union[str, None], method: str, steps: int, task_name: str, tempdir: str, frequencies: bool) -> dict:
    """
    Run UMA relaxation. The output is a dictionary with all relevant results. It is intention that this function does not return a uma calc object, since it is not pickable and might not be returnable on hpc parallel runs with ray or joblib. For local, non-parallel runs, one could totally modify this function to return the calc object as well.
    :return: dict
    """
    start_time = datetime.now()
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA device requested but not available. Falling back to CPU.')
        device = 'cpu'

    _ensure_writable_arrays_inplace(atoms)
    predict_unit = get_mlip(method, device)
    calc = FAIRChemCalculator(predict_unit=predict_unit, task_name=task_name)
    atoms.calc = calc
    atoms.info = {'charge': charge, 'spin': n_unpaired+1}   # UMA wants the multiplicity, not the number of unpaired electrons
    traj_path = Path(tempdir, 'uma_relaxation.xyz')
    opt = LBFGS(atoms, trajectory=str(traj_path), logfile=logfile)  # Traj must be str, not Path()
    opt.run(fmax=fmax, steps=steps)

    if frequencies:
        raise NotImplementedError('The frequency calculation is not yet implemented. You can implement it here easily using the uma_get_hessian function in the code above and then calculating frequencies from the Hessian matrix and the Gibbs energies from the frequencies using the ase thermochemistry module.')

    # Avoid serialization issues with Ray/joblib by removing the calc from the atoms object
    relaxed_atoms = opt.atoms.copy()
    relaxed_atoms.calc = None

    # Return all relevant results. Can't return the calc and opt objects directly since they are not pickable by Ray/joblib.
    concat_atoms = read(traj_path, ':')
    energies = [atoms.calc.results['energy'] for atoms in concat_atoms]
    try:
        dE = energies[-2] - calc.results['energy']
    except IndexError:
        dE = np.nan
    final_forces = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    results = {
        'E': calc.results['energy'],                    # final energy
        'atoms': relaxed_atoms,                         # final relaxed atoms
        'forces': calc.results['forces'].tolist(),      # final forces. Can be outcommented for less storage.
        'stress': calc.results['stress'].tolist(),      # final stress
        'input': {                                      # input parameters
            'charge': charge,
            'n_unpaired': n_unpaired,
            'method': method,
            'device': device,
            'task_name': task_name,
            'fmax': opt.fmax,
            'n_max_steps': opt.max_steps,
        },
        'opt':                                          # optimization results
            {
                'n_steps': opt.nsteps,                  # number of optimization steps taken
                'f': final_forces,                      # final max force on any atom
                'dE': dE,                               # energy change in last step
                'converged': bool(final_forces < opt.fmax), # whether the optimization converged
                'H0': opt.H0,
                'energies': energies,                   # energies at each step
                'traj': concat_atoms,                   # trajectory as list of Atoms objects. Can be outcommented for less storage.
                'time': datetime.now() - start_time,    # time taken for the optimization
            }
    }
    return results

def uma_relax_atoms(
                    atoms: Union[ase.Atoms, Path, str],
                    charge: int,
                    n_unpaired: int,
                    method: str = 'umas',
                    fmax=0.05,
                    steps=300,
                    device='cpu',
                    task_name='omol',
                    frequencies: bool=False,
                    logfile: str = None,
                    timing: bool=False
                    ) -> dict:
    """
    Relax a molecule using the FAIRChemCalculator.
    @param atoms: Atoms object to be relaxed or path to an xyz file.
    @param charge: Charge of the molecule.
    @param n_unpaired: Number of unpaired electrons of the molecule.
    @param method: 'umas' or 'umam', which corresponds to the predictor to be used for the relaxation.
    @param fmax: Maximum convergence force for the relaxation. A value of 0.05 will lead to a convergence dE of around 1E-3 eV for a typical TMC.
    @param steps: Number of steps for the relaxation. If 0, a single-point calculation is performed.
    @param device: Device to be used for the calculation. 'cpu' or 'cuda' (if available). Default is 'cpu'.
    @param task_name: Task name for the calculation. For molecules use 'omol', for others look into the FAIRChem documentation.
    @param frequencies: Whether to calculate the frequencies and return a Gibbs energy or not. Not yet implemented but possible.
    @param logfile: Logfile to save the output of the optimization. None to suppress output, '-' to print to stdout.
    @param timing: Whether to time the optimization or not.
    @return: A dictionary with the results of the relaxation, including energy, forces, stress, relaxed atoms, and other relevant information.
    """
    try:    # Try to read the atoms object from a file
        atoms = read(atoms)
    except AttributeError:
        atoms = atoms.copy()  # If atoms is already an Atoms object, copy it to avoid modifying the original
    coords_before = deepcopy(atoms.get_positions()) # For comparison after the optimization

    _ensure_writable_arrays_inplace(atoms)
    with tempfile.TemporaryDirectory() as tempdir:
        # Set up the FAIRChemCalculator with the specified method and run the optimization
        results = _run_uma(atoms=atoms, charge=charge, n_unpaired=n_unpaired, device=device, fmax=fmax, logfile=logfile, method=method, steps=steps, task_name=task_name, tempdir=tempdir, frequencies=frequencies)
        relaxed_atoms = results['atoms']

    if timing:
        print(f'UMA opt took {results["opt"]["time"]} seconds for {results["opt"]["n_steps"]} steps.')

    if steps > 0 and np.allclose(coords_before, relaxed_atoms.get_positions()):
        warnings.warn(f'Atoms did not move during the optimization. This might be due to a too high fmax ({fmax}) or too few steps ({steps}). Consider increasing these values.')

    return results





if __name__ == '__main__':

    # Todo once only: login to huggingface to download and cache the used model
    # from huggingface_hub import login; login(token='...') # provide here your huggingface token after registering. Do this just once for every model. Never share your token publicly, e.g. never commit the code to git with the token in it.

    ##########   Example: Relax a water molecule with the small UMA model   ##########
    atoms = molecule('H2O') # ase.Atoms object or path to an xyz file. Here we create a water molecule using ASE.
    method = 'umas'     # 'umas' (small) or 'umam' (medium). The small model is faster but less accurate.
    charge = 0          # Total charge of the molecule.
    n_unpaired = 0      # Number of unpaired electrons of the molecule.
    fmax = 0.05         # Max. converge force in eV/A. fmax=0.05 will give a convergence dE of around 1E-3 eV for a typical TMC.
    steps = 300         # Maximum number of steps for the relaxation. 0 means single-point calculation.
    task_name = 'omol'  # Task name for the calculation. For molecules use 'omol', for others look into the FAIRChem documentation.
    # Other options:
    logfile = None      # Logfile to save the optimization output. None to suppress output, '-' to print to stdout.
    timing = True       # Whether to print timing information or not.
    # NOT YET IMPLEMENTED:
    frequencies = False # Whether to calculate frequencies and Gibbs energy or not.
    device = 'cpu'      # Currently only 'cpu' is supported. For using uma on a gpu with 'cuda', best ask Timo.

    results = uma_relax_atoms(
                                atoms=atoms,
                                charge=charge,
                                method=method,
                                n_unpaired=n_unpaired,
                                fmax=fmax,
                                steps=steps,
                                frequencies=frequencies,
                                device=device,
                                logfile=logfile,
                                timing=timing,
                                task_name=task_name
                                )

    # Optional: uncomment to view optimization trajectory in ase gui.
    # from ase.visualize import view
    # view(results['opt']['traj'])    # List of Atoms objects representing the trajectory

    print('Done!')