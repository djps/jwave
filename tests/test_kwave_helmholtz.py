import os
from functools import partial
from typing import Tuple

import numpy as np
import pytest
from jax import device_put, devices, jit
from jax import numpy as jnp
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat

from jwave import FourierSeries
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.geometry import Domain, Medium
from jwave.utils import plot_comparison

from jwave.experimental.new_solver import bicgstabl

from .utils import log_accuracy

# Default figure settings
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300


# Setting sound speed
def _get_heterog_sound_speed(domain):
  sound_speed = np.ones(domain.N) * 1500.0
  sound_speed[50:90, 32:100] = 2300.0
  sound_speed = FourierSeries(np.expand_dims(sound_speed, -1), domain)
  return sound_speed

def _get_homog_sound_speed(domain):
  return 1500.0

# Setting density
def _get_heterog_density(domain):
  density = np.ones(domain.N) * 1000.0
  density[20:40, 65:100] = 2000.0
  density = FourierSeries(np.expand_dims(density, -1), domain)
  return density

def _get_density_interface(domain):
  density = np.ones(domain.N) * 1000.0
  density[64:] = 2000.0
  density = FourierSeries(np.expand_dims(density, -1), domain)
  return density

def _get_homog_density(domain):
  return 1000.0

# Setting attenuation
def _homog_attenuation_constructor(value = 0.1):
  def _att_setter(domain):
    return value
  return _att_setter

def heterog_attenuation_constructor(value = 0.1):
  def _att_setter(domain):
    att = np.zeros(domain.N)
    att[30:90, 64:100] = value
    att = FourierSeries(np.expand_dims(att, -1), domain)
    return att
  return _att_setter

# Setting solver
def _solver_setter(solver_type: str = 'gmres'):
  return solver_type


def _test_setter(
  N: Tuple[int] = (128,128),
  dx = 1e-3,
  PMLSize: int = 16,
  omega: float = 1.5e6,
  magnitude: float = 1.0,
  src_location: list = (32,32),
  c0_constructor = _get_homog_sound_speed,
  rho0_constructor = _get_homog_density,
  alpha_constructor = _homog_attenuation_constructor(0.0),
  rel_err = 1e-2,
  solver = _solver_setter('gmres')
):
  dx = tuple([dx]*len(N))
  assert len(N) == len(src_location), "src_location must have same length as N"
  return {
    "N" : N,
    "dx" : dx,
    "PMLSize" : PMLSize,
    "omega": omega,
    "magnitude": magnitude,
    "src_location": src_location,
    "alpha_constructor" : alpha_constructor,
    "c0_constructor" : c0_constructor,
    "rho0_constructor" : rho0_constructor,
    "rel_err" : rel_err,
    "solver" : solver,
  }

TEST_SETTINGS = {
  "helmholtz_homog": _test_setter(),
  "helmholtz_heterog_c0": _test_setter(
    c0_constructor = _get_heterog_sound_speed
  ),
  "helmholtz_inteface_rho0": _test_setter(
    src_location = (32,64),
    rho0_constructor = _get_density_interface,
    omega=1e6,
    rel_err=0.03
  ),
  "helmholtz_heterog_rho0": _test_setter(
    rho0_constructor = _get_heterog_density,
    omega=1e6,
    rel_err=0.04
  ),
  "helmholtz_heterog_alpha": _test_setter(
    src_location = (25,64),
    alpha_constructor = heterog_attenuation_constructor(100.0),
    omega=1e6,
    rel_err=0.01
  ),
  "helmholtz_homog_bicgstab": _test_setter(solver="bicgstab"),
  "helmholtz_homog_bicgstabl": _test_setter(solver="bicgstabl")
}


@pytest.mark.parametrize("test_name", TEST_SETTINGS.keys())
def test_helmholtz(
  test_name,
  use_plots = False,
  reset_mat_file = False
):
  settings = TEST_SETTINGS[test_name]
  matfile = test_name + ".mat"
  dir_path = os.path.dirname(os.path.realpath(__file__))

  # Extract simulation setup
  domain = Domain(settings["N"], settings["dx"])
  omega = settings["omega"]
  magnitude = settings["magnitude"]
  src_location = settings["src_location"]
  attenuation = settings["alpha_constructor"](domain)
  sound_speed = settings["c0_constructor"](domain)
  density = settings["rho0_constructor"](domain)
  
  # solver
  solver = settings["solver"]

  # Move everything to the CPU
  cpu = devices("cpu")[0]
  sound_speed = device_put(sound_speed, device=cpu)
  density = device_put(density, device=cpu)
  attenuation = device_put(attenuation, device=cpu)

  # Initialize simulation parameters
  medium = Medium(
    domain = domain,
    sound_speed = sound_speed,
    density = density,
    attenuation = attenuation,
    pml_size=settings["PMLSize"]
  )

  # Construct source field
  src_field = jnp.zeros(domain.N, dtype=jnp.complex64)
  src_field = src_field.at[src_location].set(magnitude)
  src_field = FourierSeries(jnp.expand_dims(src_field, -1), domain)

  # Run simulation
  @partial(jit, backend='cpu')
  def run_simulation(src_field):
    return helmholtz_solver(medium, omega, src_field, method=solver, tol=1e-5)

  # Extract last field
  solution_field = run_simulation(src_field).on_grid[:,:,0]

  # Generate the matlab results if they don't exist
  if not os.path.isfile(dir_path + '/kwave_data/' + matfile) or reset_mat_file:
    print("Generating matlab results")

    if isinstance(sound_speed, FourierSeries):
      sound_speed = sound_speed.on_grid

    if isinstance(density, FourierSeries):
      density = density.on_grid

    if isinstance(attenuation, FourierSeries):
      attenuation = attenuation.on_grid

    mdict = {
      "Nx": domain.N,
      "dx": domain.dx,
      "omega": omega,
      "sound_speed": sound_speed,
      "density": density,
      "attenuation": attenuation,
      "source_magnitude": magnitude,
      "source_location": src_location,
      "pml_size": settings["PMLSize"],
      "solver": solver,
    }
    in_filepath = dir_path + '/kwave_data/setup_' + matfile
    savemat(in_filepath, mdict)

    mat_command = f"cd('{dir_path}'); test_kwave_helmholtz(string('{in_filepath}')); exit;"
    command = f'''matlab -nodisplay -nosplash -nodesktop -r "{mat_command}"'''
    os.system(command)

  # Load the matlab results
  out_filepath = dir_path + '/kwave_data/' + matfile
  kwave = loadmat(out_filepath)
  kwave_solution_field = kwave["p_final"]
  err = abs(jnp.abs(solution_field) - jnp.abs(kwave_solution_field))

  # Remove pml
  err = err[settings["PMLSize"]:-settings["PMLSize"], settings["PMLSize"]:-settings["PMLSize"]]

  if use_plots:
    plot_comparison(
      jnp.abs(solution_field)[settings["PMLSize"]:-settings["PMLSize"], settings["PMLSize"]:-settings["PMLSize"]],
      jnp.abs(kwave_solution_field)[settings["PMLSize"]:-settings["PMLSize"], settings["PMLSize"]:-settings["PMLSize"]],
      test_name,
      ['j-Wave (abs)', 'k-Wave (abs)'],
      cmap="inferno",
      vmin=0,
      vmax=0.2
    )
    plt.show()

  # Check maximum error
  relErr = jnp.amax(err)/jnp.amax(jnp.abs(kwave_solution_field))
  print('Test name: ' + test_name)
  print('  Relative max error = ', 100*relErr, '%')
  assert relErr < settings["rel_err"], "Test failed, error above maximum limit of " + str(100*settings["rel_err"]) + "%"

  log_accuracy(test_name, relErr)

if __name__ == "__main__":
  for key in TEST_SETTINGS:
    test_helmholtz(key, use_plots = False)
