from jax import jit

import numpy as np

from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.utils import load_image_to_numpy

from matplotlib import pyplot as plt

from jwave.utils import show_field

@jit
def solver(medium, p0):
    """
    Wrapper for solver
    """
    return simulate_wave_propagation(medium, time_axis, p0=p0)

# Simulation parameters
N, dx = (256, 256), (0.1e-3, 0.1e-3)
c0 = 1500.0
cfl = 0.3
t_end = 0.8e-05
domain = Domain(N, dx)
medium = Medium(domain=domain, sound_speed=1500.0)
time_axis = TimeAxis.from_medium(medium, cfl=cfl, t_end=t_end)

# Initial pressure field
p0 = load_image_to_numpy("../docs/assets/images/jwave.png", image_size=N) / float(N[0]-1)
p0 = FourierSeries(p0, domain)

# Compile and run the simulation
pressure = solver(medium, p0)

t = 250
show_field(pressure[t])
plt.title(f"Pressure field at t={time_axis.to_array()[t]:0.3e}[s]")
plt.show()