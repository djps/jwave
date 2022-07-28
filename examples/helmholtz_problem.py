from functools import partial

import numpy as np
from jax import jit
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt

import time

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, _circ_mask
from jwave.utils import plot_complex_field

from jwave.acoustics.time_harmonic import helmholtz, helmholtz_solver


@jit
def solve_helmholtz(medium, params):
    return helmholtz_solver(medium, omega, src, params=params)

if __name__ == '__main__':

    key = random.PRNGKey(42)  # Random seed

    # Defining geometry
    nx = int(128)
    ny = int(256)
    # Grid size
    N = (nx, ny)
    # Spatial resolution
    dx = (0.001, 0.001)
    # Wavefield omega = 2*pi*f
    omega = 1.5e6
    # Target location
    target = [160, 360]

    # Defining the domain
    domain = Domain(N, dx)

    # Build the vector that holds the parameters of the apodization an the
    # functions required to transform it into a source wavefield
    src_field = jnp.zeros(N).astype(jnp.complex64)
    src_field = src_field.at[64, 22].set(1.0)
    src = FourierSeries(jnp.expand_dims(src_field, -1), domain) * omega

    # Plotting binary field
    fig0, ax0 = plot_complex_field(src)
    fig0.show()

    # --------------------------------------------------------------------------

    # Constructing medium physical properties
    sound_speed = jnp.zeros(N)
    sound_speed = sound_speed.at[20:105, 20:200].set(1.0)
    sound_speed = (
        sound_speed
        * (1 - _circ_mask(N, 90, [64, 180]))
        * (1 - _circ_mask(N, 50, [64, 22]))
        * 0.5
        + 1
    )
    sound_speed = FourierSeries(jnp.expand_dims(sound_speed, -1), domain) * 1540

    density = 1.0  # sound_speed*0 + 1.
    attenuation = 0.0  # density*0
    sound_speed = 1500.0

    medium1 = Medium(domain=domain, sound_speed=sound_speed, density=1000.0, pml_size=15)

    params = helmholtz.default_params(src, medium1, omega)

    print("\nRuntime with GMRES (homogeneous material)")
    t0 = time.time()
    field1 = solve_helmholtz(medium1, params).params.block_until_ready()
    t1 = time.time()
    time_elapsed = t1-t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

    fig1, ax1 = plot_complex_field(field1, max_intensity=2e5)
    fig1.show()

    # Hetrogeneous density
    density = jnp.ones(N)
    density = jnp.expand_dims(density.at[:64, 170:].set(1.5), -1)
    density = FourierSeries(density, domain)

    medium2 = Medium(domain=domain, sound_speed=sound_speed, density=density, pml_size=15)

    params = helmholtz.default_params(src, medium2, omega) # Parameters may be different due to different density type

    print( type(params['fft_u']['k_vec'][0]) )

    print( params.keys() )

    # Solve new problem
    print("Runtime with GMRES (heterogeneous density)")
    t0 = time.time()
    field2 = solve_helmholtz(medium2, params).params.block_until_ready()
    t1 = time.time()
    time_elapsed = t1-t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    fig2, ax2 = plot_complex_field(field2, max_intensity=2e5, suptitle="Heterogeneous density")
    fig2.show()


    # Hetrogeneous attenuation
    attenuation = jnp.zeros(N)
    attenuation = jnp.expand_dims(attenuation.at[64:110, 125:220].set(100), -1)

    medium3 = Medium(
        domain=domain,
        sound_speed=sound_speed,
        density=density,
        attenuation=attenuation,
        pml_size=15,
    )

    # Solve new problem
    print("Runtime with GMRES (heterogeneous density and attenuation)")
    t0 = time.time()
    field3 = solve_helmholtz(medium3, params).params.block_until_ready()
    t1 = time.time()
    time_elapsed = t1-t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))
    fig3, ax3 = plot_complex_field(field3, max_intensity=2e5, suptitle="Heterogeneous density and attenuation")
    fig3.show()

    plt.show()