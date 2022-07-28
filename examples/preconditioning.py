import os, time

from functools import partial

import numpy as np

from jax import numpy as jnp
from jax import jit

from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, _circ_mask
from jwave.acoustics.time_harmonic import helmholtz, helmholtz_solver

from jwave.experimental.bicgstabl import *

# Default figure settings
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300

@partial(jit, static_argnames=['method'], backend='gpu')
def fast_solver(medium, params, method):
    return helmholtz_solver(medium, omega, src_field, guess=None, method=method, checkpoint=False, params=params)

@partial(jit, backend='gpu')
def wrapper_gmres(medium, params):
    return helmholtz_solver(medium, omega, src_field, guess=None, method='gmres', checkpoint=False, params=params)

@partial(jit, backend='gpu')
def wrapper_bicgstab(medium, params):
    return helmholtz_solver(medium, omega, src_field, guess=None, method='bicgstab', checkpoint=False, params=params)

@partial(jit, backend='gpu')
def wrapper_bicgstabl(medium, params):
    return helmholtz_solver(medium, omega, src_field, guess=None, method='bicgstabl', checkpoint=False, params=params)

@jit
def solve_helmholtz(medium, params):
    return helmholtz_solver(medium, omega, src_field, method='bicgstab', params=params)


if __name__ == '__main__':

    N = (128, 256)
    dx = (1.0, 1.0)
    omega = 1.0

    # Making geometry
    domain = Domain(N, dx)

    # Physical properties
    sound_speed = jnp.ones(N)
    sound_speed = sound_speed.at[20:105, 20:200].set(1.0)
    sound_speed = sound_speed * (1.0 - _circ_mask(N, 90, [64,180])) * (1.0 - _circ_mask(N, 50, [64,22])) + 1.0
    sound_speed_f = FourierSeries(sound_speed, domain)

    density = jnp.ones(N)
    density = density.at[:64, 170:].set(1.5)
    density_f = FourierSeries(density, domain)

    attenuation = jnp.zeros(N)
    attenuation = attenuation.at[64:, 125:].set(0.03)
    attenuation_f = FourierSeries(attenuation, domain)

    medium = Medium(domain=domain,
        sound_speed=sound_speed_f,
        density=density_f,
        attenuation=attenuation_f,
        pml_size=15
    )

    # Source field
    src_field = jnp.zeros(N).astype(jnp.complex64)
    src_field = src_field.at[64, 22].set(1.0)
    src_field = FourierSeries(np.expand_dims(src_field, -1), domain)

    params = helmholtz.default_params(src_field, medium, omega)
    
    field = solve_helmholtz(medium, params)

    field = field.on_grid / jnp.amax(jnp.abs(field.on_grid))

    fig, axes = plt.subplots(3, 1)
    axes[0].imshow(sound_speed, cmap="gray")
    axes[0].imshow(attenuation, alpha=attenuation * 10.0)
    axes[0].imshow(density-1.0, alpha=(density-1.0) * 2.0, cmap="seismic")
    axes[0].set_title(f"Speed of sound (grayscale), attenuation (yellow), density (red)")
    axes[1].imshow(jnp.real(field), vmin=-0.2, vmax=0.2, cmap="seismic")
    axes[1].set_title(f"Real wavefield")
    axes[2].imshow(jnp.abs(field), vmin=0, vmax=0.2, cmap="magma")
    axes[2].set_title(f"Wavefield magnitude")
    plt.show()

    print("GMRES", end='', flush=True)
    t0 = time.time()
    wrapper_gmres(medium, params)
    #helmholtz_solver(medium, omega=1.0, source=src_field, guess=None, method='gmres', tol=1e-3).block_until_ready()
    t1 = time.time()
    time_elapsed = t1-t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

    print("BiCGStab 1", end='', flush=True)
    t0 = time.time()
    wrapper_bicgstab(medium, params)
    #wrapper_gmres(src_field, medium, method='bicgstab', omega=1.0, guess=None, tol=1e-3).block_until_ready()
    #helmholtz_solver(medium, omega=1.0, source=src_field, guess=None, method='bicgstab', tol=1e-3).block_until_ready()
    t1 = time.time()
    time_elapsed = t1-t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

    print("BiCGStab 2", end='', flush=True)
    t0 = time.time()
    fast_solver(medium, params, 'bicgstab')
    t1 = time.time()
    time_elapsed = t1-t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))

    hello_world()

    print("BiCGStabL", end='', flush=True)
    t0 = time.time()
    wrapper_bicgstabl(medium, params, 'bicgstabl')
    t1 = time.time()
    time_elapsed = t1-t0
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time taken {:0>2}:{:0>2}:{:05.2f}\n".format(int(hours), int(minutes), seconds))