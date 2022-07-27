import numpy as np
import cv2
import requests
import shutil
from functools import partial
from tqdm import tqdm
from matplotlib import pyplot as plt

from jax import numpy as jnp
from jax import random, nn, value_and_grad, jit, grad, vmap
from jax import lax
from jax.example_libraries import optimizers

from jwave.geometry import Domain, Medium, TimeAxis, Sources, _circ_mask, _points_on_circle, Sensors
from jwave.signal_processing import gaussian_window, apply_ramp, smooth, analytic_signal

from jwave.acoustics import simulate_wave_propagation
from jwave.utils import show_field

from typing import Any, Callable, Dict
from jax import numpy as jnp
from jax import jacfwd, jacrev, vmap, jit, random, eval_shape


def linear_uncertainty(fun: Callable):
    """See Measurements.jl in https://mitmath.github.io/18337/lecture19/uncertainty_programming"""

    def fun_with_uncertainty(mean, covariance, *args, **kwargs):
        mean = mean.real
        covariance = covariance.real

        out_shape = eval_shape(fun, mean, *args, **kwargs).shape

        def f(x):
            y = fun(x, *args, **kwargs)
            return jnp.ravel(y)

        # Getting output meand and covariance
        out_mean = f(mean)
        J = jacfwd(f)(mean)

        out_cov = (jnp.abs(J)**2) @ jnp.sqrt(covariance) # this factor of 4 is odd
        del J
        out_cov = jnp.reshape(out_cov, out_shape)
        out_mean = jnp.reshape(out_mean, out_shape)

        return out_mean, out_cov

    return jit(fun_with_uncertainty)




def monte_carlo(fun: Callable, trials):
    def sampling_function(mean, covariance, key):
        def _sample(mean, L, key, *args, **kwargs):
            noisy_x = mean + jnp.dot(L, random.normal(key, mean.shape))
            return fun(noisy_x, *args, **kwargs)

        mean = mean.real
        covariance = covariance.real
        keys = random.split(key, trials)

        L = jnp.linalg.cholesky(covariance)
        meanval = 0
        var = 0
        for i in range(trials):
            sample = _sample(mean, L, keys[i])
            meanval = meanval + sample/trials
            del sample

        for i in range(trials):
            sample = _sample(mean, L, keys[i])
            var = var + jnp.abs(sample-meanval)**2/trials
            del sample
        return meanval, var

    return sampling_function


def mc_uncertainty(fun: Callable, trials):
    def fun_with_uncertainty(mean, covariance, key):
        return monte_carlo(fun, trials)(mean, covariance, key)
    return fun_with_uncertainty


def read_img(path, N=256, pad=32, threshold=0.4):
    """
    load square image as gray scale and convert to numpy array
    """

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Rescale image to 192x192
    M = int(N - 2 * pad)
    img = cv2.resize(img, (M, M))

    # Convert to numpy array
    img = np.asarray(img)
    img = img / float(N-1)
    img[img < threshold] = 0.0

    # pad image to 256 symmetric
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))

    return img


def get_field(bone_sos):
    """
    Gets field
    """

    sound_speed = img * bone_sos + 1
    
    new_params = params.copy()
    new_params["acoustic_params"]["speed_of_sound"] = jnp.expand_dims(sound_speed, -1)
    
    p = solver(new_params)
    p_max = jnp.mean(jnp.abs(p)**2, axis=0)

    return jnp.sqrt(p_max + 1e-6)


@jit
def compiled_simulator(medium, sources):
    return simulate_wave_propagation(medium, time_axis, sources=sources)


if __name__ == '__main__':
    """
    At the moment (27.07.22) this does not work
    """

    url = 'https://upload.wikimedia.org/wikipedia/commons/7/77/CT_of_sclerotic_lesions_in_the_skull_in_renal_osteodystrophy.jpg'
    fname = 'ct.jpg'

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(fname, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    else: 
        print("could not get image")
    del response

    img = read_img(fname)

    #plt.imshow(img)
    #plt.colorbar()
    #plt.show()

    # Settings
    N = (256, 256)
    dx = (0.5, 0.5)
    cfl = 0.3
    t_end = 120
    num_sources = 1
    source_freq = 0.5
    source_mag = 5.0

    random_seed = random.PRNGKey(42)

    # Define domain
    domain = Domain(N, dx)

    # Define medium
    sound_speed = jnp.asarray(img) + jnp.ones(N)
    density = jnp.ones(N)
    attenuation = jnp.zeros(N) 
    pml_size = int(15)
    medium = Medium(domain, sound_speed=sound_speed, density=density, attenuation=attenuation, pml_size=pml_size)

    # Time axis
    time_axis = TimeAxis.from_medium(medium, cfl=cfl, t_end=t_end)
    # time step
    dt = time_axis.dt
    # time array
    t = time_axis.to_array()

    # Sources
    radius = 110 
    centre = (128, 128)
    x, y = _points_on_circle(num_sources, radius, centre)
    source_positions = (jnp.array(x), jnp.array(y))

    source_mag = source_mag / dt
    
    s1 = source_mag * jnp.sin(2.0 * jnp.pi * source_freq * t)
    mu = 5.0 
    sigma = 2.0
    signal = gaussian_window(apply_ramp(s1, dt, source_freq, warmup_cycles=3.0), t, mu, sigma)
    print(signal, dir(signal), signal.xla_shape)

    src_signal = jnp.stack([signal]*num_sources)
    print(src_signal, src_signal.shape, t.shape)

    sources = Sources(
        source_positions,
        src_signal,
        dt,
        domain
    )

    print(dir(medium))
    print(dir(medium.domain), medium.domain.ndim)

    print(medium.sound_speed.shape, medium.density.shape, jnp.shape(medium.attenuation) )

    print(dir(sources))
    print(dir(sources.on_grid), jnp.shape(sources.on_grid),  jnp.shape(sources.signals), )

    #pressure = compiled_simulator(medium, sources)
    #print("done 0")

    # Run simulations
    pressure = simulate_wave_propagation(medium, time_axis, sources=sources)
    print("done 1")

    params, solver = simulate_wave_propagation(
        medium=medium,
        time_array=time_axis,
        sources=sources,
        checkpoint=True
    )
    print("done 2")
    
    print( params["source_signals"].shape )
    print( params["acoustic_params"]["speed_of_sound"].shape )

    p = get_field(0.7)
    plt.imshow(p, cmap="inferno", vmax=0.4)
    plt.colorbar()

    x = jnp.array([1.])
    covariance = jnp.array([[(0.05)**2]])
    
    get_field_lup = linear_uncertainty(get_field)
    mu_linear, cov_linear = get_field_lup(x, covariance)
    plt.imshow(cov_linear, cmap="inferno", vmin=0.0, vmax=0.02)
    plt.title("Linear uncertainty propagation")
    plt.colorbar()

    get_field_lup = mc_uncertainty(get_field, 20)
    mu_mc, cov_mc = get_field_lup(x, jnp.sqrt(covariance), random.PRNGKey(0))
    plt.imshow(cov_mc, cmap="inferno", vmin=0.0, vmax=0.02)
    plt.title("Monte Carlo (N=20) Uncertainty")
    plt.colorbar()

    plt.show()
