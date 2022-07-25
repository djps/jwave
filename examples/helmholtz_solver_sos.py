from functools import partial
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
import jax
from jax import numpy as jnp
from jax import jit, value_and_grad, vmap, random
from jax.example_libraries import optimizers

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

from jwave import FourierSeries
from jwave.geometry import Domain, Medium

from jwave.utils import show_positive_field

from jwave.acoustics.operators import helmholtz
from jwave.acoustics.time_harmonic import helmholtz_solver
from jwave.signal_processing import smooth

def phase_to_apod(phases):
    """
    Constraint on monopole sources
    """
    dim = len(phases) // 2
    return jnp.exp(1j * phases[dim:]) / (1.0 + (phases[:dim]) ** 2)

def phases_to_field(phases, domain):
    """
    sets phases on field. What is position
    """
    phases = phase_to_apod(phases)
    src_field = jnp.zeros(domain.N).astype(jnp.complex64)
    src_field = src_field.at[position, 25].set(phases)
    return FourierSeries(jnp.expand_dims(src_field, -1), domain)

@jit
def update(opt_state, tol, field):
    """
    Update function
    """
    loss_and_field, gradient = loss_with_grad(get_params(opt_state), tol, field)
    lossval = loss_and_field[0]
    field = loss_and_field[1]
    return lossval, field, update_fun(k, gradient, opt_state)

def get_sos(segments, start_point=30, height=4, width=30):
    """
    Assigns local speed of sound
    """
    sos = jnp.ones(N)
    for k in range(len(segments)):
        sos = sos.at[
            start_point + k * height : start_point + (k + 1) * height, 50 : 50 + width
        ].add(jax.nn.sigmoid(segments[k]))
    return FourierSeries(jnp.expand_dims(sos, -1), domain)


def loss(field):
    """
    loss function
    """
    field = field.on_grid
    return -jnp.sum(jnp.abs(field[target[0], target[1]]))


def get_field(params, tol, field):
    """
    returns computed field
    """
    medium = Medium(domain, sound_speed=get_sos(params))
    return helmholtz_solver(
        medium, 1.0, linear_phase, guess=field, tol=tol, checkpoint=False
    )


def full_loss(params, tol, field):
    """
    returns the field and the loss function
    """
    field = get_field(params, tol, field)
    return loss(field), field


if __name__ == '__main__':

    N = (320, 512)  # Grid size
    dx = (1.0, 1.0)  # Spatial resolution
    omega = 1.0  # Wavefield omega = 2*pi*f

    # Making geometry
    domain = Domain(N, dx)

    # Build the vector that holds the parameters of the apodization an the
    # functions required to transform it into a source wavefield
    transmit_phase = jnp.concatenate([jnp.ones((32,)), jnp.ones((32,))])

    position = list(range(32, 32 + (8 * 32), 8))

    linear_phase = phases_to_field(transmit_phase, domain)

    key = random.PRNGKey(12)

    target = [60, 360]  # Target location

    key, _ = random.split(key)
    sos_control_points = random.normal(key, shape=(65,))
    sos = get_sos(sos_control_points)

    show_positive_field(sos, aspect="equal")

    medium = Medium(domain, sound_speed=get_sos(sos_control_points))
    op_params = helmholtz.default_params(linear_phase, medium, omega=1.0)
    print(op_params.keys())

    loss_with_grad = value_and_grad(full_loss, has_aux=True)
    # returns a callable which can return function and gradient as a pair.

    # loss history as empty list
    losshistory = []

    key, _ = random.split(key)
    sos_vector = random.normal(key, shape=(65,))

    init_fun, update_fun, get_params = optimizers.adam(0.1, b1=0.9, b2=0.9)

    opt_state = init_fun(sos_control_points)

    niterations = 10
    tol = 1e-3
    pbar = tqdm(range(niterations))
    
    field = -linear_phase
    for k in pbar:
        lossval, field, opt_state = update(opt_state, tol, field)
        pbar.set_description("Tol: {} Ampl: {:01.4f}".format(tol, -lossval))
        losshistory.append(lossval)

    transmit_phase = get_params(opt_state)

    jnp.savez('optimized_sos.npz', losshistory=losshistory,  target=target, position=position)

    plt.figure(figsize=(10, 6))
    plt.plot(-jnp.array(losshistory))  
    plt.title("Amplitude at target location")

    opt_sos_vector = get_params(opt_state)

    plt.figure(figsize=(10, 6))
    plt.imshow(jnp.abs(field.on_grid), vmax=0.35, cmap="inferno")
    plt.colorbar()
    plt.scatter(target[1], target[0])

    sos = get_sos(opt_sos_vector)

    plt.figure(figsize=(8, 8))
    plt.imshow(sos.on_grid[..., 0])
    plt.title("Sound speed map")
    plt.scatter(target[1], target[0], label="Target")
    plt.legend()

    plt.figure(figsize=(8, 8))
    plt.plot(sos.on_grid[..., 64, 0])

    plt.show()