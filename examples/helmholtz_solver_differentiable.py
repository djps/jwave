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

from jwave.utils import plot_complex_field, show_positive_field

from jwave.acoustics.operators import helmholtz
from jwave.acoustics.time_harmonic import helmholtz_solver, helmholtz_solver_verbose
from jwave.signal_processing import smooth

def phase_to_apod(phases):
    dim = len(phases) // 2
    return jnp.exp(1j * phases[dim:]) / (1.0 + (phases[:dim]) ** 2)

def phases_to_field(phases, domain):
    phases = phase_to_apod(phases)
    src_field = jnp.zeros(domain.N).astype(jnp.complex64)
    src_field = src_field.at[position, 25].set(phases)
    return FourierSeries(jnp.expand_dims(src_field, -1), domain)

@jit
def fixed_medium_solver(src_field, op_params, guess=None, tol=1e-3):
    return helmholtz_solver(medium, omega, src_field, guess=guess, tol=tol, params=op_params)

def loss(field):
    field = field.on_grid
    return -jnp.sum(jnp.abs(field[target[0], target[1]]))

def get_field(transmit_phase, tol, guess, op_params):
    transmit_field = phases_to_field(transmit_phase, domain)
    return fixed_medium_solver(transmit_field, op_params, guess, tol)

def full_loss(transmit_phase, tol, guess, op_params):
    field = get_field(transmit_phase, tol, guess, op_params)
    return loss(field), field

@partial(jit, static_argnums=(1,))
def update_static(opt_state, tol, guess, op_params):
    loss_and_field, gradient = loss_with_grad(get_params(opt_state), tol, guess, op_params)
    lossval = loss_and_field[0]
    field = loss_and_field[1]
    return lossval, field, update_fun(k, gradient, opt_state)

@jit
def update(opt_state, tol, field):
    """
    Problematic function
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


if __name__ == '__main__':

    key = random.PRNGKey(42)

    N = (320, 512)  # Grid size
    dx = (1.0, 1.0)  # Spatial resolution
    omega = 1.0  # Wavefield omega = 2*pi*f
    target = [160, 360]  # Target location

    # Making geometry
    domain = Domain(N, dx)

    # Build the vector that holds the parameters of the apodization an the
    # functions required to transform it into a source wavefield
    transmit_phase = jnp.concatenate([jnp.ones((32,)), jnp.ones((32,))])
    position = list(range(32, 32 + (8 * 32), 8))

    linear_phase = phases_to_field(transmit_phase, domain)

    """
    # Constructing medium physical properties
    sound_speed = jnp.ones(N)
    sound_speed = sound_speed.at[20:80, 50:80].set(2.0)
    sound_speed = sound_speed.at[80:140, 50:100].set(1.5)
    sound_speed = sound_speed.at[140:220, 45:130].set(1.3)
    sound_speed = jnp.expand_dims(sound_speed.at[220:300, 70:100].set(1.8), -1)
    sound_speed = FourierSeries(sound_speed, domain)

    medium = Medium(domain=domain, sound_speed=sound_speed, pml_size=15)

    plt.figure(figsize=(8, 5))

    plt.imshow(medium.sound_speed.on_grid)
    plt.title("Sound speed map")
    plt.scatter([25] * len(position), position, marker=".", label="Transducers")
    plt.scatter(target[1], target[0], label="Target", marker='x')
    plt.colorbar()

    op_params = helmholtz.default_params(linear_phase, medium, omega=1.0)
    print("Parameter names: " + str(op_params.keys()))

    field = fixed_medium_solver(linear_phase, op_params)
    _ = plot_complex_field(field, figsize=(20, 20))

    loss_with_grad = value_and_grad(full_loss, has_aux=True)

    losshistory = []

    init_fun, update_fun, get_params = optimizers.adam(0.1, b1=0.9, b2=0.9)

    opt_state = init_fun(transmit_phase)

    niterations = 100
    tol = 1e-3
    guess = None
    pbar = tqdm(range(niterations))
    for k in pbar:
        lossval, guess, opt_state = update_static(opt_state, tol, guess, op_params)
        # For logging
        pbar.set_description("Ampl: {:01.4f}".format(-lossval))
        losshistory.append(lossval)

    transmit_phase = get_params(opt_state)

    jnp.savez_compressed('optimized_static.npz', losshistory=losshistory, guess=guess, opt_state=opt_state, target=target, position=position)

    fig, ax = plt.subplots(1, 2, figsize=(8,2), dpi=200)

    im1 = ax[0].imshow(medium.sound_speed.on_grid, cmap="PuBu")
    cbar = fig.colorbar(im1, ax=ax[0])
    cbar.ax.get_yaxis().labelpad = 15
    ax[0].scatter([25] * len(position), position, marker=".", color="black", label="Transducers")
    ax[0].scatter(target[1], target[0], label="Target", color="green", marker='o')
    ax[0].axis('off')
    ax[0].set_title('Speed of sound map')
    ax[0].legend()

    im1 = ax[1].imshow(jnp.abs(guess.on_grid), cmap="inferno", vmax=0.5)
    cbar = fig.colorbar(im1, ax=ax[1])
    cbar.ax.get_yaxis().labelpad = 15
    ax[1].axis('off')
    ax[1].set_title('Focused field amplitude')
    ax[1].scatter(target[1], target[0], label="Target", color="green", marker='o')

    plt.figure(figsize=(10, 3))
    plt.plot(jnp.real(phase_to_apod(transmit_phase)))
    plt.plot(jnp.imag(phase_to_apod(transmit_phase)))
    # plt.plot(jnp.abs(phase_to_apod(transmit_phase)), "r.")
    plt.title("Apodization")

    plt.figure(figsize=(10, 3))
    plt.plot(-jnp.array(losshistory))
    plt.title("Amplitude at target location")
    plt.xlabel("Optimization step")
    plt.grid(True)
    
    #plt.show() 
    """

    # Speed of sound gradients
    key = random.PRNGKey(12)

    target = [60, 360]  # Target location

    key, _ = random.split(key)
    sos_control_points = random.normal(key, shape=(65,))
    sos = get_sos(sos_control_points)

    show_positive_field(sos, aspect="equal")

    medium = Medium(domain, sound_speed=get_sos(sos_control_points))
    op_params = helmholtz.default_params(linear_phase, medium, omega=1.0)
    print(op_params.keys())

    # does this work? 
    loss_with_grad = value_and_grad(full_loss, has_aux=True)
    # returns a callable which can return function and gradient as a pair.
    print(loss_with_grad, dir(loss_with_grad))


    # reset loss history
    losshistory = []

    key, _ = random.split(key)
    sos_vector = random.normal(key, shape=(65,))

    init_fun, update_fun, get_params = optimizers.adam(0.1, b1=0.9, b2=0.9)

    # here
    opt_state = init_fun(sos_control_points)

    niterations = 100
    tol = 1e-3
    pbar = tqdm(range(niterations))
    
    field = -linear_phase
    for k in pbar:
        lossval, field, opt_state = update(opt_state, tol, field)
        # For logging
        pbar.set_description("Tol: {} Ampl: {:01.4f}".format(tol, -lossval))
        losshistory.append(lossval)

    transmit_phase = get_params(opt_state)

    jnp.savez_compressed('optimized_1.npz', losshistory=losshistory, opt_state=opt_state, target=target, position=position)

    plt.plot(-jnp.array(losshistory))  #
    plt.title("Amplitude at target location")
    plt.Text(0.5, 1.0, 'Amplitude at target location')

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
