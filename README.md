<div align="center">
<img src="docs/assets/images/jwave_logo.png" alt="logo"></img>
</div>

# j-Wave: Differentiable acoustic simulations in JAX

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](LICENSE)
[![Continous Integration](https://github.com/djps/jwave/actions/workflows/main.yml/badge.svg)](https://github.com/djps/jwave/actions/workflows/main.yml) 
[![Coverage Status](https://coveralls.io/repos/github/djps/jwave/badge.svg)](https://coveralls.io/github/djps/jwave)
<!-- 
[![codecov](https://codecov.io/gh/ucl-bug/jwave/branch/main/graph/badge.svg?token=6J03OMVJS1)](https://codecov.io/gh/ucl-bug/jwave)
[![Documentation](https://github.com/ucl-bug/jwave/actions/workflows/build_docs.yml/badge.svg)](https://ucl-bug.github.io/jwave)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ucl-bug/jwave/main?labpath=docs%2Fnotebooks%2Fivp%2Fhomogeneous_medium.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xAHAognF1v9un6GNvaGPSfdVeCDK8l9z?usp=sharing) 
-->

[Install](#install) | [**UCL** Tutorials](https://ucl-bug.github.io/jwave/notebooks/ivp/homogeneous_medium.html) | [**UCL** Documentation](https://ucl-bug.github.io/jwave) | [Changelog](HISTORY.md)

j-Wave is a library of simulators for acoustic applications. Is heavily inspired by [k-Wave](http://www.k-wave.org/) (a big portion of j-Wave is a port of k-Wave in JAX), and its intented to be used as a collection of modular blocks that can be easily included into any machine learning pipeline.

Following the phylosophy of [JAX](https://jax.readthedocs.io/en/stable/), j-Wave is developed with the following principles in mind

1. To be differntiable
2. To be fast via `jit` compilation
3. Easy to run on GPUs
4. Easy to customize

<br/>

## Install

Follow the instructions to install [Jax with CUDA support](https://github.com/google/jax#installation) if you want to use your GPU.

Then, install `jaxdf` and then `jwave` using pip

```bash
pip install git+https://github.com/djps/jwave.git
```

<!--
The pip option -e will use setuptools, so use setup.py which I haven't edited.

For more details, see the [Linux install guide](docs/install/on_linux.md).

See the [Install on Windows](docs/install/on_win.md) guide for more details. -->

<br/>

## Example

This [example](examples/basic.py) simulates an acoustic initial value problem, which is often used as a simple model for photoacoustic acquisitions:

```python
from jax import jit

from jwave import FourierSeries
from jwave.acoustics.time_varying import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis
from jwave.utils import load_image_to_numpy

# Simulation parameters
N, dx = (256, 256), (0.1e-3, 0.1e-3)
domain = Domain(N, dx)
medium = Medium(domain=domain, sound_speed=1500.0)
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=.8e-05)

# Initial pressure field
p0 = load_image_to_numpy("docs/assets/images/jwave.png", image_size=N)/255.
p0 = FourierSeries(p0, domain)

# Compile and run the simulation
@jit
def solver(medium, p0):
  return simulate_wave_propagation(medium, time_axis, p0=p0)

pressure = solver(medium, p0)
```

![Simulated pressure field](docs/assets/images/readme_example_basic.png)


<br/>

## Citation

[![arXiv](https://img.shields.io/badge/arXiv-2207.01499-b31b1b.svg?style=flat)](https://arxiv.org/abs/2207.01499)

If you use `jwave` for your research, please consider citing it as:

```bibtex
@article{stanziola2022jwave,
    author={Stanziola, Antonio and Arridge, Simon R. and Cox, Ben T. and Treeby, Bradley E.},
    title = {j-Wave: An open-source differentiable wave simulator},
    publisher = {arXiv},
    year = {2022},
}
```

<br/>

## Useful Papers

* [Optimizing a DIscrete Loss (ODIL) to solve forward and inverse problems for partial differential equations using machine learning tools](https://arxiv.org/pdf/2205.04611.pdf)


<br/>

## Related Projects

1. [`ADSeismic.jl`](https://github.com/kailaix/ADSeismic.jl): a finite difference acoustic simulator with support for AD and JIT compilation in Julia.
2. [`stride`](https://github.com/trustimaging/stride): a general optimisation framework for medical ultrasound tomography.
3. [`k-wave-python`](https://github.com/waltsims/k-wave-python): A python interface to k-wave GPU accelerated binaries
4. [`jaxwell`](https://github.com/stanfordnqp/jaxwell): for Maxwell's equation. 

