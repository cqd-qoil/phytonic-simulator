# Photonic experiment simulator
---

This project provides a simple framework for designing discrete-variable quantum optics experiments using Python.
It models the propagation of single-photon states through complex optical setups and allows for post-selection (or heralding) based on particular detector clicks.
In a nutshell, the simulator follows a **photon state** > **series of optical elements** > **detector array** flow, where photons are initialised, pass through various time-ordered optical elements, and are detected by detectors that collapse the final quantum state.

Rather than doing large matrix multiplications, the simulator follows an algebraic approach based on creation operators: an initial photon state is evolved in discrete steps, where each optical element transforms the creation operators according to particular substitution rules defined by a unitary matrix (such as Jones matrices for polarisation modes, for example).
At the end of the circuit, an array of detectors is used to calculate the coincidence probability and return the final quantum state.

Some attractive features of the simulator:
- It supports multiple degrees of freedom, such as path, polarisation, time, or even custom ones.
- It is relatively straightforward to implement new optical elements by defining the suitable substitution rules for creation operators.
- At each step, it does not simulate the entire Hilbert space (which grows exponentially with the number of modes), but instead acts only the relevant modes.

The core idea takes inspiration from [Zhihao Wu et al, Quantum Sci. Technol. 6, 024001 (2021)](https://iopscience.iop.org/article/10.1088/2058-9565/abc1ba). Although the idea was very nice, at the time, the project did not provide any public code, so I decided to program my own.

## Requirements

To run the simulator, you basically only need:

- `sympy`, for all the symbolic mathematics,
- `numpy`, for everything else

## Usage
See the linked notebooks for examples. Below is a simplified guide to simulating your own experiments.

### 1. Initialise experiment
The first step is to initialise an experiment, which will be the main object in the simulator. We will add photon states, optical elements, and detectors to the `Experiment()` object.

```python
import sympy as sp
from qoilsimul import *

experiment = Experiment()
```

### 2. Define photon states

Next, you need to define **photon states** with a given number of path modes, starting from `p1`, `p2`, and so on. The label `p{n}` denotes the path information of the individual photon state. Each time you create a photon object, you should start with `p1`. After all the photons have been created, the experiment instance is automatically updated to configure the *global* path labels `a, b, c, ...` depending on how many photons you add.

You also need to define any other degrees of freedom (DOF) as required. The function `co(dof1, dof2, ...)` represents a creation operator associated with the degrees of freedom `dof1, dof2, ...` (e.g., path, polarisation, time-bin, etc.). The symbol `x` is reserved to denote creation operator. For example, one horizontally polarised photon in path `a`, created with `co(a,H)`, represents the operator $\hat{a}_ {H}^\dagger$ but will be stylised as $x_{a,H}$ in the simulator.


The following line will create a photon in a polarisation superposition but only one path mode:

```python
photon_1 = Photon(pathModes=1, dof=['path', 'pol'], state=(co(p1,H)+co(p1,V))/sp.sqrt(2))
```

whereas the following code creates a photon in a path *and* polarisation superposition:

```python
photon_2 = Photon(pathModes=2, dof=['path', 'pol'], state=(co(p1,H)+co(p2,V))/sp.sqrt(2))
```

One can also prepare multi-photon states (with an associated normalisation factor), states with symbolic amplitudes, or states with arbitrary degrees of freedom:

```python
# Two-photon state, normalised
photon_3 = Photon(pathModes=1, dof=['path'], state=co(p1)**2/sp.sqrt(2))

# Symbolic amplitude
alpha, beta = sp.var('α β') # Create symbols
photon_4 = Photon(pathModes=1, dof=['path', 'pol'], state=alpha*co(p1,H)+beta*co(p1,V))

# Arbitrary DOF
sp.var('t1:4', cls=sp.IndexedBase) # Create three time bins t1, t2, ..., t3 as as IndexedBase (important)
photon_5 = Photon(pathModes=1, dof=['path', 'time'],
                  state=(co(p1,t1)+co(p1,t2)+co(p1,t3))/sp.sqrt(3)) # Time-bin qutrit

sp.var('custom1:10', cls=sp.IndexedBase) # One is also free to create custom DOFs
photon_6 = Photon(pathModes=1, dof=['path', 'custom_dof'],
                  state=(co(p1,custom2)+co(p1,custom7))/sp.sqrt(2)) # Custom DOF superposition
```

Finally, one must add the photon objects to `experiment`. One can add however many photon states to the `experiment` instance, and it will update the global path modes accordingly.

```python
experiment.addPhotons(photon_1, photon_2)
```

One can now print the overall state of the experiment by calling `experiment.state`. Since `photon_1` had one path mode and `photon_2` had to path modes, `experiment` will now have three total path modes labelled `a,b,c` and the initial state will be

$$\frac{(x_{a,H}+x_{a,V})(x_{b,H}+x_{c,V})}{2}.$$

### 3. Creating optical elements

The simulator includes many commonly used **optical elements**, such as wave plates, beam splitters (polarising, non-polarising, and partially polarising), phase shifters, polarisation-based beam displacers, etc.
These elements are defined as objects that act on the photonic states and modify the quantum state of the photons as they pass through them.
You can define optical elements with specific parameters, including wave plate angles and beam splitter reflectivities. You can also add custom labels to each elements to identify them within the experiment.

```python
gamma, delta = sp.var('γ δ')
half_wave_plate_1 = HWP(theta=gamma)
beam_splitter_1 = BS(r=sp.Rational(1/2) label='50:50') # If not using sp.Rational, python will convert to float
beam_splitter_2 = BS(r=delta)
```

Optical elements acts (at least) on the path DOF. For example, a beam splitter with 50:50 ratio has the two substitution rules for $x_{a}$ and $x_{b}$:

$$\begin{align}
x_{a} &\xrightarrow{} \left(x_{a} + x_{b}\right)/\sqrt{2} \\
x_{b} &\xrightarrow{} \left(x_{a} - x_{b}\right)/\sqrt{2}.
\end{align}$$

Some optical elements, such as polarisation optics, also act on other DOFs.
A neat feature of the simulator is that the substitution rules for any component only affect the relevant DOFs.

In the beam splitter example above, this means that the substitution rules will actually be, under the hood:

$$\begin{align}
x_{a, \ \text{everything else}} &\xrightarrow{} \left(x_{a, \ \text{everything else}} + x_{b, \ \text{everything else}}\right)/\sqrt{2} \\
x_{b, \ \text{everything else}} &\xrightarrow{} \left(x_{a, \ \text{everything else}} - x_{b, \ \text{everything else}}\right)/\sqrt{2},
\end{align}$$

which updates the path of an incident photon but leaves everything else unchanged.
In the traditional matrix-based approach to photonic circuit simulation, one would need to define a (potentially very large) matrix which will depend on the total number of modes in the system.

### 4. Connecting one element to another

Each optical element has an array of inputs (`<Element>.i`) and outputs (`<Element>.o`); in general, `len(<Element>.i)==len(<Element>.o)`.
`Photon` objects do not have inputs but they do have outputs.

After creating an element, one must connect its input to the output of a preceding element or photon object.
This step defines the flow of photons through the optical circuit: the simulator allows you to wire together optical elements in an intuitive way.

Some optical elements may have multiple inputs. In these cases, you can inject vacuum modes into the additional inputs. The `<Experiment>.emptyMode()` method dynamically creates a new path mode and adds it to the experiment.
For example, if `experiment` had path modes `a,b,c` at initialisation, calling `emptyMode()` will automatically create and add mode `d`.

After connecting all optical elements, one must add them (in order) to the experiment, similar to adding photon objects.


```python
half_wave_plate_1.i = [photon_2.o[0]]
beam_splitter_1.i   = [photon_1.o[0], half_wave_plate_1.o[0]]
beam_splitter_2.i   = [beam_splitter_1.o[0], experiment.emptyMode()]

experiment.addElements(half_wave_plate_1, beam_splitter_1, beam_splitter_2)
```

### 5. Adding detectors

In the simulator, **detectors** are used to measure the final state of the photon(s) after they have passed through the optical elements. Detectors are connected to specific path modes of interest.

One can create an array of two detectors by calling `Detectors(numberOfDetectors=2)` and then connect its input path array to the relevant path modes, before adding it to the `Experiment` object.
The simulator will then look at two-fold coincidences only.

```python
detectors = Detectors(numberOfDetectors=2) # Create
detectors.i = [beam_splitter_2.o[0], beam_splitter_2.o[1]] # Connect
experiment.addDetectors(detectors) # Add
```

### Optional: preview the circuit

At this point, you can generate a minimalistic ASCII representation of the circuit with `experiment.generateCircuit()`, giving you an easy to read summary of the experiment.
To display it, simply print it into the terminal.

```python
experiment.generateCircuit()
print(experiment.circuit)
```
```
Output:
a : p -----50:50-BS---D
b : p -HWP-50:50-------
c : p -----------------
d : .............BS---D
```
Here, `p` denotes a path mode with an initial photon and `D` indicates a detector. One can also see the labels associated to each optical element.

### 6. Time to run!

The final step is to run the experiment by calling `experiment.build()`. The global photon state after passing through all the optical elements (but before hitting the detectors) can be obtained by calling `experiment.state`, while the state conditioned on the correct detectors clicking is given by `experiment.postSelectedState`.
You can also get the success probability of getting this post-selected state with `experiment.successProbability`.

If using symbolic variables (in either the photon state or parameters of the optical elements), you can convert the symbolic expression for the success probability into a numerical function using `s = sp.lambdify(symbol1, symbol2, ..., experiment.successProbability)`.


## What's next?

- Although the simulator works really well for pure states, I haven't implemented functionality for mixed states yet.
- Supporting multiple detector combinations at the same time. This is particularly important for multi-photon experiments, where one might be interested in recording simultaneous four-fold, three-fold, and two-fold coincidences, for example.
