# qsub Package

## Overview
The `qsub` package is a Python framework designed for expressing complex quantum algorithms and predicting their costs. At the heart of `qsub` is the `SubroutineModel` class, which facilitates the hierarchical construction of algorithms. Users can build algorithms by linking together various subroutines and profiling their overall cost.

## Installation

1. **Clone the Repository:**
   ```
   git clone git@github.com:zapatacomputing/qsub.git
   ```

2. **Navigate to the Cloned Directory:**
   ```
   cd qsub
   ```

3. **Install the Package:**
   ```
   pip install .
   ```
   Alternatively, if you're actively developing the package and want to install it in editable mode:
   ```
   pip install -e .
   ```

4. **Verify Installation:**
   You can verify that the package has been installed correctly by importing it in a Python interpreter or script:
   ```
   python
   >>> import qsub
   ```

These steps assume that you have Python and pip installed on your system. If not, please install Python from [python.org](https://www.python.org/downloads/) and follow the instructions to install pip.





## Usage

### Basic Example
Here's a quick example to get you started:

```python
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    QuantumAmplitudeEstimation,
)

# Initialize your algorithm
amp_est = QuantumAmplitudeEstimation()


# Set requirements
estimation_error = 0.001
failure_tolerance = 0.01

amp_est.set_requirements(
    estimation_error=estimation_error,
    failure_tolerance=failure_tolerance,
)

# Run the profile for this subroutine
amp_est.run_profile()
amp_est.print_profile()

print("qubits =", amp_est.count_qubits())

print(amp_est.count_subroutines())
```

### Advanced Usage
For more complex scenarios, refer to the documentation on linking subroutines and managing task hierarchies.

## Features
- `SubroutineModel`: Core class for creating and managing subroutines.
- Cost Analysis Tools: Tools for profiling the cost and resource requirements of algorithms.
- Flexible Subroutine Integration: Easily swap and integrate different subroutines.

## Contributing
We welcome contributions! If you're interested in adding more subroutines or enhancing the framework, please follow these steps:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -am 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

```
```
