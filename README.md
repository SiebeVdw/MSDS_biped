# Biped Walker Simulation & Optimization Project

## Project Overview
This project is dedicated to developing and simulating an energy-efficient gait for a biped walker, focusing particularly on optimizing the femur-tibia ratio. We model the system's dynamics using kinetic equations within an Euler-Lagrange framework to achieve an efficient and accurate simulation.

## Key Objectives

- **Develop a kinematic model** for a biped walker using Euler-Lagrange equations.
- **Simulate and visualize** the biped walker's movement to verify model accuracy.
- **Conduct forward simulations** with an ODE solver to validate the model under scenarios such as gravitational collapse and controlled motion leading to heel-strike impact.
- **Implement a co-optimization routine** using CasADi to optimize the femur-tibia ratio while balancing algorithm accuracy and runtime.

## Repository Structure
This repository includes several scripts and files essential for modeling, simulation, and optimization of the biped walker:

### Scripts and Files

- **`langrangian_equations.py`**: Contains the Lagrangian equations of motion for the biped system.
- **`model_verification.py`**: Implements verification of the kinematic model through dynamic visualization and simulation tests.
- **`parameters.json`**: Stores adjustable configuration parameters such as limb lengths and mass distributions.
- **`utils.py`**: Provides utility functions for model simulations and data handling.
- **`optimization.py`**: Contains the co-optimization routines executed via CasADi for enhancing the walker's energy efficiency.

## Getting Started
Follow these steps to run the simulations and optimization routines:


1. **Clone the repository**:
   
2. **Install dependencies**:
Ensure Python 3 and necessary packages (CasADi, NumPy, Matplotlib) are installed:

3. **Run the simulations**:
Execute the following command to verify the model and view dynamic simulations:

4. **Perform optimization**:
Execute the optimization routine and review the results:
