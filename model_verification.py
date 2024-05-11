"""
This script (jupyter notebook style) is used to verify the model. 
The script uses the T, V, M and C functions from Lagrangian_equations.py and extra help functions from utils.py.

Optional verifications are:
    - gravitaional collapse
    - frozen body (with or without heelstrike)
Results are:
    - animation, stored in a .gif file
    - energy plot
"""

#%% imports
from utils import *
import numpy as np

#%% T, V, M and C (numerical casadi functions) are imported from Lagrangian_equations.py
from Lagrangian_equations import lag_eq
T, V, M, C, B, x, xd = lag_eq()

#%% solve the equations of motion, heelstrike included

# Choose what to use:
Gravitational_collapse = False
Frozen_body = True
Fully_frozen_body = True # keep frozen body after heelstrike
Heelstrike = True

if Gravitational_collapse:
    model = dSdt
if Frozen_body or Fully_frozen_body:
    model = dSdt_frozen_body
if Heelstrike:
    integrator = rk4_heelstrike
else:
    integrator = rk4

# time array, initial conditions and counter
times = np.linspace(0, 20, 2001)
q0 = np.array([220.0, 270.0, -60.0, 0.0, -25.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * np.pi / 180
t = np.array([])
ans = np.array([])
n_heelstrikes = 0

# keep track of all times the foot hits the ground so we can add the offset to the x coordinates
transitions = []
x_values = []

while len(t) < len(times) and n_heelstrikes < 10:
    # apply rk4, will stop when the foot hits the ground
    t0, ans0, x_value = integrator(M, C, model, q0, times)

    # add to the previous solution
    if n_heelstrikes == 0:
        t = t0
        ans = ans0
    else:
        t0 = t0 + t[-1]
        t = np.concatenate([t, t0[1:]])
        ans = np.concatenate([ans, ans0[1:]])

    # Heelstrike
    if Heelstrike:
        q_min, q_d_min = ans[-1][:5], ans[-1][5:]
        q_plus, q_d_plus = heelstrike(M, q_min, q_d_min)

        # Fully frozen body
        if Fully_frozen_body:
            q_d_plus = np.array([0.0, 0.0, 0.0, 0.0, q_d_plus[-1][0]])

        # create new initial conditions
        q0 = np.concatenate([q_plus, q_d_plus.flatten()])

        transitions.append(t[-1])
        x_values.append(x_value)
        n_heelstrikes += 1

print(f"number of heelstrikes: {n_heelstrikes}")

# save the data
save_animation_data(t, ans, transitions, x_values)

#%% option to plot energy
plot_energy(t, ans, T, V)

#%% option to make a gif
make_animation(t, ans, transitions, x_values, 'model_verification_results')

# %%
