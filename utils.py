# imports
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import animation
import datetime
import casadi as ca


with open('parameters.json') as f:
    data = json.load(f)

l1, l2, l3, l4, l5 = data['l1'], data['l2'], data['l3'], data['l4'], data['l5']
r1, r2, r3, r4, r5 = data['r1'], data['r2'], data['r3'], data['r4'], data['r5']
m1, m2, m3, m4, m5 = data['m1'], data['m2'], data['m3'], data['m4'], data['m5']
I1, I2, I3, I4, I5 = data['I1'], data['I2'], data['I3'], data['I4'], data['I5']


# file with help functions

def dSdt(M, C, S, t):
    q1, q2, q3, q4, q5, q1_d, q2_d, q3_d, q4_d, q5_d = S
    m = M([q1, q2, q3, q4, q5])
    c = C([q1, q2, q3, q4, q5], [q1_d, q2_d, q3_d, q4_d, q5_d])
    sol = np.linalg.solve(m, c)
    return np.array([q1_d, q2_d, q3_d, q4_d, q5_d, sol[0][0], sol[1][0], sol[2][0], sol[3][0], sol[4][0]])

def dSdt_frozen_body(M, C, S, t):
    q1, q2, q3, q4, q5, q1_d, q2_d, q3_d, q4_d, q5_d = S
    m = M([q1, q2, q3, q4, q5])
    c = C([q1, q2, q3, q4, q5], [q1_d, q2_d, q3_d, q4_d, q5_d])
    sol = [ 0, 0, 0, 0, float(c[-1] / m[-1, -1]) ] 
    return np.array([q1_d, q2_d, q3_d, q4_d, q5_d, sol[0], sol[1], sol[2], sol[3], sol[4]])


def rk4_heelstrike(M, C, f, y0, t):
    x_value = 0.0
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    h = t[1] - t[0]
    for i in range(len(t)-1):
        k1 = f(M, C, y[i], t[i])
        k2 = f(M, C, y[i] + h/2*k1, t[i] + h/2)
        k3 = f(M, C, y[i] + h/2*k2, t[i] + h/2)
        k4 = f(M, C, y[i] + h*k3, t[i] + h)
        y[i+1] = y[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
        if (get_all_coordinates(y[i+1][:5])[-3] < 0.0):
            y = y[:i+2]
            t = t[:i+2]
            x_value = get_all_coordinates(y[i+1][:5])[-4]
            return t, y, x_value
    return t, y, x_value

def rk4(M, C, f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    h = t[1] - t[0]
    for i in range(len(t)-1):
        k1 = f(M, C, y[i], t[i])
        k2 = f(M, C, y[i] + h/2*k1, t[i] + h/2)
        k3 = f(M, C, y[i] + h/2*k2, t[i] + h/2)
        k4 = f(M, C, y[i] + h*k3, t[i] + h)
        y[i+1] = y[i] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return t, y, 0.0

def heelstrike(M, q_min, q_d_min):
    q_plus = np.array([q_min[1], q_min[0], q_min[3], q_min[2], q_min[4]])
    m_min = M(q_min)
    m_plus = M(q_plus)
    q_d_plus = np.linalg.solve(m_plus, m_min @ q_d_min)
    return q_plus, q_d_plus

def heelstrike_casadi(M, q_min, q_d_min):
    q_plus = ca.vertcat(q_min[1], q_min[0], q_min[3], q_min[2], q_min[4])
    m_min = M(q_min)
    m_plus = M(q_plus)
    q_d_plus = ca.solve(m_plus, m_min @ q_d_min)
    return ca.vertcat(q_plus[0], q_plus[1], q_plus[2], q_plus[3], q_plus[4], 
                      q_d_plus[0], q_d_plus[1], q_d_plus[2], q_d_plus[3], q_d_plus[4])

# plot T, V and E
def plot_energy(t, ans, T, V):
    T_array = np.zeros(len(t))
    V_array = np.zeros(len(t))
    E_array = np.zeros(len(t))
    for i in range(len(t)):
        T_array[i] = T(ans[i][0:5], ans[i][5:10])
        V_array[i] = V(ans[i][0:5])
        E_array[i] = T_array[i] + V_array[i]
    plt.plot(t, T_array, label='Kinetic energy')
    plt.plot(t, V_array, label='Potential energy')
    plt.plot(t, E_array, label='Total energy')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy over time')
    plt.legend()
    plt.show()

def save_animation_data(t, ans, transitions, x_values):
    date_string = datetime.datetime.now().strftime("%d%b_%Hh%M")
    np.save(f'model_verification_results/{date_string}_t.npy', t)
    np.save(f'model_verification_results/{date_string}_ans.npy', ans)
    np.save(f'model_verification_results/{date_string}_transitions.npy', transitions)
    np.save(f'model_verification_results/{date_string}_x_values.npy', x_values)
    print(f"Data saved in model_verification_results/{date_string}... .npy")

    
def make_animation(t, ans, transitions, x_values, foldername):
    (x_G1, y_G1, x_G2, y_G2, x_G3, y_G3, x_G4, y_G4, x_G5, y_G5, 
     x_B, y_B, x_C, y_C, x_D, y_D, x_E, y_E, x_F, y_F
     ) = get_all_coordinates(ans.T[:5])
    x_A = np.zeros(len(t))

    for i in range(len(transitions)):
        x_G1[t > transitions[i]] += x_values[i]
        x_G2[t > transitions[i]] += x_values[i]
        x_G3[t > transitions[i]] += x_values[i]
        x_G4[t > transitions[i]] += x_values[i]
        x_G5[t > transitions[i]] += x_values[i]
        x_A[t > transitions[i]] += x_values[i]
        x_B[t > transitions[i]] += x_values[i]
        x_C[t > transitions[i]] += x_values[i]
        x_D[t > transitions[i]] += x_values[i]
        x_E[t > transitions[i]] += x_values[i]
        x_F[t > transitions[i]] += x_values[i]

    def animate(i):
        ln1.set_data([x_A[i], x_B[i], x_C[i], x_F[i]], [0.0, y_B[i], y_C[i], y_F[i]])
        ln2.set_data([x_C[i], x_D[i], x_E[i]], [y_C[i], y_D[i], y_E[i]])

    fig, ax = plt.subplots(1, 1)

    ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
    ln2, = plt.plot([], [], 'ro--', lw=3, markersize=8)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ani = animation.FuncAnimation(fig, animate, frames = len(t), interval=50)

    # add a line at y = 0 (ground)
    ax.axhline(0, color='black', lw=1)

    # save animation in a map named gifs
    date_string = datetime.datetime.now().strftime("%d%b_%Hh%M")
    ani.save(f"model_verification_gifs/{date_string}.gif", writer='pillow', fps=25)
    print(f"Animation saved in {foldername}/{date_string}.gif")


def get_all_coordinates(q):
    q1, q2, q3, q4, q5 = q
    
    # link 3
    alpha_3 = q1 + q5 + q3 + np.pi
    x_G3 = (l3-r3)*np.sin(-alpha_3)
    y_G3 = (l3-r3)*np.cos(-alpha_3)

    x_B = l3*np.sin(-alpha_3)
    y_B = l3*np.cos(-alpha_3)

    # link 1
    alpha_1 = np.pi*3/2 - q1 - q5
    x_G1 = x_B - (l1-r1)*np.cos(alpha_1)
    y_G1 = y_B + (l1-r1)*np.sin(alpha_1)

    x_C = x_B - l1*np.cos(alpha_1)
    y_C = y_B + l1*np.sin(alpha_1)
    # link 5
    x_G5 = x_C + r5*np.sin(-q5)
    y_G5 = y_C + r5*np.cos(-q5)

    x_F = x_C + l5*np.sin(-q5)
    y_F = y_C + l5*np.cos(-q5)

    # link 2
    alpha_2 = q5 + q2 - np.pi/2
    x_G2 = x_C - r2*np.cos(alpha_2)
    y_G2 = y_C - r2*np.sin(alpha_2)

    x_D = x_C - l2*np.cos(alpha_2)
    y_D = y_C - l2*np.sin(alpha_2)

    # link 4
    alpha_4 = q5 + q2 + q4 - np.pi/2
    x_G4 = x_D - r4*np.cos(alpha_4)
    y_G4 = y_D - r4*np.sin(alpha_4)

    x_E = x_D - l4*np.cos(alpha_4)
    y_E = y_D - l4*np.sin(alpha_4)

    return (x_G1, y_G1, x_G2, y_G2, x_G3, y_G3, x_G4, y_G4, x_G5, y_G5, x_B, y_B, x_C, y_C, x_D, y_D, x_E, y_E, x_F, y_F)
