"""
Gait optimization is performed with the optimizer function.
As the ratio can be chosen, the nested optimization is also performed with the optimizer function.
"""

# imports
import numpy as np
import casadi as ca
from Lagrangian_equations import lag_eq 
from utils import heelstrike_casadi
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches


def optimizer(N, L, T, ratio = 0.5, initial_guess = None):
    # imports from Lagrangian_equations
    _, V, M, C, B, x, xd = lag_eq(ratio)
    x_G1, y_G1, x_G2, y_G2, x_G3, y_G3, x_G4, y_G4, x_G5, y_G5, x_B, y_B, x_C, y_C, x_D, y_D, x_E, y_E, x_F, y_F = x
    x_G1_d, y_G1_d, x_G2_d, y_G2_d, x_G3_d, y_G3_d, x_G4_d, y_G4_d, x_G5_d, y_G5_d, x_B_d, y_B_d, x_C_d, y_C_d, x_D_d, y_D_d, x_E_d, y_E_d, x_F_d, y_F_d = xd 
    
    # Set up the problem
    opti = ca.Opti()

    # Declare the decision variables, Q = [q, q_d] = coordinates, U = input torques
    Q = opti.variable(10, N+1) 
    U = opti.variable(4, N) 
    
    # Set the objective
    E = 1/2*sum([U[0, i]**2 + U[1, i]**2 + U[2, i]**2 + U[3, i]**2 for i in range(N)])
    opti.minimize(E)

    # Specify the dynamics 
    def f(q,u):
        return ca.vertcat(q[5],q[6],q[7],q[8],q[9],ca.solve(M(q[:5]), B@u + C(q[:5],q[5:])))

    # Create gap closing constraints
    dt = T / N
    for k in range(N):
        # Runge-Kutta 4 integration
        k1 = f(Q[:,k], U[:,k])
        k2 = f(Q[:,k] + dt/2*k1, U[:,k])
        k3 = f(Q[:,k] + dt/2*k2, U[:,k])
        k4 = f(Q[:,k] + dt*k3, U[:,k])
        x_next = Q[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)

        # Euler integration (faster but less accurate)
        # x_next = Q[:,k] + dt*f(Q[:,k], U[:,k])

        opti.subject_to(Q[:,k+1] == x_next)

    ### Boundary conditions ###

    # BC1: position and velocity of foot at start and end of the trajectory
    opti.subject_to(x_E(Q[:5,0]) == - L)
    opti.subject_to(y_E(Q[:5,0]) == 0.0)
    opti.subject_to(x_E(Q[:5,-1]) == L)
    opti.subject_to(y_E(Q[:5,-1]) == 0.0)
    opti.subject_to(y_E_d(Q[:5,0], Q[5:,0]) > 0.0)
    opti.subject_to(y_E_d(Q[:5,-1], Q[5:,-1]) < 0.0)

    # BC2: Result of heelstrike at the end equals the start of the step
    opti.subject_to(Q[:, 0] == heelstrike_casadi(M, Q[:5, -1], Q[5:, -1]))

    # # BC3: postion of all joints above ground at all times
    opti.subject_to(y_E(Q[:5, :]) > 0.0)
    opti.subject_to(y_B(Q[:5, :]) > 0.0)
    opti.subject_to(y_C(Q[:5, :]) > 0.0)
    opti.subject_to(y_D(Q[:5, :]) > 0.0)
    opti.subject_to(y_F(Q[:5, :]) > 0.0)

    # # BC4: no overextended knees (Q3 and Q4)
    opti.subject_to(Q[2, :] <= 0.0)
    opti.subject_to(Q[3, :] <= 0.0)
    # opti.subject_to(Q[2, :] >= -np.pi/2)
    # opti.subject_to(Q[3, :] >= -np.pi/2)

    # # BC5: do not hit stond with any joint
    # opti.subject_to((x_E(Q[:5,:]) - 0.3)**2 + y_E(Q[:5,:])**2 > 0.2**2)
    # opti.subject_to((x_B(Q[:5,:]) - 0.3)**2 + y_B(Q[:5,:])**2 > 0.2**2)
    # opti.subject_to((x_C(Q[:5,:]) - 0.3)**2 + y_C(Q[:5,:])**2 > 0.2**2)
    # opti.subject_to((x_D(Q[:5,:]) - 0.3)**2 + y_D(Q[:5,:])**2 > 0.2**2)
    # opti.subject_to((x_F(Q[:5,:]) - 0.3)**2 + y_F(Q[:5,:])**2 > 0.2**2)
   

    ### initial guess ##
    if initial_guess is None:
        initial_guess = np.ones(10) 
        # initial_guess[4] = 0.1

    opti.set_initial(Q[0, :], initial_guess[0] )
    opti.set_initial(Q[1, :], initial_guess[1] )
    opti.set_initial(Q[2, :], initial_guess[2] )
    opti.set_initial(Q[3, :], initial_guess[3] )
    opti.set_initial(Q[4, :], initial_guess[4] )
    opti.set_initial(Q[5, :], initial_guess[5] )
    opti.set_initial(Q[6, :], initial_guess[6] )
    opti.set_initial(Q[7, :], initial_guess[7] )
    opti.set_initial(Q[8, :], initial_guess[8] )
    opti.set_initial(Q[9, :], initial_guess[9] )

    ### solve ###
    opti.solver('ipopt')
    sol = opti.solve()

    return sol.value(Q), sol.value(U), sol.value(E)







if __name__ == "__main__":
    # parameters
    N = 100
    L = 0.79
    T = 0.5
    ratio = 0.5
    # initial_guess = np.load('gait_optimization_results/guessQ.npy')
    initial_guess = None
    filename = 'no_overextended_knees'

    # run optimizer
    # Q, U, E = optimizer(N, L, T, ratio, initial_guess)

    # save values for Q, U and E in gait_optimization_results
    # np.save('gait_optimization_results/' + filename + '_Q' + '.npy', Q)
    # np.save('gait_optimization_results/' + filename + '_U' + '.npy', U)
    # np.save('gait_optimization_results/' + filename + '_E' + '.npy', E)

    Q = np.load('gait_optimization_results/' + filename + '_Q' + '.npy')
    U = np.load('gait_optimization_results/' + filename + '_U' + '.npy')
    E = np.load('gait_optimization_results/' + filename + '_E' + '.npy')

    ### print energy ###
    print(f" \n The total energy is Energy: {E} \n\n")

    ### make animation ###
    _, _, _, _, _, x, xd = lag_eq(ratio)
    x_G1, y_G1, x_G2, y_G2, x_G3, y_G3, x_G4, y_G4, x_G5, y_G5, x_B, y_B, x_C, y_C, x_D, y_D, x_E, y_E, x_F, y_F = x
    x_G1_d, y_G1_d, x_G2_d, y_G2_d, x_G3_d, y_G3_d, x_G4_d, y_G4_d, x_G5_d, y_G5_d, x_B_d, y_B_d, x_C_d, y_C_d, x_D_d, y_D_d, x_E_d, y_E_d, x_F_d, y_F_d = xd 

    t = np.linspace(0, T, len(Q.T))

    x_B_list = [float(x_B(q)) for q in Q[:5,:].T]
    y_B_list = [float(y_B(q)) for q in Q[:5,:].T]
    x_C_list = [float(x_C(q)) for q in Q[:5,:].T]
    y_C_list = [float(y_C(q)) for q in Q[:5,:].T]
    x_D_list = [float(x_D(q)) for q in Q[:5,:].T]
    y_D_list = [float(y_D(q)) for q in Q[:5,:].T]
    x_E_list = [float(x_E(q)) for q in Q[:5,:].T]
    y_E_list = [float(y_E(q)) for q in Q[:5,:].T]
    x_F_list = [float(x_F(q)) for q in Q[:5,:].T]
    y_F_list = [float(y_F(q)) for q in Q[:5,:].T]

    fig, ax = plt.subplots(1, 1)

    # circle = patches.Circle((0.3, 0), 0.2, fill=True, color='black')
    # ax.add_patch(circle)

    ax.set_aspect('equal')

    ln1, = plt.plot([], [], 'ro--', lw=3, markersize=8)
    ln2, = plt.plot([], [], 'ro--', lw=3, markersize=8)


    def animate(i):
        ln1.set_data([0.0, x_B_list[i], x_C_list[i], x_F_list[i]], [0.0, y_B_list[i], y_C_list[i], y_F_list[i]])
        ln2.set_data([x_C_list[i], x_D_list[i], x_E_list[i]], [y_C_list[i], y_D_list[i], y_E_list[i]])
        return ln1, ln2 

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ani = animation.FuncAnimation(fig, animate, frames = len(t), interval=50)

    # add a line at y = 0 (ground)
    ax.axhline(0, color='black', lw=1)

    # save animation in a map named gifs
    print(f"gait_optimization_gifs/ratio_{str(ratio)[2:]}_{filename}.gif")
    ani.save(f"gait_optimization_gifs/ratio_{str(ratio)[2:]}_{filename}.gif", writer='pillow', fps=25)
    plt.show()

    ### plot input torques ###
    t = np.linspace(0, T, N+1)
    plt.plot(t[:-1], U[0,:], label='U1')
    plt.plot(t[:-1], U[1,:], label='U2')
    plt.plot(t[:-1], U[2,:], label='U3')
    plt.plot(t[:-1], U[3,:], label='U4')
    plt.xlabel('Time [s]')
    plt.ylabel('Input torques [Nm]')
    plt.title(f'Input Torques for Initial Problem Statement')
    plt.legend()
    plt.show()


