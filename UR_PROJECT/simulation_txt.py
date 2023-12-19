import pybullet as pb
import time
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
import sim_utils
import ddp

def simulate():
    simDT = 0.005
    start_time = 0
    end_time = 3
    time_span = np.arange(start_time, end_time, simDT).flatten()

    robotID, robotModel = sim_utils.simulationSetup(simDT)
    N = np.shape(time_span)[0]

    init_state = np.array([-np.pi, 0.0, 0, 0])
    # target_state = np.array([0.0, 0.0, 0.0, 0.0])

    # np.random.seed(1)
    # initial_guess = 0.0 * np.random.normal(loc = 0, scale = 1, size = (np.shape(time_span)[0], 1))

    input("press ENTER to START the simulation:")

    jointIndices = range(2)
    for i in jointIndices:
        pb.resetJointState(robotID, i, init_state[i])

    states = np.loadtxt('States_ddp.csv', delimiter = ',')
    inputs = np.loadtxt('Inputs_ddp.csv', delimiter = ',')
    K_feedback = np.loadtxt('K_feedback_ddp.csv', delimiter = ',')
    
    u_cont = np.zeros(2)
    x_state = np.zeros((N, 4))

    u_cont_mat = np.zeros((N, 1))

    for i in range(N):
        
        #x_state = states[i, :]
        q, qdot = sim_utils.getState(robotID, jointIndices)
        x_state[i, :] = np.hstack((q, qdot))
        error_f = x_state[i, :] - states[i, :]
        # error_f[0] = ddp.normalize_angle(error_f[0])
        # error_f[1] = ddp.normalize_angle(error_f[1])

        #mol = K_feedback[i, :] @ x_state
        mol = K_feedback[i, :] @ error_f
        u_cont[1] = inputs[i] + mol
        u_cont_mat[i, :] = u_cont[1]

        pb.setJointMotorControlArray(robotID, jointIndices, controlMode = pb.TORQUE_CONTROL, forces = u_cont)
        pb.stepSimulation()
        time.sleep(simDT)
    
    plt.figure(figsize=(10, 10))
    plt.plot(inputs, color = 'red')
    plt.title("Input Plot")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()
    
    plt.figure(figsize=(10, 10))
    plt.plot(states[:, 0], color = 'red')
    plt.title("Q1 Plot")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(states[:, 1], color = 'green')
    plt.title("Q2 Plot")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(states[:, 2], color = 'green')
    plt.title("Q1dot Plot")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(states[:, 3], color = 'green')
    plt.title("Q2dot Plot")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

############## PLOT PYBULLET ##############

    plt.figure(figsize=(10, 10))
    plt.plot(u_cont_mat, color = 'red')
    plt.title("Input Plot DDP")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(x_state[:, 0], color = 'red')
    plt.title("Q1 Plot_x_state")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(x_state[:, 1], color = 'green')
    plt.title("Q2 Plot_x_state")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(x_state[:, 2], color = 'green')
    plt.title("Q1dot Plot_x_state")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.plot(x_state[:, 3], color = 'green')
    plt.title("Q2dot Plot_x_state")
    plt.xlabel("Time")
    plt.ylabel("u")
    plt.show()
    input("press ENTER to CLOSE the simulation:")
     
    pb.disconnect()
