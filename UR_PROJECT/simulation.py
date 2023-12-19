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
    target_state = np.array([0.0, 0.0, 0.0, 0.0])

    np.random.seed(1)
    initial_guess = 0.0 * np.random.normal(loc = 0, scale = 1, size = (np.shape(time_span)[0], 1))
    # initial_guess[:, 0] = 0.1 * np.sin(2 * np.pi * time_span) 

    input("press ENTER to START the simulation:")

    jointIndices = range(2)
    for i in jointIndices:
        pb.resetJointState(robotID, i, init_state[i])
    
    ddp_ = ddp.ddp(robotModel, initial_guess, init_state, target_state, simDT)
    
    (states, inputs, k_feedforward, K_feedback, current_cost) = ddp_
    u_cont = np.zeros(2)
    x_state = np.zeros((N, 4))

    u_cont_mat = np.zeros((N, 1))

    for i in range(N):
        
        #x_state = states[i, :]
        q, qdot = sim_utils.getState(robotID, jointIndices)
        x_state[i, :] = np.hstack((q, qdot))
        error_f = x_state[i, :] - states[i, :]

        #mol = K_feedback[i, :] @ x_state
        mol = K_feedback[i, :] @ error_f
        u_cont[1] = inputs[i] + mol
        u_cont_mat[i, :] = u_cont[1]

        pb.setJointMotorControlArray(robotID, jointIndices, controlMode = pb.TORQUE_CONTROL, forces = u_cont)
        pb.stepSimulation()
        time.sleep(simDT)


    ####### PLOT #####
    
    # Inputs DDP and PyBullet
    plt.figure(figsize = (10, 10))
    plt.plot(inputs, color = 'blue', label = 'Inputs DDP')
    plt.plot(u_cont_mat, color = 'green', label = 'Inputs PyBullet')
    plt.title("Inputs (DDP) ( With K_feedback)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Torque [Nm]")
    plt.legend()
    plt.grid()
    plt.savefig('Inputs DDP and PyBullet ( With K_feedback).png')
    plt.show()
    
    # Q1 e Q2 DDP
    plt.figure(figsize = (10, 10))
    plt.plot(states[:, 0], color = 'blue', label = 'Q1')
    plt.plot(states[:, 1], color = 'green', label = 'Q2')
    plt.title("Q1 and Q2 DDP ( With K_feedback)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Radians [rad]")
    plt.legend()
    plt.grid()
    plt.savefig('Q1 and Q2 DDP ( With K_feedback).png')
    plt.show()
    
    # Q1_DOT Q2_DOT DDP
    plt.figure(figsize = (10, 10))
    plt.plot(states[:, 2], color = 'blue', label = 'Q1_DOT')
    plt.plot(states[:, 3], color = 'green', label = 'Q2_DOT')
    plt.title("Q1_DOT and Q2_DOT DDP ( With K_feedback)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Angular Velocity [rad/s]")
    plt.legend()
    plt.grid()
    plt.savefig('Q1_DOT and Q2_DOT DDP ( With K_feedback).png')
    plt.show()

    # Q1 e Q2 PyBullet
    plt.figure(figsize = (10, 10))
    plt.plot(x_state[:, 0], color = 'blue', label = 'Q1')
    plt.plot(x_state[:, 1], color = 'green', label = 'Q2')
    plt.title("Q1 and Q2 PyBullet ( With K_feedback)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Radians [rad]")
    plt.legend()
    plt.grid()
    plt.savefig('Q1 and Q2 PyBullet ( With K_feedback).png')
    plt.show()

    # Q1 DDP Q1 Pybullet
    plt.figure(figsize = (10, 10))
    plt.plot(states[:, 0], color = 'blue', label = 'Q1 DDP')
    plt.plot(x_state[:, 0], color = 'green', label = 'Q1 PyBullet')
    plt.title("Q1 DDP and Q1 PyBullet ( With K_feedback)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Radians [rad]")
    plt.legend()
    plt.grid()
    plt.savefig('Q1 DDP and Q1 PyBullet ( With K_feedback).png')
    plt.show()

    # Q2 DDP Q2 Pybullet
    plt.figure(figsize = (10, 10))
    plt.plot(states[:, 1], color = 'blue', label = 'Q2 DDP')
    plt.plot(x_state[:, 1], color = 'green', label = 'Q2 PyBullet')
    plt.title("Q2 DDP and Q2 PyBullet ( With K_feedback)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Radians [rad]")
    plt.legend()
    plt.grid()
    plt.savefig('Q2 DDP and Q2 PyBUllet ( With K_feedback).png')
    plt.show()
    
    input("press ENTER to CLOSE the simulation:")
    pb.disconnect()
