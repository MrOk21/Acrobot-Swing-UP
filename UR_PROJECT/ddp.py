import numpy as np
import math
import matplotlib.pyplot as plt
import pinocchio as pin
import casadi as ca

simDT = 0.005
start_time = 0
end_time = 3
time_span = np.arange(start_time, end_time, simDT).flatten()
n_timestep = np.shape(time_span)[0]

n, m = (4, 1)
N = n_timestep
Q = 0.1 * np.eye(n)
Q[1,1] = 0.5
R = 100 * np.eye(m)
Qter = 1000 * np.eye(4)
Q_T = Qter
Q_c = Q
R_K = R

def rollout(robotModel, x_init, u, simDT):
    states = np.zeros((n_timestep + 1, 4))
    inputs = np.zeros((n_timestep, 1))
    current_state = x_init
    states[0, :] = current_state
    q = x_init[:2]
    dq = x_init[2:]
    current_input = u[0]
    u_temp = np.zeros(2)
    for i in range(0, n_timestep):
        u_temp[1] = u[i]
        ddq = pin.aba(robotModel.model, robotModel.data, q, dq, u_temp)
        q = q + dq * simDT

        dq = dq + ddq * simDT
        next_state = np.hstack((q, dq))

        current_input = u[i]
        states[i + 1, :] = next_state   
        inputs[i] = current_input
                
    return states, inputs

def compute_cost(n_timestep, states, inputs, x_des, Q_T, R_K):
    total_cost = 0.0
    Q_c = Q

    for ii in range(0, n_timestep):
        current_x = states[ii , :]
        current_u = inputs[ii]
        error = current_x - x_des
        error[0] = normalize_angle(error[0])
        error[1] = normalize_angle(error[1])
        current_cost = current_u.T @ R_K @ current_u + (error).T @ Q_c @ (error)
        total_cost = total_cost + current_cost

        terminal_difference = (x_des - states[-1, :]).flatten()
        terminal_difference[0] = normalize_angle(terminal_difference[0])
        terminal_difference[1] = normalize_angle(terminal_difference[1])
        terminal_cost = terminal_difference.T @ Q_T @ terminal_difference
        total_cost = total_cost + terminal_cost
    return total_cost

def normalize_angle(angle):
    c = math.cos(angle)
    s = math.sin(angle)
    angle = math.atan2(s,c) 
    return angle

def backward_pass(robotModel, x_des, states, inputs, Q_T, simDT):
    
    opt = ca.Opti()
    X = opt.variable(4)
    U = opt.variable(1)

    L_ = lambda error, u: (error).T @ Q @ (error) + u.T @ R @ u
    L_ter_ = lambda error: (error).T @ Qter @ (error)
    L_ = ca.Function('L', [X, U], [L_(X,U)])
    L_ter = ca.Function('L_ter', [X], [L_ter_(X)])
    Lx = ca.Function('Lx', [X, U], [ca.jacobian(L_(X,U), X)])
    Lu = ca.Function('Lu', [X, U], [ca.jacobian(L_(X,U), U)])
    Lxx = ca.Function('Lxx', [X, U], [ca.jacobian(ca.jacobian(L_(X,U), X), X)])
    Lux = ca.Function('Lux', [X, U], [ca.jacobian(ca.jacobian(L_(X,U), U), X)])
    Luu = ca.Function('Luu', [X, U], [ca.jacobian(ca.jacobian(L_(X,U), U), U)])
    L_terx = ca.Function('L_terx', [X], [ca.jacobian(L_ter_(X), X)])
    L_terxx = ca.Function('L_terxx', [X], [ca.jacobian(ca.jacobian(L_ter_(X), X), X)])

    end_difference = (states[-1, :] - x_des)
    end_difference[0] = normalize_angle(end_difference[0])
    end_difference[1] = normalize_angle(end_difference[1])
    end_difference = end_difference.flatten()
    

    V = np.zeros(N + 1)
    V[-1] = L_ter(end_difference)
    Vx = np.zeros((n, N + 1))
    Vx[:, -1] = L_terx(end_difference)
    Vxx = np.zeros((n, n, N + 1))
    Vxx[:, :, -1] = L_terxx(end_difference)

    # Compute Jacobian in symbolic way
    K_small = np.zeros((n_timestep, 1))
    K_big = np.zeros((n_timestep, 1, 4))

    # Initialize cost reduction to converge
    expected_cost_reduction = 0
    expected_cost_reduction_grad = 0
    expected_cost_reduction_hess = 0
    
    fx_left = np.zeros((2, 2))
    fx_right = np.eye(2)
    fx1 = np.hstack((fx_left, fx_right))
    input1 = np.zeros(2)

    # Backward Pass
    for i in reversed(range(n_timestep)):
        current_u = inputs[i]

        q = states[i, :2] 
        dq = states[i, 2:]
        input1[1] = inputs[i]
        pin.computeABADerivatives(robotModel.model, robotModel.data, q, dq, input1)
        ddq_dq = robotModel.data.ddq_dq 
        ddq_dv = robotModel.data.ddq_dv 
        ddq_dtau = robotModel.data.Minv 
    
        fx2 = np.hstack((ddq_dq, ddq_dv))
        ide = np.eye(4)
        fx = ide + simDT * np.vstack((fx1, fx2))
        fu = simDT * np.vstack(([0], [0], ddq_dtau[0, 1], ddq_dtau[1, 1]))
        
        error1 = (states[i, :] - x_des)
        error1[0] = normalize_angle(error1[0])
        error1[1] = normalize_angle(error1[1])

        Q_x = np.array(Lx(error1, inputs[i])).flatten() + fx.T @ Vx[:, i + 1]
        Q_u = np.array(Lu(error1, inputs[i])).flatten() + fu.T @ Vx[:, i + 1]

        Q_xx = Lxx(error1, inputs[i]) + fx.T @ Vxx[:, :, i + 1] @ fx
        Q_ux = Lux(error1, inputs[i]) + fu.T @ (Vxx[:, :, i + 1] ) @ fx
        Q_uu = Luu(error1, inputs[i]) + fu.T @ (Vxx[:, :, i + 1] ) @ fu
        
        # Compute gains
        Q_uu_inv = np.linalg.inv(Q_uu)
        k = - Q_uu_inv @ Q_u
        K = - Q_uu_inv @ Q_ux
        
        K_small[i, :] = k 
        K_big[i, :, :] = K

        # Update the expected reduction
        current_cost_reduction_grad = -Q_u.T @ k
        current_cost_reduction_hess = 0.5 * k.T @ Q_uu @ k
        current_cost_reduction = current_cost_reduction_grad + current_cost_reduction_hess
        
        expected_cost_reduction_grad +=  current_cost_reduction_grad
        expected_cost_reduction_hess +=  current_cost_reduction_hess
        expected_cost_reduction +=  current_cost_reduction

        V[i] = V[i + 1] - 0.5 * k.T @ Q_uu @ k
        Vx[:, i] = np.array(Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k).flatten()
        Vxx[:, :, i] = (Q_xx + Q_ux.T @ K + K.T @ Q_ux + K.T @ Q_uu @ K)

    expected_cost_reduction_grad_ = expected_cost_reduction_grad
    expected_cost_reduction_hess_ = expected_cost_reduction_hess
    expected_cost_reduction_ = expected_cost_reduction

    # Store gain schedule
    K_feedforward = K_small
    K_feedback = K_big
    return (K_feedforward, K_feedback, expected_cost_reduction_, expected_cost_reduction_grad_, expected_cost_reduction_hess_)

def forward_pass(robotModel, x_init, states, inputs, K_feedforward, K_feedback, alpha, simDT):
        x_new = np.zeros((n_timestep + 1, 4))
        u_new = np.zeros((n_timestep, 1))
        current_state = x_init
        x_new[0, :] = current_state
        x_new[1, :] = current_state
        
        q = current_state[:2] 
        dq = current_state[2:]
        u_new_aba = np.zeros(2)

        for t in range(0, n_timestep):
            curr_feedforward = alpha * K_feedforward[t, :] 
            error_fp = current_state - states[t, :]
            error_fp[0] = normalize_angle(error_fp[0])
            error_fp[1] = normalize_angle(error_fp[1])
            curr_feedback = K_feedback[t, :, :] @ (error_fp)
            curr_input = inputs[t] + np.hstack((curr_feedback + curr_feedforward))
            u_new[t] = curr_input
            u_new_aba[1] = u_new[t]

            ddq = pin.aba(robotModel.model, robotModel.data, q, dq, u_new_aba)
            q = q + (dq * simDT) 
            dq = dq + (ddq * simDT)
            next_state = np.hstack((q, dq))
            
            x_new[t + 1, :] = next_state
            current_state = next_state
            
            q = current_state[:2] 
            dq = current_state[2:]
        return (x_new, u_new)

def ddp(robotModel, u, x_init, x_des, simDT):    
    states, inputs = rollout(robotModel, x_init, u, simDT)
    current_cost = compute_cost(n_timestep, states, inputs, x_des, Q_T, R_K)

    learning_speed = 0.95
    low_learning_rate = 0.05
    low_expected_reduction = 1e-2
    c = 0.1
    iterations = 3000

    for i in range(0, iterations):
        print('Starting iteration: ', i)

        (K_feedforward, K_feedback, expected_reduction, expected_cost_reduction_grad_, expected_cost_reduction_hess_) = backward_pass(robotModel, x_des, states, inputs, Q_T, simDT)

        if(abs(expected_reduction) < low_expected_reduction):
        # If the expected reduction is low, then end the optimization
            print("Stopping optimization, optimal trajectory")
            break
            
        learning_rate = 0.7
        c_flag = 0

        while(learning_rate > 0.05 and c_flag == 0):
            (new_states, new_inputs) = forward_pass(robotModel, x_init, states, inputs, K_feedforward, K_feedback, learning_rate, simDT)
            new_cost = compute_cost(n_timestep, new_states, new_inputs, x_des, Q_T, R_K)
            
            cost_difference = (current_cost - new_cost)
            expected_cost_redu = ((learning_rate * expected_cost_reduction_grad_) + (learning_rate * learning_rate) * expected_cost_reduction_hess_)
            c_flag = cost_difference/expected_cost_redu > c

            if(c_flag == 1):
                current_cost = new_cost
                states = new_states
                inputs = new_inputs
                print(":)")
                
            else:
                learning_rate = learning_speed * learning_rate

        if(learning_rate < low_learning_rate):
            print("Stopping optimization, low learning rate")
            break
    # Return the current trajectory
    
    np.savetxt('Inputs_ddp.csv', inputs, delimiter = ',')
    np.savetxt('States_ddp.csv', states, delimiter = ',')
    np.savetxt('K_feedback.csv', states, delimiter = ',')

    return states, inputs, K_feedforward, K_feedback, current_cost
