import numpy as np
import matplotlib.pyplot as plt
from pid_controller import TwoLinkRobot, PIDController, ComputedTorqueController

def reference_trajectory(t):
    """
    Generate desired position, velocity, and acceleration at time t.
    q_d = [sin(t), cos(2t)]^T
    """
    q_d = np.array([np.sin(t), np.cos(2 * t)])
    dq_d = np.array([np.cos(t), -2 * np.sin(2 * t)])
    ddq_d = np.array([-np.sin(t), -4 * np.cos(2 * t)])
    return q_d, dq_d, ddq_d

def run_simulation():
    # Simulation Parameters
    dt = 0.002  # Time step (s) - 500 Hz control loop
    t_max = 10.0
    time_steps = np.arange(0, t_max, dt)
    
    # ----------------------------------------------------
    # 1. Independent Joint PID Control Simulation
    # ----------------------------------------------------
    robot_pid = TwoLinkRobot()
    
    # Initial states (slightly off from desired starting point [0, 1] to test recovery)
    q_pid = np.array([0.2, 0.8])
    dq_pid = np.array([0.0, 0.0])
    
    # Independent joint PID controllers
    # Joint 1 has higher inertia, needs larger gains
    pid_joint1 = PIDController(kp=150.0, ki=20.0, kd=30.0, windup_limit=50.0)
    pid_joint2 = PIDController(kp=80.0, ki=15.0, kd=15.0, windup_limit=30.0)
    
    # Data storage
    history_time = []
    history_q_d = []
    history_q_pid = []
    history_tau_pid = []
    
    for t in time_steps:
        q_d, dq_d, ddq_d = reference_trajectory(t)
        
        # Position error
        e = q_d - q_pid
        
        # Compute joint control torques
        tau1 = pid_joint1.update(e[0], dt)
        tau2 = pid_joint2.update(e[1], dt)
        tau = np.array([tau1, tau2])
        
        # Record
        history_time.append(t)
        history_q_d.append(q_d.copy())
        history_q_pid.append(q_pid.copy())
        history_tau_pid.append(tau.copy())
        
        # Step simulation forward
        q_pid, dq_pid = robot_pid.rk4_step(q_pid, dq_pid, tau, dt)
        
    # Convert lists to arrays
    history_q_d = np.array(history_q_d)
    history_q_pid = np.array(history_q_pid)
    history_tau_pid = np.array(history_tau_pid)
    
    # ----------------------------------------------------
    # 2. Computed Torque Control (CTC) Simulation
    # ----------------------------------------------------
    robot_ctc = TwoLinkRobot()
    
    # Initial states (same as PID simulation)
    q_ctc = np.array([0.2, 0.8])
    dq_ctc = np.array([0.0, 0.0])
    
    # Gain matrices for CTC (critically damped decoupled system)
    # ddq_err + Kd * dq_err + Kp * q_err = 0
    # For omega_n = 15, zeta = 1.0 (critical damping)
    # Kp = omega_n^2 = 225, Kd = 2 * zeta * omega_n = 30
    Kp = np.diag([225.0, 225.0])
    Kd = np.diag([30.0, 30.0])
    Ki = np.diag([10.0, 10.0])  # Small integral gain for robustness
    
    ctc_controller = ComputedTorqueController(robot_ctc, Kp, Kd, Ki)
    
    history_q_ctc = []
    history_tau_ctc = []
    
    for t in time_steps:
        q_d, dq_d, ddq_d = reference_trajectory(t)
        
        # CTC Torque computation
        tau = ctc_controller.update(q_ctc, dq_ctc, q_d, dq_d, ddq_d, dt)
        
        # Record
        history_q_ctc.append(q_ctc.copy())
        history_tau_ctc.append(tau.copy())
        
        # Step simulation forward
        q_ctc, dq_ctc = robot_ctc.rk4_step(q_ctc, dq_ctc, tau, dt)
        
    history_q_ctc = np.array(history_q_ctc)
    history_tau_ctc = np.array(history_tau_ctc)
    
    # ----------------------------------------------------
    # Plotting and Saving Results
    # ----------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    
    # Joint 1 Tracking
    axs[0, 0].plot(time_steps, history_q_d[:, 0], 'k--', label='Desired')
    axs[0, 0].plot(time_steps, history_q_pid[:, 0], 'r-', alpha=0.8, label='Independent Joint PID')
    axs[0, 0].plot(time_steps, history_q_ctc[:, 0], 'g-', alpha=0.8, label='Computed Torque Control (CTC)')
    axs[0, 0].set_title('Joint 1 Position Tracking')
    axs[0, 0].set_ylabel('Position (rad)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    
    # Joint 2 Tracking
    axs[0, 1].plot(time_steps, history_q_d[:, 1], 'k--', label='Desired')
    axs[0, 1].plot(time_steps, history_q_pid[:, 1], 'r-', alpha=0.8, label='Independent Joint PID')
    axs[0, 1].plot(time_steps, history_q_ctc[:, 1], 'g-', alpha=0.8, label='Computed Torque Control (CTC)')
    axs[0, 1].set_title('Joint 2 Position Tracking')
    axs[0, 1].set_ylabel('Position (rad)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    
    # Joint 1 Error
    axs[1, 0].plot(time_steps, history_q_d[:, 0] - history_q_pid[:, 0], 'r-', label='PID Error')
    axs[1, 0].plot(time_steps, history_q_d[:, 0] - history_q_ctc[:, 0], 'g-', label='CTC Error')
    axs[1, 0].set_title('Joint 1 Position Tracking Error')
    axs[1, 0].set_ylabel('Error (rad)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    
    # Joint 2 Error
    axs[1, 1].plot(time_steps, history_q_d[:, 1] - history_q_pid[:, 1], 'r-', label='PID Error')
    axs[1, 1].plot(time_steps, history_q_d[:, 1] - history_q_ctc[:, 1], 'g-', label='CTC Error')
    axs[1, 1].set_title('Joint 2 Position Tracking Error')
    axs[1, 1].set_ylabel('Error (rad)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    
    # Joint 1 Torque
    axs[2, 0].plot(time_steps, history_tau_pid[:, 0], 'r-', alpha=0.7, label='PID Torque')
    axs[2, 0].plot(time_steps, history_tau_ctc[:, 0], 'g-', alpha=0.7, label='CTC Torque')
    axs[2, 0].set_title('Joint 1 Control Torque')
    axs[2, 0].set_xlabel('Time (s)')
    axs[2, 0].set_ylabel('Torque (N*m)')
    axs[2, 0].grid(True)
    axs[2, 0].legend()
    
    # Joint 2 Torque
    axs[2, 1].plot(time_steps, history_tau_pid[:, 1], 'r-', alpha=0.7, label='PID Torque')
    axs[2, 1].plot(time_steps, history_tau_ctc[:, 1], 'g-', alpha=0.7, label='CTC Torque')
    axs[2, 1].set_title('Joint 2 Control Torque')
    axs[2, 1].set_xlabel('Time (s)')
    axs[2, 1].set_ylabel('Torque (N*m)')
    axs[2, 1].grid(True)
    axs[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150)
    plt.close()
    print("Simulation completed. Plot saved to trajectory_comparison.png")

if __name__ == '__main__':
    run_simulation()
