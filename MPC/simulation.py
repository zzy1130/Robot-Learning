import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from mpc_controller import DMCController

class StableGimbalSim:
    """
    Simulation model of a closed-loop camera gimbal position servo axis.
    It is modeled as a stable second-order system with input delay:
    G(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2) * e^(-L * s)
    
    State equations:
    x1_dot = x2
    x2_dot = -2*zeta*omega_n*x2 - omega_n^2*x1 + omega_n^2*u(t - L)
    
    where:
    x1 : angular position (rad)
    x2 : angular velocity (rad/s)
    u  : target position command (rad)
    """
    def __init__(self, omega_n=15.0, zeta=0.8, delay_sec=0.04, dt=0.01):
        self.omega_n2 = omega_n**2
        self.two_zeta_omega = 2.0 * zeta * omega_n
        self.dt = dt
        
        # Calculate delay in terms of discrete steps
        self.delay_steps = int(round(delay_sec / dt))
        self.delay_queue = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)
        
        self.x = np.zeros(2)  # [position, velocity]
        
    def step(self, u):
        """
        Step simulation forward using RK4.
        """
        # Append new control input to the queue and retrieve the delayed control input
        if self.delay_steps > 0:
            self.delay_queue.append(u)
            u_delayed = self.delay_queue[0]
        else:
            u_delayed = u
            
        def dynamics(state, ctrl):
            pos, vel = state[0], state[1]
            dpos = vel
            dvel = -self.two_zeta_omega * vel - self.omega_n2 * pos + self.omega_n2 * ctrl
            return np.array([dpos, dvel])
            
        # RK4 Integration
        k1 = dynamics(self.x, u_delayed)
        k2 = dynamics(self.x + 0.5 * self.dt * k1, u_delayed)
        k3 = dynamics(self.x + 0.5 * self.dt * k2, u_delayed)
        k4 = dynamics(self.x + self.dt * k3, u_delayed)
        
        self.x += (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self.x[0]  # Return current position (y)
        
    def reset(self):
        self.x = np.zeros(2)
        if self.delay_steps > 0:
            self.delay_queue = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)

class PIDController:
    """
    Standard PID Controller with clamping anti-windup.
    """
    def __init__(self, kp, ki, kd, u_min=-12.0, u_max=12.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False
        
    def update(self, error, dt):
        if dt <= 0:
            return 0.0
            
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        if not self.initialized:
            d_term = 0.0
            self.initialized = True
        else:
            d_term = self.kd * (error - self.prev_error) / dt
            
        self.prev_error = error
        
        # Control output
        u = p_term + i_term + d_term
        
        # Clamp output (anti-windup)
        u_clamped = np.clip(u, self.u_min, self.u_max)
        if u != u_clamped:
            # Limit/reset integral to prevent windup
            self.integral -= error * dt
            
        return u_clamped
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False


def generate_step_response(plant, P):
    """
    Perform an open-loop step response test on the plant to obtain DMC coefficients.
    """
    plant.reset()
    step_response = []
    # Apply a unit step input u = 1.0
    for _ in range(P):
        y = plant.step(1.0)
        step_response.append(y)
    return step_response


def main():
    # Simulation parameters
    dt = 0.01  # 100 Hz control loop
    t_max = 8.0
    time_steps = np.arange(0, t_max, dt)
    
    # 1. Initialize the Plant (gimbal position servo loop with 0.04s delay)
    omega_n = 15.0
    zeta = 0.8
    delay_sec = 0.04
    
    plant_for_coefficients = StableGimbalSim(omega_n=omega_n, zeta=zeta, delay_sec=delay_sec, dt=dt)
    
    # 2. DMC prediction parameters
    P = 45  # Prediction horizon (0.45s) covering the plant settling time and 0.04s delay
    M = 5   # Control horizon
    alpha = 0.7  # Smoothing factor
    
    # Generate step response coefficients
    step_response = generate_step_response(plant_for_coefficients, P)
    print(f"Generated step response coefficients (first 5): {step_response[:5]}")
    
    # Define reference target trajectory (step changes)
    def get_target(t):
        if t < 1.0:
            return 0.0
        elif t < 4.0:
            return 1.0
        else:
            return -0.5
            
    # Initialize simulation logs
    history_t = []
    history_ref = []
    
    history_y_dmc = []
    history_u_dmc = []
    
    history_y_pid = []
    history_u_pid = []
    
    # ----------------------------------------------------
    # Simulation 1: DMC Controller
    # ----------------------------------------------------
    plant_dmc = StableGimbalSim(omega_n=omega_n, zeta=zeta, delay_sec=delay_sec, dt=dt)
    dmc = DMCController(
        step_response=step_response,
        P=P,
        M=M,
        alpha=alpha,
        Q_weight=1.0,
        R_weight=0.1
    )
    
    for t in time_steps:
        ref = get_target(t)
        y = plant_dmc.x[0]  # Current state position
        
        # Update controller
        u = dmc.update(y, ref, dt)
        # Limit control input (gimbal command limits, e.g. -12 to 12 rad)
        u = np.clip(u, -12.0, 12.0)
        # Keep controller's internal u in sync with clamp
        dmc.u = u
        
        # Step plant
        plant_dmc.step(u)
        
        # Save logs
        history_t.append(t)
        history_ref.append(ref)
        history_y_dmc.append(y)
        history_u_dmc.append(u)
        
    # ----------------------------------------------------
    # Simulation 2: Classical PID Controller
    # ----------------------------------------------------
    plant_pid = StableGimbalSim(omega_n=omega_n, zeta=zeta, delay_sec=delay_sec, dt=dt)
    
    # PID tuned for stability under 0.04s delay.
    # Higher gains cause oscillations due to delay, showing typical lag limitation.
    pid = PIDController(kp=2.0, ki=1.0, kd=0.15, u_min=-12.0, u_max=12.0)
    
    for t in time_steps:
        ref = get_target(t)
        y = plant_pid.x[0]
        
        # Compute control
        error = ref - y
        u = pid.update(error, dt)
        
        # Step plant
        plant_pid.step(u)
        
        # Save logs
        history_y_pid.append(y)
        history_u_pid.append(u)
        
    # ----------------------------------------------------
    # Plotting results
    # ----------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 1: Trajectory Tracking
    axs[0].plot(time_steps, history_ref, 'k--', label='Desired Target')
    axs[0].plot(time_steps, history_y_dmc, 'g-', label='DMC (Model Predictive)')
    axs[0].plot(time_steps, history_y_pid, 'r-', label='PID (Feedback Only)')
    axs[0].set_ylabel('Position (rad)')
    axs[0].set_title('Gimbal Position Trajectory Tracking (Input Delay = 0.04s)')
    axs[0].grid(True)
    axs[0].legend(loc='lower right')
    
    # Subplot 2: Tracking Error
    error_dmc = np.array(history_ref) - np.array(history_y_dmc)
    error_pid = np.array(history_ref) - np.array(history_y_pid)
    axs[1].plot(time_steps, error_dmc, 'g-', label='DMC Error')
    axs[1].plot(time_steps, error_pid, 'r-', label='PID Error')
    axs[1].set_ylabel('Error (rad)')
    axs[1].set_title('Position Tracking Error')
    axs[1].grid(True)
    axs[1].legend(loc='lower right')
    
    # Subplot 3: Control Input (Voltage/Command)
    axs[2].plot(time_steps, history_u_dmc, 'g-', label='DMC Input u(t)')
    axs[2].plot(time_steps, history_u_pid, 'r-', label='PID Input u(t)')
    axs[2].set_ylabel('Control Input (rad)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_title('Control Input Effort')
    axs[2].grid(True)
    axs[2].legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('/Users/zhongzhiyi/Robot-Learning/MPC/trajectory_comparison.png', dpi=150)
    plt.close()
    print("Simulation complete. Plot saved as trajectory_comparison.png")

if __name__ == '__main__':
    main()
