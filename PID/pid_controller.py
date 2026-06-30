import numpy as np

class PIDController:
    """
    A standard PID controller implementation with windup limit.
    """
    def __init__(self, kp, ki, kd, windup_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.windup_limit = windup_limit
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False

    def update(self, error, dt):
        """
        Compute control output.
        """
        if dt <= 0:
            return 0.0
            
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        if self.windup_limit is not None:
            self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        i_term = self.ki * self.integral
        
        # Derivative term
        if not self.initialized:
            d_term = 0.0
            self.initialized = True
        else:
            d_term = self.kd * (error - self.prev_error) / dt
            
        self.prev_error = error
        
        return p_term + i_term + d_term

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.initialized = False


class TwoLinkRobot:
    """
    Simulation model of a 2-DOF planar RR (Revolute-Revolute) robotic arm
    moving in a vertical plane (subject to gravity).
    """
    def __init__(self, m1=2.0, m2=1.5, l1=0.5, l2=0.4, g=9.81):
        self.m1 = m1  # Mass of link 1 (kg)
        self.m2 = m2  # Mass of link 2 (kg)
        self.l1 = l1  # Length of link 1 (m)
        self.l2 = l2  # Length of link 2 (m)
        self.g = g    # Acceleration due to gravity (m/s^2)
        
        # Center of mass (COM) located at the midpoint of each link
        self.lc1 = l1 / 2.0
        self.lc2 = l2 / 2.0
        
        # Moment of inertia about the center of mass for thin rods
        self.I1 = (1.0 / 12.0) * m1 * (l1 ** 2)
        self.I2 = (1.0 / 12.0) * m2 * (l2 ** 2)

    def get_dynamics_matrices(self, q, dq):
        """
        Calculate Mass matrix M(q), Coriolis matrix C(q, dq), and gravity vector g(q)
        for the joint states:
        q = [q1, q2]^T
        dq = [dq1, dq2]^T
        """
        q1, q2 = q[0], q[1]
        dq1, dq2 = dq[0], dq[1]
        
        # 1. Mass Matrix M(q)
        # M11 = m1*lc1^2 + I1 + m2*(l1^2 + lc2^2 + 2*l1*lc2*cos(q2)) + I2
        # M12 = M21 = m2*(lc2^2 + l1*lc2*cos(q2)) + I2
        # M22 = m2*lc2^2 + I2
        m11 = self.m1 * (self.lc1 ** 2) + self.I1 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2 + 2 * self.l1 * self.lc2 * np.cos(q2)) + self.I2
        m12 = self.m2 * (self.lc2 ** 2 + self.l1 * self.lc2 * np.cos(q2)) + self.I2
        m21 = m12
        m22 = self.m2 * (self.lc2 ** 2) + self.I2
        
        M = np.array([[m11, m12],
                      [m21, m22]])
        
        # 2. Coriolis/Centrifugal Matrix C(q, dq)
        # C11 = -m2*l1*lc2*sin(q2)*dq2
        # C12 = -m2*l1*lc2*sin(q2)*(dq1 + dq2)
        # C21 =  m2*l1*lc2*sin(q2)*dq1
        # C22 =  0
        h = self.m2 * self.l1 * self.lc2 * np.sin(q2)
        c11 = -h * dq2
        c12 = -h * (dq1 + dq2)
        c21 = h * dq1
        c22 = 0.0
        
        C = np.array([[c11, c12],
                      [c21, c22]])
        
        # 3. Gravity Vector G(q)
        # g1 = (m1*lc1 + m2*l1)*g*cos(q1) + m2*lc2*g*cos(q1 + q2)
        # g2 = m2*lc2*g*cos(q1 + q2)
        # Note: assuming standard vertical gravity pointing downward
        g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * self.g * np.cos(q1) + self.m2 * self.lc2 * self.g * np.cos(q1 + q2)
        g2 = self.m2 * self.lc2 * self.g * np.cos(q1 + q2)
        
        G = np.array([g1, g2])
        
        return M, C, G

    def forward_dynamics(self, q, dq, tau):
        """
        Compute joint accelerations ddq given positions, velocities, and torques.
        ddq = M^-1 * (tau - C*dq - G)
        """
        M, C, G = self.get_dynamics_matrices(q, dq)
        
        # Torque equation: M * ddq + C * dq + G = tau
        # => ddq = M^-1 * (tau - C * dq - G)
        rhs = tau - C.dot(dq) - G
        ddq = np.linalg.solve(M, rhs)
        return ddq

    def rk4_step(self, q, dq, tau, dt):
        """
        Numerical integration using Runge-Kutta 4th order.
        """
        # State vector x = [q, dq]
        # dx/dt = [dq, ddq]
        
        def state_deriv(q_curr, dq_curr):
            ddq_curr = self.forward_dynamics(q_curr, dq_curr, tau)
            return dq_curr, ddq_curr

        # k1
        dq_k1, ddq_k1 = state_deriv(q, dq)
        
        # k2
        q_k2 = q + 0.5 * dt * dq_k1
        dq_k2 = dq + 0.5 * dt * ddq_k1
        dq_k2_deriv, ddq_k2 = state_deriv(q_k2, dq_k2)
        
        # k3
        q_k3 = q + 0.5 * dt * dq_k2_deriv
        dq_k3 = dq + 0.5 * dt * ddq_k2
        dq_k3_deriv, ddq_k3 = state_deriv(q_k3, dq_k3)
        
        # k4
        q_k4 = q + dt * dq_k3_deriv
        dq_k4 = dq + dt * ddq_k3
        dq_k4_deriv, ddq_k4 = state_deriv(q_k4, dq_k4)
        
        # Update
        q_next = q + (dt / 6.0) * (dq_k1 + 2*dq_k2_deriv + 2*dq_k3_deriv + dq_k4_deriv)
        dq_next = dq + (dt / 6.0) * (ddq_k1 + 2*ddq_k2 + 2*ddq_k3 + ddq_k4)
        
        return q_next, dq_next


class ComputedTorqueController:
    """
    Computed Torque Controller (CTC) for a 2-DOF robotic manipulator.
    Compensates for full system dynamics and tracks trajectories using PID in outer loop.
    """
    def __init__(self, robot_model, kp_matrix, kd_matrix, ki_matrix=None):
        self.robot = robot_model
        self.Kp = np.array(kp_matrix)  # 2x2 matrix of proportional gains
        self.Kd = np.array(kd_matrix)  # 2x2 matrix of derivative gains
        
        if ki_matrix is not None:
            self.Ki = np.array(ki_matrix)
        else:
            self.Ki = np.zeros((2, 2))
            
        self.integral_error = np.zeros(2)

    def update(self, q, dq, q_d, dq_d, ddq_d, dt):
        """
        Compute control torques tau based on current state (q, dq)
        and desired trajectory state (q_d, dq_d, ddq_d).
        """
        # Tracking error
        e = q_d - q
        de = dq_d - dq
        
        # Update integral error
        self.integral_error += e * dt
        
        # Auxiliary control input: u = ddq_d + Kd * de + Kp * e + Ki * integral_error
        u = ddq_d + self.Kd.dot(de) + self.Kp.dot(e) + self.Ki.dot(self.integral_error)
        
        # Get dynamics matrices based on current state
        M, C, G = self.robot.get_dynamics_matrices(q, dq)
        
        # Computed Torque Control Law: tau = M * u + C * dq + G
        tau = M.dot(u) + C.dot(dq) + G
        
        return tau

    def reset(self):
        self.integral_error = np.zeros(2)
