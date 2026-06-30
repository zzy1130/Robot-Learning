import numpy as np

class DMCController:
    """
    Dynamic Matrix Control (DMC) - A classic industrial Model Predictive Control (MPC) algorithm
    based on the step response model of the system.
    """
    def __init__(self, step_response, P, M, alpha, Q_weight=1.0, R_weight=1.0, h_vector=None):
        """
        Parameters:
        -----------
        step_response : list or np.ndarray
            Step response coefficients (a_1, a_2, ..., a_N). N should be >= P.
        P : int
            Prediction horizon.
        M : int
            Control horizon (M <= P).
        alpha : float
            Reference trajectory smoothing factor (0 <= alpha < 1).
            Smaller alpha means faster tracking but potentially higher overshoot.
        Q_weight : float or np.ndarray
            Error tracking weight. If float, Q = Q_weight * I.
        R_weight : float or np.ndarray
            Control increment penalty weight. If float, R = R_weight * I.
        h_vector : np.ndarray, optional
            Feedback correction vector of length P.
            If None, default to a constant vector of 0.5 for robust filtering.
        """
        self.P = P
        self.M = M
        self.alpha = alpha
        
        # Ensure step response coefficients has at least P elements
        self.a = np.array(step_response)[:P]
        if len(self.a) < P:
            raise ValueError(f"Step response coefficients length ({len(self.a)}) must be at least P ({P})")
            
        # 1. Build the Dynamic Matrix A (P x M)
        self.A = np.zeros((P, M))
        for i in range(P):
            for j in range(M):
                if i >= j:
                    self.A[i, j] = self.a[i - j]
                    
        # 2. Weight matrices Q (P x P) and R (M x M)
        if isinstance(Q_weight, (int, float)):
            self.Q = np.eye(P) * Q_weight
        else:
            self.Q = np.diag(Q_weight)
            
        if isinstance(R_weight, (int, float)):
            self.R = np.eye(M) * R_weight
        else:
            self.R = np.diag(R_weight)
            
        # 3. Shift Matrix S (P x P)
        self.S = np.zeros((P, P))
        for i in range(P - 1):
            self.S[i, i + 1] = 1.0
        self.S[P - 1, P - 1] = 1.0  # Keep the last prediction constant
        
        # 4. Correction vector h (P x 1)
        if h_vector is None:
            # Default to robust constant correction
            self.h = np.ones((P, 1)) * 0.5
        else:
            self.h = np.array(h_vector).reshape((P, 1))
            
        # 5. Pre-compute the optimization gain matrix:
        # du = (A^T * Q * A + R)^-1 * A^T * Q * (Y_ref - Y_0)
        # We can pre-compute: G_opt = (A^T * Q * A + R)^-1 * A^T * Q
        try:
            self.G_opt = np.linalg.inv(self.A.T.dot(self.Q).dot(self.A) + self.R).dot(self.A.T).dot(self.Q)
        except np.linalg.LinAlgError:
            raise ValueError("Optimization matrix is singular. Check R_weight and step response.")
            
        # Initialize state variables
        self.u = 0.0
        self.Y0 = np.zeros((P, 1))  # Predicted output sequence: [y(k+1|k-1), ..., y(k+P|k-1)]^T
        
    def update(self, y_now, target, dt):
        """
        Compute control output u(k) based on current output measurement y(k) and reference target.
        
        Parameters:
        -----------
        y_now : float
            Current measured output of the system: y(k).
        target : float
            Desired setpoint: w(k).
        dt : float
            Sample time.
            
        Returns:
        --------
        u : float
            Control signal to apply to the plant: u(k).
        """
        # 1. Compute Reference Trajectory (Y_ref: P x 1)
        # y_ref(k+i) = alpha * y_ref(k+i-1) + (1-alpha) * target
        # y_ref(k) = y_now
        Y_ref = np.zeros((self.P, 1))
        prev_ref = y_now
        for i in range(self.P):
            val = self.alpha * prev_ref + (1.0 - self.alpha) * target
            Y_ref[i, 0] = val
            prev_ref = val
            
        # 2. Feedback Correction
        # Compare actual output y_now with predicted output y0(k|k-1)
        # Note: self.Y0[0, 0] is the prediction of y(k) made at step k-1
        error_pred = y_now - self.Y0[0, 0]
        Y_cor = self.Y0 + self.h * error_pred
        
        # 3. Shift predictions forward in time
        Y0_shifted = self.S.dot(Y_cor)
        
        # 4. Solve Optimization Problem for control increment sequence: dU (M x 1)
        # dU = G_opt * (Y_ref - Y0_shifted)
        dU = self.G_opt.dot(Y_ref - Y0_shifted)
        
        # 5. Extract first control increment and update control action
        du_k = dU[0, 0]
        self.u += du_k
        
        # 6. Update predictions using only the executed control increment du_k
        self.Y0 = Y0_shifted + self.a.reshape((self.P, 1)) * du_k
        
        return self.u
        
    def reset(self):
        """
        Reset controller internal states.
        """
        self.u = 0.0
        self.Y0 = np.zeros((self.P, 1))
