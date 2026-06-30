import numpy as np

class CartPoleEnv:
    """
    极简手写 Cart-Pole 物理仿真器
    状态空间: [x, x_dot, theta, theta_dot]
    - x: 小车位置
    - x_dot: 小车速度
    - theta: 摆杆角度 (0 为垂直向上，单位：弧度)
    - theta_dot: 摆杆角速度
    动作空间: 0 (向左推 10N), 1 (向右推 10N)
    """
    def __init__(self):
        # 物理常数
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # 摆杆半长 (COM to joint)
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # 时间步长 dt = 0.02s

        # 终止条件
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * np.pi / 180  # 12度

        # 状态初始化
        self.state = None
        self.steps_beyond_terminated = None
        self.max_steps = 500
        self.step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # 初始化状态加上微小的随机噪声，防止对称性
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.step_count = 0
        self.steps_beyond_terminated = None
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        assert action in [0, 1], f"Invalid action: {action}"
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 动力学计算 (标准 Cart-Pole 方程)
        temp = (force + self.polemass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # 状态更新 (欧拉积分)
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        self.state = (x, x_dot, theta, theta_dot)
        self.step_count += 1

        # 终止条件判断
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        truncated = bool(self.step_count >= self.max_steps)

        # 奖励设计：立直每一步给 1.0 的奖励
        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # 刚出界时的奖励
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, terminated, truncated
