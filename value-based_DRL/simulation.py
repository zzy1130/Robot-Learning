import numpy as np
import torch
import matplotlib.pyplot as plt
from cartpole_env import CartPoleEnv
from dqn_agent import DQNAgent

# 设置随机种子，保证可复现性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_agent(agent_type="D3QN", num_episodes=200, seed=42):
    set_seed(seed)
    env = CartPoleEnv()
    
    # 动作与状态维度
    state_dim = 4
    action_dim = 2
    
    # 初始化智能体
    if agent_type == "DQN":
        agent = DQNAgent(
            state_dim, action_dim, lr=1e-3, gamma=0.99,
            double_dqn=False, dueling_dqn=False, tau=0.005
        )
    elif agent_type == "D3QN":
        agent = DQNAgent(
            state_dim, action_dim, lr=1e-3, gamma=0.99,
            double_dqn=True, dueling_dqn=True, tau=0.005
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # 探索率衰减参数
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.95  # 乘性衰减
    epsilon = epsilon_start

    rewards_history = []
    avg_q_estimations = []
    
    print(f"\n=================== 开始训练 {agent_type} 智能体 ===================")
    
    for episode in range(num_episodes):
        state = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False
        
        q_est_sum = 0
        update_count = 0
        
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, float(terminated))
            state = next_state
            episode_reward += reward
            
            # 进行网络参数更新
            loss, avg_q_batch = agent.update()
            if avg_q_batch is not None:
                q_est_sum += avg_q_batch
                update_count += 1
                
        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        rewards_history.append(episode_reward)
        avg_q_val = (q_est_sum / update_count) if update_count > 0 else 0.0
        avg_q_estimations.append(avg_q_val)
        
        if (episode + 1) % 20 == 0:
            # 计算最近 20 个回合的平均奖励
            recent_avg_r = np.mean(rewards_history[-20:])
            print(f"Episode {episode+1:3d}/{num_episodes} | 最近平均得分: {recent_avg_r:5.1f} | 探索率: {epsilon:.3f} | 平均Q估计: {avg_q_val:5.2f}")

    return rewards_history, avg_q_estimations

def plot_and_save_comparison(dqn_r, dqn_q, d3qn_r, d3qn_q):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 绘制累积奖励曲线 (滑动窗口平滑处理)
    def smooth(data, window_size=10):
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed
    
    axes[0].plot(dqn_r, color='#ff7f0e', alpha=0.3, label='DQN Episode Score')
    axes[0].plot(smooth(dqn_r), color='#d62728', linewidth=2.5, label='DQN Smoothed (10 ep)')
    axes[0].plot(d3qn_r, color='#1f77b4', alpha=0.3, label='D3QN Episode Score')
    axes[0].plot(smooth(d3qn_r), color='#1f77b4', linewidth=2.5, label='D3QN Smoothed (10 ep)')
    
    axes[0].set_title("学习曲线对比 (Episode Reward Comparison)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("训练回合 (Episode)", fontsize=11)
    axes[0].set_ylabel("累积得分 (Reward)", fontsize=11)
    axes[0].legend(loc='lower right', frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. 绘制平均Q估计对比，以展示过估偏差 (Overestimation Bias)
    # 随着控制稳定，真实的累积回报上限约为 1 / (1 - gamma) = 1 / 0.01 = 100 左右 (实际得分上限 500)
    axes[1].plot(dqn_q, color='#d62728', linewidth=2, label='DQN Q-estimation')
    axes[1].plot(d3qn_q, color='#2ca02c', linewidth=2, label='D3QN (Double Dueling) Q-estimation')
    
    axes[1].set_title("Q值过估偏差对比 (Q-Value Overestimation Comparison)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("训练回合 (Episode)", fontsize=11)
    axes[1].set_ylabel("网络平均预测 Q 值", fontsize=11)
    axes[1].legend(loc='upper left', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("trajectory_comparison.png", dpi=150)
    plt.close()
    print("\n[成功] 训练完成！对比结果已保存至 'trajectory_comparison.png'")

if __name__ == "__main__":
    # 运行 DQN 和 D3QN 训练
    dqn_rewards, dqn_q_est = train_agent("DQN", num_episodes=200, seed=42)
    d3qn_rewards, d3qn_q_est = train_agent("D3QN", num_episodes=200, seed=42)
    
    # 绘制保存对比曲线
    plot_and_save_comparison(dqn_rewards, dqn_q_est, d3qn_rewards, d3qn_q_est)
