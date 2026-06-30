import os
import matplotlib
matplotlib.use('Agg')  # 无界面模式
import matplotlib.pyplot as plt

def render_equation(latex_str, filename, fontsize=16, dpi=200):
    """
    使用 matplotlib 渲染 LaTeX 公式并保存为透明背景的 PNG
    """
    fig = plt.figure(figsize=(1, 1))  # 初始小尺寸，后续 tight_layout 会自动调整
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # 渲染公式，设置 horizontalalignment 和 verticalalignment
    text = ax.text(0.5, 0.5, f"${latex_str}$", 
                  fontsize=fontsize, 
                  ha='center', va='center', 
                  color='black')
    
    # 获取文字范围并调整画布大小
    fig.canvas.draw()
    bbox = text.get_window_extent(fig.canvas.get_renderer())
    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
    
    # 微调边距以防边缘裁剪
    pad = 0.1
    width = bbox_inches.width + pad * 2
    height = bbox_inches.height + pad * 2
    fig.set_size_inches(width, height)
    
    # 再次定位文字到新的中心
    text.set_position((0.5, 0.5))
    
    # 创建 images 目录如果它不存在
    os.makedirs("images", exist_ok=True)
    filepath = os.path.join("images", filename)
    
    # 保存为透明透明背景 PNG
    plt.savefig(filepath, dpi=dpi, transparent=True, bbox_inches='tight', pad_inches=pad)
    plt.close(fig)
    print(f"[成功] 渲染公式: {filename}")

if __name__ == "__main__":
    # 定义需要渲染的 LaTeX 公式
    equations = {
        "eq_bellman.png": r"Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q^*(s', a')",
        "eq_dqn_loss.png": r"L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]",
        "eq_ddqn_target.png": r"Y^{DoubleQ} = r + \gamma Q\left(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-\right)",
        "eq_dueling_q.png": r"Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)",
        "eq_cartpole_theta_acc.png": r"\ddot{\theta} = \frac{g \sin\theta + \cos\theta \left( \frac{-F - m l \dot{\theta}^2 \sin\theta}{M + m} \right)}{l \left( \frac{4}{3} - \frac{m \cos^2\theta}{M + m} \right)}",
        "eq_cartpole_x_acc.png": r"\ddot{x} = \frac{F + m l \left( \dot{\theta}^2 \sin\theta - \ddot{\theta} \cos\theta \right)}{M + m}",
        "eq_contraction.png": r"\Vert \mathcal{B}V - \mathcal{B}\bar{V} \Vert_{\infty} \leq \gamma \Vert V - \bar{V} \Vert_{\infty}",
        "eq_projection.png": r"\Pi V = \arg\min_{V' \in \Omega} \sum_{s} \Vert V'(s) - V(s) \Vert^2"
    }
    
    for filename, latex in equations.items():
        render_equation(latex, filename)
