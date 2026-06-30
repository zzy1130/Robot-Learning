# Robot Learning 教程：模型预测控制 (Model Predictive Control) 与动态矩阵控制 (Dynamic Matrix Control)

本章重点讲解在机器人控制领域广泛应用的 **模型预测控制 (Model Predictive Control, MPC)** 算法，并结合 Gitee 上的开源云台控制项目 [clangwu/mpc_control](https://gitee.com/clangwu/mpc_control)，重点剖析其核心子类——**动态矩阵控制 (Dynamic Matrix Control, DMC)** 算法。

我们将通过数学公式推导、控制器架构框图（Mermaid 流程图）以及 Python 物理仿真实验，深入对比经典 PID 控制与模型预测控制 (DMC) 在具有 **纯输入延时 (Input Delay)** 的机器人云台轨迹跟踪任务中的性能差异。

---

## 1. 模型预测控制 (MPC) 与动态矩阵控制 (DMC) 概述

### 1.1 什么是 MPC？
模型预测控制并不是指一种具体的单一算法，而是一类以 **“预测模型”、“滚动优化”、“反馈校正”** 为共同特征的闭环控制策略的总称。它的基本思想是：
1. **预测模型 (Predictive Model)**：根据系统的历史输入输出以及当前的控制计划，预测未来一段时间内（预测时域 **P**）系统输出的演化趋势。
2. **滚动优化 (Receding Horizon Optimization)**：通过求解一个带约束的二次规划或极值问题，计算出未来一段时间内（控制时域 **M**）的最优控制增量序列。在每个采样时刻，算法只执行控制增量序列的第一个元素。
3. **反馈校正 (Feedback Correction)**：在每个新采样时刻，利用系统实际测量输出对预测模型进行修正，克服由于建模误差、环境扰动等引起的不确定性。

### 1.2 什么是 DMC？
**动态矩阵控制 (DMC)** 是工业界最经典、应用最广泛的一种模型预测控制算法。它采用系统的 **阶跃响应 (Step Response)** 作为其预测模型。DMC 具有以下核心优势：
- **无需解析的状态空间方程**：直接使用通过实验测量的非参数化阶跃响应数据，极大降低了系统辨识的门槛。
- **天然的延时补偿**：由于阶跃响应数据天然地包含了纯时间滞后，因此预测模型能够自动学习并处理带有纯延时的系统。

### 1.3 传统 PID 与 MPC 的对比
在面对具有强滞后、多变量强耦合或控制量存在饱和限幅的机器人系统（如机械臂关节、高性能防抖云台）时，传统 PID 的局限性尤为突出：
- **时滞导致的不稳定**：在有纯延时的系统里，PID 的反馈无法预知控制输入的滞后效果，增益稍微调大便会导致严重的超调或自激振荡。
- **无法前瞻未来状态**：PID 是一种典型的“滞后反应式”控制，而 MPC 通过在时域上预测未来的响应，能够提前调整控制量，实现更平滑的无超调跟踪。

---

## 2. 动态矩阵控制 (DMC) 算法数学原理

### 2.1 预测模型与动态矩阵
假设系统在单位阶跃输入下，各个采样时刻的输出响应值为 *a_1, a_2, ..., a_N*。在时刻 *k*，预测时域为 **P**，控制时域为 **M** (其中 **M** <= **P**)。
在未来 **P** 个步长内，由于控制量增量 *dU* 产生的系统预测输出向量与自由响应向量 *Y_0(k)* 之间满足以下预测模型：

<p align="center">
  <img src="images/eq_dmc_prediction_model.png" alt="DMC Prediction Model" width="300">
</p>

其中，动态矩阵 *A* 的维度为 **P** x **M**，其元素由阶跃响应系数填充而成：

<p align="center">
  <img src="images/eq_dmc_matrix_a.png" alt="DMC Dynamic Matrix A" width="450">
</p>

### 2.2 参考轨迹 (Reference Trajectory)
为了避免系统输出变化过于剧烈产生冲击，我们通常不要求系统输出突变到目标设定值 *w(k)*，而是引入平滑衰减因子 *alpha* (0 <= *alpha* < 1) 来引导系统输出向设定值靠拢。参考轨迹的递归表达式为：

<p align="center">
  <img src="images/eq_dmc_ref_traj.png" alt="Reference Trajectory" width="450">
</p>

### 2.3 反馈校正与滚动移位
在时刻 *k* 采集到实际输出 *y(k)* 后，我们可以计算出当前预测误差，并引入长度为 **P** 的反馈校正向量 *h* 进行闭环修正：

<p align="center">
  <img src="images/eq_dmc_correction.png" alt="Feedback Correction" width="400">
</p>

校正后的未来预测向量在进入下一个采样时刻时，需要通过一个移位算子 *S* 进行左移操作（丢弃已发生的历史预测，并在末尾复制最后一项）：

<p align="center">
  <img src="images/eq_dmc_shift.png" alt="Shift Vector" width="250">
</p>

### 2.4 二次型目标函数与解析求解
DMC 的优化目标是在控制输入增量不过大的前提下，使预测输出尽可能接近参考轨迹。其目标函数可以写为：

<p align="center">
  <img src="images/eq_dmc_objective.png" alt="Objective Function" width="420">
</p>

其中，*Q* 为误差加权矩阵，*R* 为控制增量惩罚矩阵。对上式求导并令其为 0，可以得到无约束情况下的解析最优解：

<p align="center">
  <img src="images/eq_dmc_optimal_du.png" alt="Analytical Optimal solution" width="400">
</p>

在当前步，我们只把解出的第一个控制增量施加给系统，并滚动更新控制器输出：

<p align="center">
  <img src="images/eq_dmc_control_law.png" alt="Control Output Update" width="280">
</p>

---

## 3. 云台系统物理建模与延时挑战

### 3.1 传递函数与状态空间模型
机器人云台（如相机的俯仰/偏航轴）通常由带有纯延时的直流伺服电机驱动。云台的连续传递函数通常可写为二阶时滞系统：

<p align="center">
  <img src="images/eq_gimbal_tf.png" alt="Gimbal Transfer Function" width="260">
</p>

其中，*theta* 为云台倾角，*V* 为控制电压。*K* 为放大系数，*tau* 为机械时间常数，*L* 为通信或信号处理带来的纯时滞。
将其转换为状态空间方程（设定状态变量 *x_1 = theta*, *x_2 = dtheta/dt*）可写为：

<p align="center">
  <img src="images/eq_gimbal_state_space.png" alt="State Space Equation" width="450">
</p>

### 3.2 控制时滞的危害
当时滞 *L* 存在时，由于信息反馈的滞后，PID 控制器计算的当前误差反映的是过去状态。若 PID 控制器增益较大，控制器会频繁过度输出，造成系统发散。
而 DMC 能够通过阶跃响应模型中的零响应系数，天然感知未来 *L* 时间内输出不会发生改变的“先验信息”，从而在优化求解时合理规划控制输入，实现高精度无超调跟踪。

---

## 4. 动态矩阵控制 (DMC) 结构框图

```mermaid
graph TD
    subgraph "输入与设定值 (Inputs & Target)"
        setpoint["目标设定值 w(k)"]
    end

    subgraph "预测模型与反馈校正 (DMC Predictor & Corrector)"
        error_calc["当前实测输出 y(k)"]
        correction["反馈校正 Y_cor(k) = Y_0(k) + h * e_y(k)"]
        shift["移位操作 S * Y_cor(k)"]
        ref_traj["参考轨迹 Y_ref(k)"]
    end

    subgraph "滚动优化 (Receding Horizon Optimizer)"
        optimization["最优控制增量计算: ΔU = (A^T Q A + R)^-1 A^T Q (Y_ref - Y_0)"]
        control_act["累加控制输入: u(k) = u(k-1) + Δu(k)"]
    end

    subgraph "物理系统 (Gimbal Plant)"
        gimbal["控制量延时 -> 机器人云台物理仿真"]
    end

    setpoint --> ref_traj
    error_calc --> correction
    correction --> shift
    shift -->|移位预测 Y_0(k)| optimization
    ref_traj --> optimization
    optimization --> control_act
    control_act -->|施加控制 u(k)| gimbal
    gimbal -->|实测反馈 y(k)| error_calc
```

---

## 5. Python 代码与仿真说明

本工程在当前路径下提供了以下两个 Python 实现文件：
1. **[mpc_controller.py](mpc_controller.py)**：
   - 实现了基于二次型优化求解的工业 `DMCController`。
   - 自定义预测时域 **P**、控制时域 **M**、参考轨迹滤波常数 *alpha*、权重矩阵 *Q* 与 *R*。
   - 自动生成移位矩阵 *S*，并在后台预先计算高维伪逆矩阵提升控制器实时性能。
2. **[simulation.py](simulation.py)**：
   - 构建了具有输入延时 *L* = 0.04s（在采样周期 *dt* = 0.01s 下相当于 4 步延迟）的二阶云台物理仿真类 `GimbalSim`，使用第四阶龙格-库塔 (`rk4_step`) 进行连续物理模拟。
   - 运行 open-loop 单位阶跃响应测试，自动为控制器获取阶跃系数 *a_i*。
   - 在多步跃变目标轨迹下，对比具有输入延时的云台系统在 **PID 控制** 与 **动态矩阵控制 (DMC)** 下的位置跟踪响应、控制输入能量与误差表现。
   - 绘制对比曲线并自动保存为 `trajectory_comparison.png`。

### 5.1 仿真结果对比与深度分析

运行 `simulation.py` 后生成的对比图如下：

![Simulation Tracking Comparison](trajectory_comparison.png)

#### 重点结果指标分析：
1. **跟踪精度与系统超调**：
   - **PID 控制器**（红色曲线）：由于受到 0.04s 纯时滞的影响，即使经过精细调参，位置波形在目标变化时刻依然出现了明显的相位滞后和超调，且振荡收敛缓慢。如果尝试将比例系数 **K_p** 继续调大，系统会由于正反馈积累立刻陷入发散振荡。
   - **动态矩阵控制 DMC**（绿色曲线）：在目标突变后表现极其稳健。不仅没有出现严重的超调与振荡，而且在极短的上升时间后紧紧锁定了设定值。这证明了 DMC 具有卓越的时滞补偿能力。
2. **控制输入（电压曲线）**：
   - **PID 控制器** 的输出电压在设定值跃变时刻表现出剧烈的波动甚至饱和，这是因为延迟使得误差积累，导致 PID 的积分与比例项产生过度补偿。
   - **DMC 预测控制器** 表现出的电压输出极其平缓、合理。它不仅能在阶跃开始前通过模型预测“有计划地”分配能量，而且在接近设定值时能提前减小动作，有效消除了伺服机构受到的物理冲击。

---

## 6. 总结与后续

动态矩阵控制 (DMC) 很好地诠释了通过“已知模型预测未来”以摆脱传统负反馈时滞限制的核心思路。在控制理论的发展史中，MPC 的出现架起了从“无模型 PID”到“模型驱动现代控制”的坚实桥梁。

在后续章节中，我们将探索当模型变得非线性、状态难以完全获取时，如何结合现代强化学习技术，使用 **深度强化学习控制 (DRL)** 在更复杂的机器人交互场景下进行端到端的灵巧运动控制。
