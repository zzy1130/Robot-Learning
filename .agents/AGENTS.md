# Robot Learning Tutorial Project Goals

This workspace is dedicated to building a comprehensive, step-by-step tutorial on Robot Learning, structured into several key control and learning branches.

## Project Structure & Content Guidelines
For each control/learning branch, we will create a dedicated folder containing:
1. **README.md**: A clear, mathematically rigorous explanation of the control/learning techniques, including technical details, equations, and illustrative images/diagrams from research.
2. **Python Implementation**: Clean, modular Python implementation of the algorithms from scratch.
3. **Simulation/Demo**: A Python script running a demo (e.g., simulating a robotic system like a mass-spring-damper, a robotic arm, or a cart-pole) to visualize and verify the control performance.

## Roadmap & Branches
- [x] **PID Control (Completed)**
  - Classical PID (Proportional-Integral-Derivative) control.
  - Advanced/subsequent versions: Computed Torque Control (CTC) for multi-joint robotic manipulators.
  - Python code and simulations (e.g., trajectory tracking of a robotic arm or mass-spring-damper).
- [x] **MPC (Model Predictive Control) (Completed)**
  - Dynamic Matrix Control (DMC) based on step response.
  - Delay compensation and rolling horizon optimization for stable second-order gimbal servo systems.
- [ ] **DRL (Deep Reinforcement Learning)**
- [ ] **Imitation Learning**

## Markdown & Rendering Guidelines (Learned from PID Branch)
To ensure the `README.md` in all branches renders perfectly on online platforms (Gitee/GitHub):
1. **Equations and Math Blocks**: Do not use raw LaTeX math blocks (`$$ ... $$`). Instead, pre-compile LaTeX equations locally to transparent PNG images (using matplotlib math rendering) and save them inside the folder's `images/` directory. Reference them using `<p align="center"><img src="images/eq_name.png" width="X"></p>` where `X` is an appropriate width (e.g., 200 to 500 pixels).
2. **Inline Math Terms**: Avoid `$ ... $` for inline variables and simple expressions. Instead, use plain markdown (e.g. `**K_p**`, `*M(q)*`, `*tau*`) for readability and foolproof rendering.
3. **Mermaid Diagrams**: Subgraph titles containing special characters (like parentheses `(`, `)`) must be enclosed in double quotes (e.g., `subgraph "Title (Subtitle)"`) to prevent syntax parsing errors in Markdown rendering engines.
4. **Offline Assets**: All diagrams, illustrations, and formulas must be stored locally under the respective branch's `images/` folder so they are self-contained and render offline.

## Algorithm Rigor & Verification Guidelines (Learned from MPC Branch)
To prevent model-plant mismatch, algorithmic bugs, and poor control performance, subsequent development (especially DRL and Imitation Learning) must adhere to:
1. **Physics & Modeling Consistency (物理与建模一致性)**:
   - Carefully analyze the stability and character of the plant model before implementing controllers.
   - For standard step-response controllers (like DMC), ensure the plant is asymptotically stable (e.g., closed-loop position servo or velocity loop), otherwise step response coefficients will not settle.
2. **Algorithmic State Consistency (算法状态更新一致性)**:
   - In receding horizon or sequential decision-making algorithms, always update internal model predictions using the **actually executed control actions** (e.g., the first element of the optimized sequence), NOT the entire planned future sequence, since the sequence is re-planned at every time step.
3. **Mandatory Simulation Verification (强制运行仿真验证)**:
   - **CRITICAL**: Before committing code or finalizing the README, the agent **MUST** run the python simulation script locally in the shell.
   - Visually check the resulting tracking plot (e.g., `trajectory_comparison.png`) to ensure the controller converges stably with expected dynamics. If the curve oscillates, has steady-state error, or looks worse than a simple baseline (like PID), treat it as a warning of a bug or modeling error, and resolve it before committing.
4. **Intuitive & Physical Context (物理背景与直观阐释)**:
   - READMEs must explicitly define the real-world application background (e.g., RoboMaster gimbal, delay sources) and use intuitive analogies (e.g., driving with delay) to help learners understand the physical motivation behind the math.


