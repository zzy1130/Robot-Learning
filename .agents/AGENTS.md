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
- [ ] **MPC (Model Predictive Control)**
- [ ] **DRL (Deep Reinforcement Learning)**
- [ ] **Imitation Learning**

## Markdown & Rendering Guidelines (Learned from PID Branch)
To ensure the `README.md` in all branches renders perfectly on online platforms (Gitee/GitHub):
1. **Equations and Math Blocks**: Do not use raw LaTeX math blocks (`$$ ... $$`). Instead, pre-compile LaTeX equations locally to transparent PNG images (using matplotlib math rendering) and save them inside the folder's `images/` directory. Reference them using `<p align="center"><img src="images/eq_name.png" width="X"></p>` where `X` is an appropriate width (e.g., 200 to 500 pixels).
2. **Inline Math Terms**: Avoid `$ ... $` for inline variables and simple expressions. Instead, use plain markdown (e.g. `**K_p**`, `*M(q)*`, `*tau*`) for readability and foolproof rendering.
3. **Mermaid Diagrams**: Subgraph titles containing special characters (like parentheses `(`, `)`) must be enclosed in double quotes (e.g., `subgraph "Title (Subtitle)"`) to prevent syntax parsing errors in Markdown rendering engines.
4. **Offline Assets**: All diagrams, illustrations, and formulas must be stored locally under the respective branch's `images/` folder so they are self-contained and render offline.

