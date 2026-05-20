# Executing Code

The code requires Python 3.11.0. Later Python versions may or may not work. Assuming Python 3.11.0 is installed, run:

```bash
py -3.11 -m venv myvenv  # Creates venv named myvenv using Python 3.11
.\myvenv\Scripts\activate  # Use Activate.ps1 for PowerShell; use "activate" for bash on Linux/Mac
pip install -r requirements.txt  # Make sure you are in the same directory as requirements.txt
```

To run the code with pits, open the Jupyter notebook and set `RUN_WITH_PITS = True`. By default it is `False`.

# Policy-Gradient-vs-Value-Based-RL-in-JAX

Gymnasium was used for the environment because it is a widely used library with a broad range of tutorials and documentation available online, which simplifies implementation. The environment is a simple 7×7 grid. The agent starts at the top-right corner at point (1, 1) of the environment and works toward point (5, 5). The agent will not move if it attempts to move off any edge of the grid. The agent cannot move diagonally; it can only move (select an action) up, down, left, or right. The experiment was run twice: once with pits and once without. A pit adds a negative reward when the agent steps onto that square and resets the agent’s position to (1, 1) when entered. The episode continues even if a pit has been entered.

Observations provided to the reinforcement algorithms are: the normalized coordinates, the Manhattan distance to the goal, and the direction to the goal.

A small reward is given if a step moves the agent closer to the goal, and a small penalty is given for moving away. In addition, a constant small penalty is added to the reward to encourage shorter paths. A large penalty is applied if the agent steps in a pit. Lastly, a large reward is provided for reaching the goal.

A wrapper using `gym.Wrapper` was implemented.

The research questions investigated in this project are:

1. Compare DQN vs REINFORCE-wb in a blank environment. As required by the assignment, 10 different seeds are tried for each algorithm and set of hyperparameter settings (learning rates: 0.001, 0.01, 0.1; discount factors: γ = 0.9, 0.99).
2. Observe the effects of a vertical pit (as defined above) on the learning of DQN and REINFORCE-wb.
