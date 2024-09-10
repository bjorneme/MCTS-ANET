# Monte Carlo Tree Search and Neural Networks for Hex
This project implements an On-Policy Monte Carlo Tree Search (MCTS) system combined with reinforcement learning and a neural network to play the board game Hex.

## Project Overview
- **Goal**: Train an intelligent agent to play Hex using MCTS, reinforcement learning, and a policy network.
- **MCTS**: Guides the agent's decisions by simulating games and using the results to train the policy network.
- **Reinforcement Learning**: Updates the policy network after each game based on the results of many simulated games.
- **Neural Network (ANET)**: Learns to predict optimal moves from game states and is periodically updated throughout training.

## Key Features
- **Hex Game**: Simulate and play the board game Hex using a k x k grid (3 ≤ k ≤ 10).
- **MCTS**: Performs simulations to evaluate and backpropagate move values, refining the policy over time.
- **Training and Competitions**: The agent improves by playing games, and periodically saved models compete in the Tournament of Progressive Policies (TOPP).
- **On-Policy Training**: The same policy network is used for decision-making during simulations and training.
