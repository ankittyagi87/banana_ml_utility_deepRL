import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

scores = np.load("results/scores.npy")
scores_double_dqn = np.load("results/scores_double_dqn.npy")
scores_double_dqn_dueling = np.load("results/scores_double_dqn_dueling.npy")
scores_double_dqn_dueling_prioritized = np.load("results/scores_double_dqn_dueling_prioritized.npy")

t = np.arange(len(scores))
scores = pd.DataFrame(scores).rolling(window=100).mean()
scores_double_dqn = pd.DataFrame(scores_double_dqn).rolling(window=100).mean()
scores_double_dqn_dueling = pd.DataFrame(scores_double_dqn_dueling).rolling(window=100).mean()
scores_double_dqn_dueling_prioritized = pd.DataFrame(scores_double_dqn_dueling_prioritized).rolling(window=100).mean()
plt.plot(t[:800], scores[:800])
plt.plot(t[:800], scores_double_dqn[:800])
plt.plot(t[:800], scores_double_dqn_dueling[:800])
plt.plot(t[:800], scores_double_dqn_dueling_prioritized[:800])
plt.title("Scores of different agents")
plt.legend(['QNetwork', 'Double QNetwork', 'Double QNetwork with dueling', 'Double dueling QNetwork with prioritized replay'])
plt.xlabel('Number of episodes')
plt.ylabel('Scores')
plt.savefig('results/Scores_of_different_agents.png')

plt.clf()
scores = np.load("results/scores.npy")
scores_lr_2 = np.load("results/scores_lr_2_10-4.npy")
scores_lr_3 = np.load("results/scores_lr_10-3.npy")

t = np.arange(len(scores))
scores = pd.DataFrame(scores).rolling(window=100).mean()
scores_lr_2 = pd.DataFrame(scores_lr_2).rolling(window=100).mean()
scores_lr_3 = pd.DataFrame(scores_lr_3).rolling(window=100).mean()

plt.plot(t[:800], scores[:800])
plt.plot(t[:800], scores_lr_2[:800])
plt.plot(t[:800], scores_lr_3[:800])
plt.title("Scores for different learning rates")
plt.legend(['lr-5e-4', 'lr-2e-4', 'lr-1e-3'])
plt.xlabel('Number of episodes')
plt.ylabel('Scores')
plt.savefig('results/Scores_for_different_learning_rates.png')