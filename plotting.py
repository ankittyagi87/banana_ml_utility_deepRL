import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

scores = np.load("results/scores.npy")
scores_double_dqn = np.load("results/scores_double_dqn.npy")
scores_double_dqn_dueling = np.load("results/scores_double_dqn_dueling.npy")
scores_double_dqn_dueling_prioritized = np.load("results/scores_double_dqn_dueling_prioritized.npy")

t = np.arange(len(scores))
scores = pd.DataFrame(scores).rolling(window=50).mean()
scores_double_dqn = pd.DataFrame(scores_double_dqn).rolling(window=50).mean()
scores_double_dqn_dueling = pd.DataFrame(scores_double_dqn_dueling).rolling(window=50).mean()
scores_double_dqn_dueling_prioritized = pd.DataFrame(scores_double_dqn_dueling_prioritized).rolling(window=50).mean()
plt.plot(t[:900], scores[:900])
plt.plot(t[:900], scores_double_dqn[:900])
plt.plot(t[:900], scores_double_dqn_dueling[:900])
plt.plot(t[:900], scores_double_dqn_dueling_prioritized[:900])
plt.title("Scores of different agents")
plt.legend(['QNetwork', 'Double QNetwork', 'Double QNetwork with dueling', 'Double dueling QNetwork with prioritized replay'])
plt.xlabel('Number of episodes')
plt.ylabel('Scores')
plt.savefig('foo.png')