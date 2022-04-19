import matplotlib.pyplot as plt

# File to debug
from models.epsilon_greedy_sample_average import EpsilonGreedySampleAverage
from models.ucb_sample_average import UcbSampleAverage
from models.gradient_bandits import GradientBandits
from n_armed_testbed import NArmedTestbed

K = 10

# rl_method = EpsilonGreedySampleAverageUCB(K=K, c=2, Q0=5, eps=0.01)
# rl_method = EpsilonGreedySampleAverage(K=K, Q0=5, eps=0)
rl_method = GradientBandits(K=10, use_baseline=False)

testbed = NArmedTestbed(N=2000, K=K, solver=rl_method, mean=4)
rewards, choices = testbed.run(1000, alpha=0.4)

plt.plot(rewards.mean(axis=0))

plt.plot(choices.mean(axis=0))