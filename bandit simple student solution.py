# -*- coding: utf-8 -*-
"""

 multiarmed_bandit.py  (author: Anson Wong / git: ankonzoid)
 Adapted from Github

"""
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Settings
# =========================
bandit_probs = [0.1, 0.5, 0.9]  # bandit probabilities of success
N_bandits = len(bandit_probs)
N_experiments = 1000  # number of experiments to perform
N_episodes = 100 # number of episodes per experiment
epsilon = 0.1  # probability of random exploration (fraction)
save_fig = False  # if false -> plot, if true save as file in same directory
save_format = ".png"  # ".pdf" or ".png"


def get_reward(action, bandit_probs):
    rand = np.random.random()  # [0.0,1.0)
    reward = 1 if (rand < bandit_probs[action]) else 0
    return reward


def update_Q(k, Q, action, reward):
    # Update Q action-value using:
    # Q(a) <- Q(a) + 1/(k+1) * (r(a) - Q(a))
    k[action] += 1  # update action counter k -> k+1
    k[action] = k[action] + 1
    Q[action] += (1./k[action]) * (reward - Q[action])
    return (k, Q)


def choose_action(Q, epsilon, N_bandits):
    if epsilon > np.random.random():
        action = np.random.randint(N_bandits)
    else:
        action = np.random.choice(np.flatnonzero(Q == Q.max()))
    # Choose action using an epsilon-greedy agent
    # np.random.random()  # [0.0,1.0)
    # np.random.randint() # select a random integer between 0 and input
    # np.random.choice(np.flatnonzero(Q == Q.max())) # choose randomly among best options
    return action

def choose_action_2(Q, tau, N_bandits):
    """softmax action selection over m alternatives"""
    if tau > 0:
        # print('Q', Q)
        e = np.exp(np.asarray(Q) / tau)
        # print('e', e)
        dist = e / np.sum(e)
        # print('dist', dist)
        roulette = np.random.random()
        # print('Roulette', roulette)
        cum_prob = 0
        for i in range(N_bandits):
            cum_prob = cum_prob + dist[i]
            # print('cum_prob', i, cum_prob)
            if cum_prob >= roulette:
                action = i
                # print('action', i)
                break  # stops computing probability of action
                # selection as soon as cumulative probability exceeds roulette
    else:  # Means that either first did not work or tau = 0
        action = np.random.choice(np.flatnonzero(Q == Q.max()))
    return action

# =========================
# Define an experiment
# =========================
def experiment(N_episodes):
    action_history = []
    reward_history = []
    k = np.zeros(N_bandits, dtype=np.int)  # number of times action
    Q = np.zeros(N_bandits, dtype=np.float)  # estimated value
    for episode in range(N_episodes):

        # Choose action from agent (from current Q estimate)
        action = choose_action_2(Q, epsilon, N_bandits)
        # Pick up reward from bandit for chosen action
        reward = get_reward(action, bandit_probs)
        # Update Q action-value estimates
        (k, Q) = update_Q(k, Q, action, reward)
        # Append to history
        action_history.append(action)
        reward_history.append(reward)
    return (np.array(action_history), np.array(reward_history))

# =========================
#
# Start multi-armed bandit simulation
#
# =========================


print("Running multi-armed bandits with N_bandits = {} and agent epsilon = {}".format(N_bandits, epsilon))
reward_history_avg = np.zeros(N_episodes)  # reward history experiment-averaged
action_history_sum = np.zeros((N_episodes, N_bandits))  # sum action history
for i in range(N_experiments):
    # bandit = Bandit(bandit_probs)  # initialize bandits
    # agent = Agent(bandit, epsilon)  # initialize agent
    (action_history, reward_history) = experiment(N_episodes)
    # perform experiment

    if (i + 1) % (N_experiments / 20) == 0:
        print("[Experiment {}/{}]".format(i + 1, N_experiments))
        print("  N_episodes = {}".format(N_episodes))
        # print("  bandit choice history = {}".format(action_history + 1))
        # print("  reward history = {}".format(reward_history))
        print("  average reward = {}".format(np.sum(reward_history) / len(reward_history)))
        print("")
    # Sum up experiment reward (later to be divided to represent an average)
    reward_history_avg += reward_history
    # Sum up action history
    for j, (a) in enumerate(action_history):
        action_history_sum[j][a] += 1

reward_history_avg /= np.float(N_experiments)
# print("reward history avg = {}".format(reward_history_avg))

# =========================
# Plot reward history results
# =========================
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(reward_history_avg)
plt.xlabel("Episode number")
plt.ylabel("Rewards collected".format(N_experiments))
plt.title("Bandit reward history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon))
ax = plt.gca()
ax.set_xscale("log", nonposx='clip')
plt.xlim([1, N_episodes])
if save_fig:
    output_file = "MAB_rewards" + save_format
    plt.savefig(output_file, bbox_inches="tight")
else:
    plt.show()

# =========================
# Plot action history results
# =========================
plt.figure(figsize=(8, 6), dpi=100)
for i in range(N_bandits):
    action_history_sum_plot = 100 * action_history_sum[:, i] / N_experiments
    plt.plot(list(np.array(range(len(action_history_sum_plot)))+1),
             action_history_sum_plot,
             linewidth=5.0,
             label="Bandit #{}".format(i+1))
plt.title("Bandit action history averaged over {} experiments (epsilon = {})".format(N_experiments, epsilon), fontsize=11)
plt.xlabel("Episode Number", fontsize=11)
plt.ylabel("Bandit Action Choices (%)", fontsize=11)
leg = plt.legend(loc='upper left', shadow=True, fontsize=11)
ax = plt.gca()
ax.set_xscale("log", nonpositive='clip')
plt.xlim([1, N_episodes])
plt.ylim([0, 100])
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
for legobj in leg.legendHandles:
    legobj.set_linewidth(16.0)
if save_fig:
    output_file = "MAB_actions" + save_format
    plt.savefig(output_file, bbox_inches="tight")
else:
    plt.show()


