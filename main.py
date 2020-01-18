import matplotlib.pyplot as plt
import numpy as np
import argparse

import random
import os
import sys
import string

from time import time
from datetime import datetime


ARMS = 10
RUNS = 10
STEPS_PER_RUN = 1000
TRAINING_STEPS = 10
TESTING_STEPS = 5

NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SEED = None

# #############################################################################
#
# Parser
#
# #############################################################################


def get_arguments():
    def _str_to_bool(s):
        '''Convert string to boolean (in argparse context)'''
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Creating a k-armed bandit.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('-k', '--arms', type=int, default=ARMS,
                        help='Number of arms on the bandit. Default: '
                        + str(ARMS))
    parser.add_argument('-n', '--runs', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(RUNS))
    parser.add_argument('-s', '--steps', type=int, default=STEPS_PER_RUN,
                        help='Number of steps in each run. One run step is '
                        'the ensemble of training steps and testing steps. '
                        'Default: ' + str(STEPS_PER_RUN))
    parser.add_argument('--training_steps', type=int, default=TRAINING_STEPS,
                        help='Number of runs to be executed. Default: '
                        + str(TRAINING_STEPS))
    parser.add_argument('--testing_steps', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(TESTING_STEPS))
    parser.add_argument('-c', type=float, default=1,
                        help='Constant for the upper confidence bound (UCB) '
                        'case. This constant is ignored for the other agents. '
                        'Default: c=1')

    return parser.parse_args()


# #############################################################################
#
# Plotting
#
# #############################################################################


def plot_reward_distr(data):
    '''Creates a violing plot for the reward distributions for each one of the
    k actions in the k-armed bandit.'''

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=True)

    ax.set_ylabel('Reward distribution')
    ax.set_title('The {}-armed testbed'.format(len(data)))

    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xlim(0.25, len(data) + 0.75)
    ax.set_xlabel('Action')

    ax.violinplot(data,
                  showmeans=True,
                  widths=0.25,
                  showmedians=False,
                  showextrema=False)

    plt.axhline(y=0, color='r', linestyle='--')
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    plt.show()


def plot_line_variance(ax, data, gamma=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.

    ax:     axis object where the plot will be drawn
    data:   data of shape (num_trials, timesteps)
    gamma:  (optional) scaling of the standard deviation around the average
            if ommitted, gamma = 1.'''

    avg = np.average(data, axis=0)
    std = np.std(data, axis=0)

    ax.plot(avg)
    ax.fill_between(len(avg), avg + gamma * std, avg - gamma * std, alpha=0.1)


def plot3(args, training_return, regret, reward):
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True)
    fig.suptitle('This is a somewhat long figure title', fontsize=16)

    plot_line_variance(axs[0], training_return)
    axs[0].set_title('Average training return')

    plot_line_variance(axs[1], regret)
    axs[1].set_title('Average regret per step')

    plot_line_variance(axs[2], reward)
    axs[2].set_title('Policy reward')

# #############################################################################
#
# Helper functions
#
# #############################################################################


def softmax(x):

    # subtract max for numerical stability
    # (does not change result because of identity softmax(x) = softmax(x + c))
    z = x - max(x)

    return np.exp(z) / np.sum(np.exp(z))


def random_argmax(vector):
    '''Helper function to select argmax at random... not just first one.'''

    index = np.random.choice(np.where(vector == vector.max())[0])

    return index

# #############################################################################
#
# Main
#
# #############################################################################


class Bandit():

    def __init__(self, agent, agent_param, k=10, seed=SEED):

        np.random.seed(seed)

        self.k = k

        self.agent = agent
        self.agent_param = agent_param

        print('Initializing {}-armed bandit...\n\nThe true values q_*(a) for '
              'each action a=0, 1,..., {} were selected according to a normal '
              'distribution with mean zero and unit variance and then the '
              'actual rewards were selected according to a mean q_*(a) unit '
              'variance normal distribution.'.format(k, k-1))

        # defines the true value q_star for each action a=0, 1, ..., k
        self.q_star = np.random.randn(k)

        # defines the rewards distributions for each action a=0, 1, ..., k
        # according to normal densities mean q_star(a) and variance 1
        # Q = np.random.normal(q_star, 1)

        # generates data for violin plot
        data = [sorted(np.random.normal(action, 1, 10000))
                for action in self.q_star]
        plot_reward_distr(data)

    def get_reward(self, action):
        '''Action produces a reward from a normal distribution with mean
        q_*(action) and variance 1'''

        reward = np.random.normal(self.q_star[action], 1)

        return reward

    def get_regret(self, action):
 
        regret = (max(self.q_star) - self.q_star[action])

        return regret

    def boltzmann(self):
        pass

    def ucb(self, action, step, c=1):

        action = random_argmax(
            self.Q[action] + c * np.sqrt(np.log(step) / self.N[action]))
        return action

    def thompson(self, observation):
        '''Picks action according to Thompson sampling with Beta posterior for
        action selection.'''

        sampled_means = self._get_posterior_sample()
        action = random_argmax(sampled_means)
        return action

    def train(self, num_steps):

        for i in range(num_steps):
            # choose the best action
            action = random_argmax(self.Q)
            print('Training step {} chose action {}'.format(i + 1, action + 1))

            # calculate reward and update variables
            R = self.get_reward(action)
            self.N[action] += 1
            self.Q[action] = (self.Q[action]
                              + (R - self.Q[action]) / self.N[action])

        action = random_argmax(self.Q)
        print('Optimal action: {}'.format(action + 1))
        return action, self.Q[action]

    def test(self, action, num_steps):

        reward = 0
        for i in range(num_steps):
            action = self.ucb(action, i + 1, self.agent_param)
            reward += self.get_reward(action)
            print('Test step {} chose action {}'.format(i + 1, action + 1))
        # calculates average reward
        reward = reward / num_steps
        print('Optimal action: {}'.format(action + 1))
        return reward

    def run(self, num_runs, num_steps, training_steps, testing_steps):

        t0 = time()

        self.training_return = np.zeros((num_runs, num_steps))
        self.regret = np.zeros((num_runs, num_steps))
        self.reward = np.zeros((num_runs, num_steps))

        for i in range(num_runs):
            t1 = time()
            self._onerun(i, num_steps, training_steps, testing_steps)
            t = time() - t1
            print('Run {:2d} completed in {:2f} seconds.'.format(i, t))

        t = time() - t0
        print('{} runs completed in {:2f} seconds.'.format(num_runs, t))

    def _onerun(self, idx, num_steps, training_steps, testing_steps):

        # randomly seeds the generator at the start of each run
        np.random.seed(np.random.randint(2**32))

        # initialise variables
        self.Q = np.zeros(self.k)
        self.N = np.zeros(self.k)

        for i in range(num_steps):
            action, self.training_return[idx, i] = self.train(training_steps)
            self.reward[idx, i] = self.test(action, testing_steps)
            self.regret[idx, i] = self.get_regret(action)

    def _get_posterior_sample(self):
        self.prior_success = np.array([a0 for arm in range(self.k)])
        self.prior_failure = np.array([b0 for arm in range(self.k)])

        return np.random.beta(self.prior_success, self.prior_failure)


def main():

    # parses command line arguments
    args = get_arguments()

    # create Bandit environment and define agent
    env = Bandit(agent='ucb', agent_param=1, k=args.arms, seed=args.seed)

    # run bandit
    env.run(args.runs, args.steps, args.training_steps, args.testing_steps)

    # plot results
    plot3(args, env.training_return, env.regret, env.reward)


if __name__ == '__main__':
    main()
