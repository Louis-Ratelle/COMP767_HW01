import matplotlib.pyplot as plt
import numpy as np
import argparse

import os
import sys

from time import time
from datetime import datetime


ARMS = 10
RUNS = 10
STEPS_PER_RUN = 1000
TRAINING_STEPS = 10
TESTING_STEPS = 5

NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
#SEED = None
SEED = 1

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


def plot_violin(data):
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
    # plt.show()


def plot_line_variance(ax, data, gamma=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.

    ax:     axis object where the plot will be drawn
    data:   data of shape (num_trials, timesteps)
    gamma:  (optional) scaling of the standard deviation around the average
            if ommitted, gamma = 1.'''

    avg = np.average(data, axis=0)
    std = np.std(data, axis=0)

    ax.plot(avg + gamma * std, 'b--', linewidth=0.5)
    ax.plot(avg - gamma * std, 'b--', linewidth=0.5)
    ax.fill_between(range(len(avg)),
                    avg + gamma * std,
                    avg - gamma * std,
                    facecolor='blue',
                    alpha=0.1)
    ax.plot(avg)

def plot3(title, training_return, regret, reward):
    '''Creates the three required plots: average training return, regret
    and testing policy reward.'''

    fig, axs = plt.subplots(nrows=1, ncols=3,
                            constrained_layout=True,
                            figsize=(10,3))

    fig.suptitle(title, fontsize=12)

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
    '''Softmax implementation for a vector x.'''

    # subtract max for numerical stability
    # (does not change result because of identity softmax(x) = softmax(x + c))
    z = x - max(x)

    return np.exp(z) / np.sum(np.exp(z), axis=0)


def random_argmax(vector):
    '''Select argmax at random... not just first one.'''

    index = np.random.choice(np.where(vector == vector.max())[0])

    return index

# #############################################################################
#
# Agent definition
#
# #############################################################################


class Agent(object):
    def update(cls):
        '''Updates agent variables at the end of each step.'''
        pass

    def choose_action(cls):
        pass


class UCB(Agent):
    def __init__(self, c=1):
        self.c = c

    def __repr__(self):
        return 'UCB(c={})'.format(self.c)

    def reset(self, env):
        self.step = 1
        self.Q = np.zeros(env.k)
        self.N = np.zeros(env.k) + 1e-6    # avoids division by zero

    def choose_action(self):

        action = random_argmax(self.Q + self.c * np.sqrt(np.log(self.step) / self.N))
        self.step += 1

        return action

    def best_action(self):
        return random_argmax(self.Q)

    def update(self, action, R_list):
        R = R_list[-1]

        self.N[action] += 1
        self.Q[action] = self.Q[action] + (R - self.Q[action]) / self.N[action]


class Boltzmann(Agent):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def __repr__(self):
        return 'Boltzmann(alpha={})'.format(self.alpha)

    def reset(self, env):
        # initialise preferences
        self.H = np.zeros(env.k)

        self.Q = np.zeros(env.k)
        self.N = np.zeros(env.k)

        self.k = env.k

    def choose_action(self):

            # initialise probabilities according to preferences
            self.prob = softmax(self.H)

            # select action with highest probability
            # action = random_argmax(self.prob)
            action = np.random.choice(range(self.k), p=self.prob)
            return action

    def best_action(self):
        return random_argmax(self.Q)

    def update(self, action, R_list):
        other_actions = (np.arange(self.k) != action)

        R = R_list[-1]
        avg_return = np.average(R_list)

        self.N[action] += 1
        self.Q[action] = (self.Q[action]
                          + (R - self.Q[action]) / self.N[action])

        # print('Avg_return: {}'.format(avg_return), 'Q: {}'.format(self.Q[action]), avg_return == self.Q[action])
        # update preferences
        self.H[action] = (self.H[action]
                          + self.alpha
                          * (R - avg_return)
                          * (1 - self.prob[action]))

        self.H[other_actions] = (self.H[other_actions]
                                 - self.alpha
                                 * (R - avg_return)
                                 * self.prob[other_actions])


class Thompson(Agent):

    def __init__(self, mu = 0):
        self.mu = mu

    def __repr__(self):
        return "Thompson"

    def reset(self, env):

        # Keep initial values for formulas in update below
        self.mu_initial = np.ones(env.k) * self.mu
        self.sigma_initial = np.ones(env.k) * 5.0

        # Keep means and variances for all priors (these will be the updated posteriors)
        self.mu_0 = np.zeros(env.k)
        self.sigma_0 = np.ones(env.k)

        self.Q = np.zeros(env.k)
        self.N = np.zeros(env.k)

        self.k = env.k

    def choose_action(self):
        # Get samples from the posteriors (which act as priors)
        self.sample_mu = np.random.normal(self.mu_0, self.sigma_0)
        return random_argmax(self.sample_mu)

    def best_action(self):
        return random_argmax(self.Q)

    def update(self, action, R_list):
        R = R_list[-1]
        self.N[action] += 1
        self.Q[action] = (self.Q[action] + (R - self.Q[action]) / self.N[action])

        # The next two formulas are from the unpublished book of Mike Jordan "An Introduction to Probabilistic Graphical Models" page 193
        # when you have a Gaussian prior for mu and a Gaussian for the data (here we assume sigmas for both Gaussians are 1)
        self.mu_0[action] = (self.N[action] / (self.N[action]+1)) * self.Q[action] + (1 / (self.N[action]+1)) * self.mu_initial[action]
        self.sigma_0[action] = self.sigma_initial[action] * 1/(self.N[action]+1)



    """
    def thompson(self, observation):
        '''Picks action according to Thompson sampling with Beta posterior for
        action selection.'''

        sampled_means = self._get_posterior_sample()
        action = random_argmax(sampled_means)
        return action

    def _get_posterior_sample(self):
        self.prior_success = np.array([a0 for arm in range(self.k)])
        self.prior_failure = np.array([b0 for arm in range(self.k)])

        return np.random.beta(self.prior_success, self.prior_failure)
    
    """

# #############################################################################
#
# Environment definition
#
# #############################################################################


class Bandit():

    def __init__(self, agent, k=10, seed=SEED):

        np.random.seed(seed)

        self.k = k
        self.agent = agent

        # print('Initializing {}-armed bandit...\n\nThe true values q_*(a) for '
        #       'each action a=0, 1,..., {} were selected according to a normal '
        #       'distribution with mean zero and unit variance and then the '
        #       'actual rewards were selected according to a mean q_*(a) unit '
        #       'variance normal distribution.'.format(k, k-1))

        # defines the true value q_star for each action a=0, 1, ..., k
        self.q_star = np.random.randn(k)

    def plot_reward_distr(self):
        # generates data for violin plot
        data = [sorted(np.random.normal(action, 1, 10000))
                for action in self.q_star]
        plot_violin(data)

    def get_reward(self, action):
        '''Action produces a reward from a normal distribution with mean
        q_*(action) and variance 1'''

        reward = np.random.normal(self.q_star[action], 1)

        return reward

    def get_regret(self, action):

        regret = (max(self.q_star) - self.q_star[action])

        return regret

    def train(self, num_steps):

        R = []
        for step in range(num_steps):
            # choose the best action
            action = self.agent.choose_action()
            #print('Training step {} chose action {}'.format(i + 1, action + 1))

            # calculate reward and update variables
            R.append(self.get_reward(action))
            self.agent.update(action, R)

        action = self.agent.best_action()
        avg_return = np.average(R)
        return action, avg_return

    def test(self, action, num_steps):

        reward = 0
        for step in range(num_steps):
            reward += self.get_reward(action)
        # calculates average reward
        reward = reward / num_steps
        # print('Average reward: {:2f}'.format(reward))
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
            # print('Run {:2d} completed in {:2f} seconds.'.format(i + 1, t))

        t = time() - t0
        print('{} runs completed in {:2f} seconds.'.format(num_runs, t))

    def _onerun(self, idx, num_steps, training_steps, testing_steps):
        '''Executes one run of the bandit algorithm. One run executes
        num_steps in total. Each step in the run executes a number of
        trainining_steps followed by a number of test steps.

        num_steps:          number of steps in each run
        tranining_steps:    number of training steps
        test_steps:         number of test steps'''

        # randomly seeds the generator at the start of each run
        np.random.seed(None)

        # resets agent in the beginning of each run
        self.agent.reset(self)

        for step in range(num_steps):
            action, self.training_return[idx, step] = self.train(training_steps)
            self.reward[idx, step] = self.test(action, testing_steps)
            self.regret[idx, step] = self.get_regret(action)

# #############################################################################
#
# Main
#
# #############################################################################


def main():

    # parses command line arguments
    args = get_arguments()
    k = args.arms

    # create Bandit environment and define agent
    env = Bandit(agent=None, k=k, seed=args.seed)

    # run experiments with different agents
    for agent_name in [UCB, Boltzmann]:
        for param in 2.0 ** np.array([-2, -1, 0, 1, 2]):
            agent = agent_name(param)

            # # create Bandit environment and define agent
            # env = Bandit(agent=agent, k=k, seed=args.seed)
            env.agent = agent

           # run bandit
           env.run(args.runs,
                   args.steps,
                   args.training_steps,
                   args.testing_steps)
    
           # plot results
           separator = '_'
           title = separator.join([str(agent)])
           plot3(title, env.training_return, env.regret, env.reward)
    
           plt.show()
    
    for agent_name in [Thompson]:
        for param in [-10, -5, -1, 0, 1, 5, 10]:
            agent = agent_name(param)
            
            # env = Bandit(agent=agent, k=k, seed=args.seed)
            env.agent = agent
            
            # run bandit
            env.run(args.runs,
                    args.steps,
                    args.training_steps,
                    args.testing_steps)

            # plot results
            separator = '_'
            title = separator.join([str(agent)])
            plot3(title, env.training_return, env.regret, env.reward)

            plt.show()

    env.plot_reward_distr()
    plt.show()


if __name__ == '__main__':
    main()
