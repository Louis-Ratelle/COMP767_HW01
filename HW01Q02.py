import numpy as np
import gym
import argparse
import matplotlib.pyplot as plt
#from HW01Q01 import plot_line_variance

SEED = None
GAMMA = 0.9
TOL = 1e-6
RUNS = 5
TRAINING_EPISODES = 10
TESTING_EPISODES = 5
TEST_EVERY = 1
ENV = 'FrozenLake-v0'
NB_STEPS_IF_FALL_HOLE = 200

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

    parser = argparse.ArgumentParser(
        description='Applying dynamic programming '
        'to a discrete gym environment.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('--env', type=str, default=ENV,
                        help="The environment to be used. Environment must "
                        "have discrete actions and states. Common choices "
                        "are: 'FrozenLake8x8-v0','Taxi-v3', etc. Default:"
                        + ENV)
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help='Defines the discount rate. Default: '
                        + str(GAMMA))
    parser.add_argument('--tol', type=float, default=TOL,
                        help='Defines the tolerance for convergence of the '
                        'algorithms. Default: ' + str(TOL))
    parser.add_argument('--value_iteration', action="store_true",
                        help='If this flag is set, value iteration will be '
                        'used. If the flag is missing, policy iteration '
                        'will be used by default.')
    parser.add_argument('--render_policy', action="store_true",
                        help='If this flag is set, the optimal policy will '
                        'be applied to one episode of the environment and '
                        'will present the results.')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='If this flag is set, the algorithm will generate '
                        'more output, useful for debugging.')
    parser.add_argument('-n', '--runs', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(RUNS))
    parser.add_argument('--training_episodes', type=int,
                        default=TRAINING_EPISODES,
                        help='Number of training episodes to be executed.'
                        'Default: ' + str(TRAINING_EPISODES))
    parser.add_argument('--testing_episodes', type=int, default=TESTING_EPISODES,
                        help='Number of testing episodes to be executed.'
                        'Default: ' + str(TESTING_EPISODES))
    parser.add_argument('--test_every', type=int, default=TEST_EVERY,
                        help='Number of training iterations to execute before '
                        'each test. Default: ' + str(TEST_EVERY))

    return parser.parse_args()


# #############################################################################
#
# Plotting
#
# #############################################################################


def plot2(title, cumulative_reward_3d, timesteps_3d):
    '''Creates the two required plots: cumulative_reward and number of timesteps
    per episode.'''

    fig, axs = plt.subplots(nrows=1, ncols=2,
                            constrained_layout=True,
                            figsize=(10, 3))

    fig.suptitle(title, fontsize=12)

    cumulative_reward_2d = np.array(np.mean(cumulative_reward_3d, axis = 2))
    cumulative_reward_1d = np.array(np.max(np.max(cumulative_reward_3d, axis=2),axis=0))
    print("cumulative_reward_1d to do max reward per time step: ", cumulative_reward_1d)
    #print(cumulative_reward_2d.shape)
    plot_line_variance(axs[0], cumulative_reward_2d)
    axs[0].set_title('Cumulative reward')

    timesteps_2d = np.array(np.mean(timesteps_3d, axis=2))
    timesteps_1d = np.array(np.min(np.min(timesteps_3d, axis=2), axis=0))
    print("timesteps_1d to do min episode length per time step: ", timesteps_1d)
    plot_line_variance(axs[1], timesteps_2d)
    axs[1].set_title('Timesteps per episode')
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

    # ax.plot(avg + gamma * std, 'r--', linewidth=0.5)
    # ax.plot(avg - gamma * std, 'r--', linewidth=0.5)
    ax.fill_between(range(len(avg)),
                    avg + gamma * std,
                    avg - gamma * std,
                    facecolor='red',
                    alpha=0.2)
    ax.plot(avg)

# #############################################################################
#
# Policy
#
# #############################################################################


class Policy():
    def __init__(self, env, gamma=1, bVerbose=False, tol=1e-6):
        self.env = env
        self.gamma = gamma
        self.bVerbose = bVerbose
        self.tol = tol

        self.counter = 0

        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise NotImplementedError

        self.V = np.zeros(env.observation_space.n)
        self.pi = np.zeros(env.observation_space.n, dtype=int)

        self.env.reset()

        for s in range(env.observation_space.n):
            self.pi[s] = env.action_space.sample()


    def eval(self, bValueIteration=False):
        '''Evaluates the policy value for each state s in
        env.observation_space'''
        delta = np.infty
        i = 0

        self.counter += 1
        while delta > self.tol:
            delta = 0
            i += 1
            for s in range(self.env.observation_space.n):
                V_old = self.V[s]
                self.V[s] = self._getvalue(s, self.pi[s])
                delta = max(delta, np.abs(V_old - self.V[s]))
        print('Policy evaluation #{} completed in {} steps.'.format(self.counter, i))
        if self.bVerbose:
            print('V: {}\n'.format(self.V))
        self.iterate()
        print('***** Policy iteration completed in {} steps.'.format(self.counter))
        return self.V, self.pi

    def iterate(self):
        '''Iterates policy evaluation to find an optimal policy and
        optimal value. The algorithm keeps updating the policy until
        it finds a stable policy that cannot be further improved (according
        to the defined tolerance).

        list of outputs:
        V:      optimal value of the policy for each state s in
                env.observation_space
        pi:     the optimal action for each state s'''

        stable = True
        for s in range(self.env.observation_space.n):
            old_action = self.pi[s]

            values = []
            for action in range(self.env.action_space.n):
                values.append(self._getvalue(s, action))

            self.pi[s] = np.argmax(values)
            # self.pi[s] = np.argmax([self._getvalue(s, action) for action in range(self.env.action_space.n)])

            if self.bVerbose:
                print('state {} : {}'.format(s, values))

            if old_action != self.pi[s]:
                stable = False

        if self.bVerbose:
            print('pi: {}'.format(self.pi)) 

        if not stable:
            self.eval()

    def value_iteration(self):
        '''Returns an estimation of the optimal policy by performing
        only one sweep (one update of each state) of policy evaluation.

        output: an estimation of the optimal policy'''

        training_episodes = args.training_episodes
        testing_episodes = args.testing_episodes
        test_every = args.test_every

        delta = np.infty
        i = 0

        rewards = []
        steps = []

        while delta > self.tol:
            delta = 0
            i += 1
            for s in range(self.env.observation_space.n):
                V_old = self.V[s]

                # self.V[s] = np.max([self._getvalue(s, action) for action in range(self.env.action_space.n)])
                values = []
                for action in range(self.env.action_space.n):
                    values.append(self._getvalue(s, action))

                self.V[s] = np.max(values)

                if self.bVerbose:
                    print('state {} : {}'.format(s, values))

                delta = max(delta, np.abs(V_old - self.V[s]))

            if self.bVerbose:
                print('Step: {} V: {}'.format(i, self.V))

            # run tests
            if i % test_every == 0:
                for s in range(self.env.observation_space.n):
                    self.pi[s] = np.argmax(
                        [self._getvalue(s, action) for action in range(self.env.action_space.n)])

                reward, num_steps = self.test(testing_episodes)
                rewards.append(reward)
                steps.append(num_steps)

        for s in range(self.env.observation_space.n):
            self.pi[s] = np.argmax([self._getvalue(s, action) for action in range(self.env.action_space.n)])

        # test again one last time
        reward, num_steps = self.test(testing_episodes)
        rewards.append(reward)
        steps.append(num_steps)

        print('***** Finished value iteration in {} steps.'.format(i))

        return self.V, self.pi, rewards, steps

    def test(self, num_episodes=5, max_steps=None, render=False):

        cumulative_reward = np.zeros(num_episodes)
        num_steps = np.zeros(num_episodes)
        if max_steps is None:
            max_steps = np.infty

        for episode in range(num_episodes):
            num_steps[episode] = 0
            observation = self.env.reset()

            if render:
                print('=' * 80)
                print('Episode {}'.format(episode))
                print('=' * 80)
                self.env.render()

            done = False

            power_gamma = 1
            while not done and num_steps[episode] < max_steps:
                num_steps[episode] += 1
                action = self.pi[observation]
                observation, reward, done, info = self.env.step(action)
                cumulative_reward[episode] += reward * power_gamma
                power_gamma = power_gamma * self.gamma

                if render:
                    print('Step {}: reward {}, cumulative reward: {}'.format(
                        num_steps[episode],
                        reward,
                        cumulative_reward[episode]))
                    print('-' * 80)
                    self.env.render()

                if done and reward <= 0:
                    num_steps[episode] = NB_STEPS_IF_FALL_HOLE

            # only count steps for wins
            # if reward <= 0:
            #     num_steps[episode] = np.nan

        #return np.mean(cumulative_reward), np.nanmean(num_steps)
        return cumulative_reward, num_steps

    def _getvalue(self, state, action):
        '''For a given state and action, returns the value of that
        state according to the current policy iteration.'''

        # for a given state and action, P[state][action] returns a list of
        # tuples in the form (p, s1, r, b) containing respectively the
        # probability, state, return and boolean for all possible states s1
        # originating from s. The boolean determines if the state s1 is
        # terminal or not.
        p, s1, r, _ = zip(*self.env.P[state][action])

        # convert tuples to arrays
        p = np.array(p)
        s1 = np.array(s1, dtype=int)
        r = np.array(r)
        # b = np.array(b, dtype=bool)

        return np.sum(p * (r + self.gamma * self.V[s1]))


def render_policy(env, pi, num_episodes=20, max_steps=100):

    for episode in range(num_episodes):
        cumulative_reward = 0
        i = 0
        print('=' * 80)
        print('Episode {}'.format(episode))
        print('=' * 80)
        env.render()
        observation = env.reset()
        # for t in range(max_steps):
        done = False
        while not done:
            i += 1
            action = pi[observation]
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            print('Step {}: reward {}, cumulative reward: {}'.format(i, reward, cumulative_reward))
            print('-' * 80)
            env.render()

# #############################################################################
#
# Main
#
# #############################################################################
def run(n_runs, policy, bValueIteration=False):

    rewards = []
    steps = []

    env = policy.env
    gamma = policy.gamma
    bVerbose = policy.bVerbose
    tol = policy.tol

    for run in range(n_runs):
        np.random.seed(np.random.randint(0, 2**32 - 1))
        policy.__init__(env, gamma, bVerbose, tol)
        print('Run {}'.format(run + 1))

        V, pi, reward, num_steps = one_run(policy, bValueIteration)
        rewards.append(reward)
        steps.append(num_steps)

    rewards= np.array(rewards)
    steps = np.array(steps)
    #print("rewards: ", rewards)
    #print(rewards.shape)
    #print("steps: ", steps)
    #print(steps.shape)
    #plot(train)


    #plot(test)
    plot2("Test plots", rewards, steps)





def one_run(policy, bValueIteration=False):

    if bValueIteration:
        V, pi, rewards, steps = policy.value_iteration()
    else:
        V, pi, rewards, steps = policy.eval()

    return V, pi, rewards, steps


args = get_arguments()

def main():

    # sets the seed for random experiments
    np.random.seed(args.seed)

    # sets the environment
    env = gym.make(args.env)
    env.reset()

    pol = Policy(env,
                 gamma=args.gamma,
                 bVerbose=args.verbose,
                 tol=args.tol)
    # if args.value_iteration:
    #     V, pi = pol.value_iteration()

    # else:
    #     V, pi = pol.eval()
    # print('V: {}\n\npi:{}'.format(V, pi))

    run(args.runs, pol, args.value_iteration)

    if args.render_policy:
        render_policy(env, pi, num_episodes=1)

    env.close()


if __name__ == '__main__':
    main()
