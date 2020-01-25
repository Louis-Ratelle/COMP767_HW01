import numpy as np
import gym

env = gym.make('FrozenLake-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

for i in range(env.observation_space.n):
    print('observation_space: {}'.format(i))
print('action_space: {}'.format(env.action_space))
env.close()


class Policy():
    def __init__(self, env, tol=1e-6):
        self.env = env
        self.tol = tol

        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise NotImplementedError

        for s in range(env.observation_space.n):
            self.V[s] = 0
            self.pi[s] = env.action_space.sample()
            self.p[s1, r, s, a] = 0        # TODO

    def eval(self):
        '''Evaluates the policy value for each state s in
        env.observation_space'''
        delta = np.infty

        while delta > self.tol:
            delta = 0
            for s in range(self.env.observation_space.n):
                V_old = self.V[s]
                self.V[s] = np.sum(self.p[s1, r, s, self.pi[s]] * (r + gamma * V(s1))) for (s1, r) in       # TODO
                delta = max(delta, np.abs(V_old - self.V[s]))

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
            self.pi[s] = 0           # TODO
            if old_action != self.pi[s]:
                stable = False
        if stable:
            return self.V, self.pi
        else:
            self.eval()

    def value_iteration(self):
        '''Returns an estimation of the optimal policy by performing
        only one sweep (one update of each state) of policy evaluation.

        output: an estimation of the optimal policy'''
        self.eval()

        for s in range(self.env.observation_space.n):
            self.pi[s] = np.argmax()     # TODO

        return self.pi
