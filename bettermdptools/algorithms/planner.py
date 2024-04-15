"""
Author: Miguel Morales
BSD 3-Clause License

Copyright (c) 2018, Miguel Morales
All rights reserved.
https://github.com/mimoralea/gdrl/blob/master/LICENSE
"""

"""
modified by: John Mansfield

documentation added by: Gagandeep Randhawa
"""

"""
Model-based learning algorithms: Value Iteration and Policy Iteration

Assumes prior knowledge of the type of reward available to the agent
for iterating to an optimal policy and reward value for a given MDP.
"""

import time
import warnings

import numpy as np

from bettermdptools.utils.convergence_function import UtilityConvergenceFunction, get_max_value_less_than_theta
from bettermdptools.utils.decorators import print_runtime


class Planner:
    def __init__(self, P):
        self.P = P

    @print_runtime
    def value_iteration(
            self,
            gamma=1.0,
            number_of_iterations=1000,
            convergence_function: UtilityConvergenceFunction = get_max_value_less_than_theta(1e-10),
        ):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        number_of_iterations {int}:
            Number of iterations

        convergence_function {UtilityConvergenceFunction}:
            Function that takes in the past and current values of the utility function and returns a boolean indicating
            if you have converged.  This is useful for iterative algorithms like value iteration and policy iteration.

        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((number_of_iterations, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        while i < number_of_iterations-1 and not converged:
            i += 1
            Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
            for s in range(len(self.P)):
                for a in range(len(self.P[s])):
                    for prob, next_state, reward, done in self.P[s][a]:
                        Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
            next_v = np.max(Q, axis=1)
            converged = convergence_function(V, next_v)
            V = next_v
            V_track[i] = V
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check theta and n_iters.  ")

        pi = {s:a for s, a in enumerate(np.argmax(Q, axis=1))}
        return V, V_track, pi

    @print_runtime
    def policy_iteration(
            self,
            gamma=1.0,
            number_of_iterations=50,
            convergence_function: UtilityConvergenceFunction = get_max_value_less_than_theta(1e-10),
            stats = False
        ):
        """
        PARAMETERS:

        gamma {float}:
            Discount factor

        number_of_iterations {int}:
            Number of iterations

        convergence_function {UtilityConvergenceFunction}:
            Function that takes in the past and current values of the utility function and returns a boolean indicating
            if you have converged.  This is useful for iterative algorithms like value iteration and policy iteration.


        RETURNS:

        V {numpy array}, shape(possible states):
            State values array

        V_track {numpy array}, shape(n_episodes, nS):
            Log of V(s) for each iteration

        pi {lambda}, input state value, output action value:
            Policy mapping states to actions.
        """
        random_actions = np.random.choice(tuple(self.P[0].keys()), len(self.P))

        pi = {s: a for s, a in enumerate(random_actions)}
        # initial V to give to `policy_evaluation` for the first time
        V = np.zeros(len(self.P), dtype=np.float64)
        V_track = np.zeros((number_of_iterations, len(self.P)), dtype=np.float64)
        i = 0
        converged = False
        if stats:
            collected_stats = []
        while i < number_of_iterations-1 and not converged:
            if stats:
                start = time.monotonic()
            i += 1
            old_pi = pi
            V = self.policy_evaluation(pi, V, gamma, convergence_function)
            V_track[i] = V
            pi = self.policy_improvement(V, gamma)
            converged = (old_pi == pi)
            if stats:
                collected_stats.append(time.monotonic() - start)
        if not converged:
            warnings.warn("Max iterations reached before convergence.  Check n_iters.")
        if stats:
            return V, V_track, pi, collected_stats
        return V, V_track, pi

    def policy_evaluation(self, pi, prev_V, gamma=1.0, convergence_function = get_max_value_less_than_theta(1e-10)):
        while True:
            V = np.zeros(len(self.P), dtype=np.float64)
            for s in range(len(self.P)):
                for prob, next_state, reward, done in self.P[s][pi[s]]:
                    V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
            if convergence_function(prev_V, V):
                break
            prev_V = V.copy()
        return V

    def policy_improvement(self, V, gamma=1.0):
        Q = np.zeros((len(self.P), len(self.P[0])), dtype=np.float64)
        for s in range(len(self.P)):
            for a in range(len(self.P[s])):
                for prob, next_state, reward, done in self.P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        new_pi = {s: a for s, a in enumerate(np.argmax(Q, axis=1))}
        return new_pi
