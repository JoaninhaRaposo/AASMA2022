import argparse
import logging
import random
import time
import gym
import numpy as np

import math
from scipy.spatial.distance import cityblock
from enum import Enum

import lbforaging
from lbforaging.agent import Agent
from random import randint
from typing import List, Tuple
from lbforaging.utils import compare_results

logger = logging.getLogger(__name__)

N_ACTIONS = 6
NONE, NORTH, SOUTH, WEST, EAST, FIGHT = range(N_ACTIONS)

def _game_loop(env, agents, render, n_episodes):
    results = np.zeros(n_episodes)

    for episode in range(n_episodes):
        if episode % 5 == 0:
            print(f"Episode {episode} out of {n_episodes} completed.")

        steps = 0
        obs = env.reset()
        done = False

        if render:
            env.render()
            time.sleep(0.5)

        while not done:
            steps += 1

            actions = []
            for agent in agents:
                agent.see(obs)
                if agent.hasEnv:
                    actions.append(agent.action(env))
                else:
                    actions.append(agent.action())
            
            # print(actions)
            nobs, nreward, ndone, _ = env.step(actions)

            obs = nobs
            if sum(nreward) > 0:
                print(nreward)

            if render:
                env.render()
                time.sleep(0.1)

            done = np.all(ndone)
            #done = False
        results[episode] = steps

    return results
        # print(env.players[0].score, env.players[1].score)

class CoordAgent(Agent):
    
    """
    A baseline agent for the SimplifiedPredatorPrey environment.
    The greedy agent finds the nearest prey and moves towards it.
    """

    def __init__(self, size, agent_id, n_agents):
        super(CoordAgent, self).__init__(f"Coord Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.hasEnv = 1
        self.size = size

    def action(self, env) -> int:
        obs = [int(x) for x in list(env._make_gym_obs()[0][0])]

        agents_positions = obs[(len(obs) - self.n_agents * 3):]
        prey_positions = obs[:(len(obs) - self.n_agents * 3)]

        fire_positions = []
        for ele in range(0,len(prey_positions)):
            if ele % 3 != 2:
                fire_positions.append(prey_positions[ele])
                
        agent_position = agents_positions[self.agent_id * 3], agents_positions[(self.agent_id * 3) + 1]
        # print("Agents", agents_positions)
        # print("Fires", prey_positions)
        closest_prey = self.closest_prey(agent_position, fire_positions, self.agent_id, self.size)
        prey_found = closest_prey is not None
        # print(closest_prey)
        return self.direction_to_go(agent_position, closest_prey) if prey_found else random.randrange(N_ACTIONS)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, prey_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if abs_distances[1] > abs_distances[0]:
            return self._close_horizontally(distances)
        elif abs_distances[1] < abs_distances[0]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def closest_prey(self, agent_position, prey_positions, agent_id, size):
        """
        Given the positions of an agent and a sequence of positions of all prey,
        returns the positions of the closest prey
        """
        min = math.inf
        closest_prey_position = None
        n_preys = int(len(prey_positions) / 2)
        for p in range(n_preys):
            prey_position = prey_positions[p * 2], prey_positions[(p * 2) + 1]
            distance = cityblock(agent_position, prey_position)
            if prey_position == (-1, -1):
                continue
            if agent_id == 0:
                if distance < min and prey_position[0] < size//2:
                    min = distance
                    closest_prey_position = prey_position
            else:
                if distance < min and prey_position[0] >= size // 2:
                    min = distance
                    closest_prey_position = prey_position
        return closest_prey_position

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if np.absolute(distances[0]) + np.absolute(distances[1]) < 2:
            return FIGHT
        if distances[1] > 0:
            return EAST
        elif distances[1] < 0:
            return WEST
        else:
            return NONE

    def _close_vertically(self, distances):
        if np.absolute(distances[0]) + np.absolute(distances[1]) < 2:
            return FIGHT
        if distances[0] > 0:
            return SOUTH
        elif distances[0] < 0:
            return NORTH
        else:
            return NONE

class GreedyAgent(Agent):
    
    """
    A baseline agent for the SimplifiedPredatorPrey environment.
    The greedy agent finds the nearest prey and moves towards it.
    """

    def __init__(self, agent_id, n_agents):
        super(GreedyAgent, self).__init__(f"Greedy Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = N_ACTIONS
        self.hasEnv = 1

    def action(self, env) -> int:
        obs = [int(x) for x in list(env._make_gym_obs()[0][0])]

        agents_positions = obs[(len(obs) - self.n_agents * 3):]
        prey_positions = obs[:(len(obs) - self.n_agents * 3)]

        fire_positions = []
        for ele in range(0,len(prey_positions)):
            if ele % 3 != 2:
                fire_positions.append(prey_positions[ele])
                
        agent_position = agents_positions[self.agent_id * 3], agents_positions[(self.agent_id * 3) + 1]
        closest_prey = self.closest_prey(agent_position, fire_positions)
        prey_found = closest_prey is not None
        return self.direction_to_go(agent_position, closest_prey) if prey_found else random.randrange(N_ACTIONS)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, prey_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        distances = np.array(prey_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)

        if abs_distances[1] > abs_distances[0]:
            return self._close_horizontally(distances)
        elif abs_distances[1] < abs_distances[0]:
            return self._close_vertically(distances)
        else:
            roll = random.uniform(0, 1)
            return self._close_horizontally(distances) if roll > 0.5 else self._close_vertically(distances)

    def closest_prey(self, agent_position, prey_positions):
        """
        Given the positions of an agent and a sequence of positions of all prey,
        returns the positions of the closest prey
        """
        min = math.inf
        closest_prey_position = None
        n_preys = int(len(prey_positions) / 2)
        for p in range(n_preys):
            prey_position = prey_positions[p * 2], prey_positions[(p * 2) + 1]
            distance = cityblock(agent_position, prey_position)
            if prey_position == (-1, -1):
                continue
            if distance < min:
                min = distance
                closest_prey_position = prey_position
        return closest_prey_position

    # ############### #
    # Private Methods #
    # ############### #

    def _close_horizontally(self, distances):
        if np.absolute(distances[0]) + np.absolute(distances[1]) < 2:
            return FIGHT
        if distances[1] > 0:
            return EAST
        elif distances[1] < 0:
            return WEST
        else:
            return NONE

    def _close_vertically(self, distances):
        if np.absolute(distances[0]) + np.absolute(distances[1]) < 2:
            return FIGHT
        if distances[0] > 0:
            return SOUTH
        elif distances[0] < 0:
            return NORTH
        else:
            return NONE


class RandomAgent(Agent):
    
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__("Random Agent")
        self.n_actions = n_actions
        self.hasEnv = 0

    def action(self) -> int:
        return np.random.randint(self.n_actions)


def main(game_count=1, render=False):
    env = gym.make("Foraging-8x8-2p-2f-coop-v2")
    obs = env.reset()
    size = 8

    teams = {
        
        "Random Team": [
            RandomAgent(6),
            RandomAgent(6)
        ],
        
        "Greedy Team": [
            GreedyAgent(agent_id=0, n_agents=2),
            GreedyAgent(agent_id=1, n_agents=2)
        ],

        "Coord Team": [
            CoordAgent(size, agent_id=0, n_agents=2),
            CoordAgent(size, agent_id=1, n_agents=2)
        ]
    }

    parser.add_argument("--episodes", type=int, default=10)
    opt = parser.parse_args()

    results = {}
    for team, agents in teams.items():
        result = _game_loop(env, agents, render, opt.episodes)
        results[team] = result

    compare_results(
        results,
        title="Teams Comparison on LBForaging Environment",
        colors=["orange", "green", "blue"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
