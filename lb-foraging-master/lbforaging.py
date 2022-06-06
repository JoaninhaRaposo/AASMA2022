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

logger = logging.getLogger(__name__)

N_ACTIONS = 6
NONE, NORTH, SOUTH, WEST, EAST, FIGHT = range(N_ACTIONS)

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    FIGHT = 5

def _game_loop(env, agent, agent1, render):
    fire_limit = env.max_fire + 1

    obs = env.reset()
    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        
        
        if 5 == randint(0,20) and fire_limit >= env._fire_spawned:
            r = random.choice([0,1])
            if r == 1:
                env.spawn_fires(2, max_level=2)
            else:
                env.spawn_big_fires(2)
        

        actions = env.action_space.sample()

        agent.see(obs)
        agent1.see(obs)

        action = agent.action(env)
        action1 = agent1.action(env)
        
        nobs, nreward, ndone, _ = env.step([action, action1])

        obs = nobs
        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.1)

        done = np.all(ndone)
        #done = False
    # print(env.players[0].score, env.players[1].score)

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

    def action(self, env) -> int:
        obs = [int(x) for x in list(self.observation[0])]
        print(obs)

        agents_positions = obs[(self.n_agents * 3):]
        prey_positions = obs[:self.n_agents * 3]

        fire_positions = []
        for ele in range(0,len(prey_positions)):
            if ele % 3 != 2:
                fire_positions.append(prey_positions[ele])
                
        agent_position = agents_positions[self.agent_id * 3], agents_positions[(self.agent_id * 3) + 1]
        # print("Agents", agent_position)
        # print("Fires", fire_positions)
        closest_prey = self.closest_prey(agent_position, fire_positions)
        prey_found = closest_prey is not None
        # print(closest_prey)
        a = self.direction_to_go(agent_position, closest_prey)
        print(Action(a))
        return a if prey_found else random.randrange(N_ACTIONS)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def direction_to_go(self, agent_position, prey_position):
        """
        Given the position of the agent and the position of a prey,
        returns the action to take in order to close the distance
        """
        print(agent_position)
        print(prey_position)
        distances = np.array(prey_position) - np.array(agent_position)
        abs_distances = np.absolute(distances)
        print("d", distances)
        print("a", abs_distances)
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

    def action(self) -> int:
        return np.random.randint(self.n_actions)


def main(game_count=1, render=False):
    env = gym.make("Foraging-8x8-2p-2f-coop-v2")
    obs = env.reset()

    print(env.action_space.sample())
    agent = GreedyAgent(0, 2)
    agent1 = GreedyAgent(1, 2)

    #agent = RandomAgent(6)
    #agent1 = RandomAgent(6)

    for episode in range(game_count):
        _game_loop(env, agent, agent1, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
