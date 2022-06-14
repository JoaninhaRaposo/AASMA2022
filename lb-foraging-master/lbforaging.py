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
    fires_fighted = np.zeros(n_episodes)

    for episode in range(n_episodes):
       

        steps = 0
        total_fires_fighted = 0
        obs = env.reset()
        done = False

        if render:
            env.render()
            time.sleep(1)

        while not done:
            steps += 1

            actions = []
            for agent in agents:
                agent.see(obs)
                if agent.hasEnv:
                    actions.append(agent.action(env))
                else:
                    actions.append(agent.action())
            

            nobs, nreward, ndone, _ = env.step(actions)

            obs = nobs
            

            if render:
                env.render()
                time.sleep(0.1)

            done = np.all(ndone)
            #done = False
        results[episode] = steps
        fires_fighted[episode] = env.get_total_fires_fighted()
        print(f"Episode {episode+1} out of {n_episodes} completed.")

   
    return results, fires_fighted
        
class ConventionAgent(Agent):
    
    def __init__(self, agent_id: int, n_agents: int, social_conventions: List): #social_conventions: List with [1] Agent order
        super(ConventionAgent, self).__init__(f"Convention Agent")
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.conventions = social_conventions
        self.n_actions = N_ACTIONS
        self.hasEnv = 1
        
    def action(self,env) -> int:
        
        agent_order = self.conventions  
        
        
        obs = [int(x) for x in list(env._make_gym_obs()[0][0])]
        
        agents_positions = obs[(len(obs) - self.n_agents * 3):]
        prey_positions = obs[:(len(obs) - self.n_agents * 3)]

        fire_positions = []
        for ele in range(0,len(prey_positions)):
            if ele % 3 != 2:
                fire_positions.append(prey_positions[ele])
                
        agent_position = agents_positions[self.agent_id * 3], agents_positions[(self.agent_id * 3) + 1]
        
      
        #Find closest fire to self agent
        closest_prey = self.closest_prey(agent_position, fire_positions)
        prey_found = closest_prey is not None
        #a = self.direction_to_go(agent_position, closest_prey)

        #Find the closest fire to the other agent
        if(self.agent_id == 1):
            other_id = 0
        else:
            other_id = 1
        other_agent_position = agents_positions[other_id * 3], agents_positions[(other_id * 3) + 1]
        closest_prey_to_other = self.closest_prey(other_agent_position,fire_positions)
 
        other_prey_found = closest_prey_to_other is not None
        if prey_found:  
            if other_prey_found:
                if closest_prey == closest_prey_to_other:
                    if self.agent_id == agent_order[0]: #agent have priority
                        
                        return self.direction_to_go(agent_position, closest_prey)
                    else: #does not have priority: chooses another fire
                       
                        for i in range((len(fire_positions)//2)-1):
                            if(fire_positions[2*i]==closest_prey_to_other[0] and fire_positions[2*i+1]==closest_prey_to_other[1]):
                                del fire_positions[2*i]
                                del fire_positions[2*i]
                    
                        new_closest_prey = self.closest_prey(agent_position, fire_positions)
                        new_prey_found = new_closest_prey is not None
                        if new_prey_found:
                            return self.direction_to_go(agent_position, new_closest_prey)
                        else:
                            return random.randrange(N_ACTIONS)
                else:
                    return self.direction_to_go(agent_position, closest_prey)

            else:
                return self.direction_to_go(agent_position, closest_prey)
        else:

            return random.randrange(N_ACTIONS)

        
       
    
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
            if prey_position == (-1, -1):
                continue
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

        closest_prey = self.closest_prey(agent_position, fire_positions, self.agent_id, self.size)
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
    
    agent_order = [0,1] #Random order
    random.shuffle(agent_order)
    teams = {
        
        "Random Team": [
            RandomAgent(6),
            RandomAgent(6)
        ],
        
        "Greedy Team": [
            GreedyAgent(agent_id=0, n_agents=2),
            GreedyAgent(agent_id=1, n_agents=2)
        ],

        "Coordinated Team": [
            CoordAgent(size, agent_id=0, n_agents=2),
            CoordAgent(size, agent_id=1, n_agents=2)
        ],
        "Convention Team": [
            ConventionAgent(0,2,agent_order),
            ConventionAgent(1,2,agent_order)
        ]

    }

    parser.add_argument("--episodes", type=int, default=50)
    opt = parser.parse_args()

    results = {}
    total_fires = {}
    for team, agents in teams.items():
        print("Starting {}".format(team))
        result , fires_fighted = _game_loop(env, agents, render, opt.episodes)
        results[team] = result
        total_fires[team] = fires_fighted

    compare_results(
        results,
        title="Teams Comparison on LBForaging Environment",
        colors=["purple", "green", "blue", "violet"]
    )

    compare_results(
        total_fires,
        title="Teams Comparison on LBForaging Environment",
        metric="Fires fighted in 10 steps per episode",
        colors=["purple", "green", "blue", "violet"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
