import argparse
import logging
import random
import time
import gym
import numpy as np
import lbforaging
from lbforaging.agent import Agent
from random import randint

logger = logging.getLogger(__name__)


def _game_loop(env, agent, agent1, render):
    fire_limit = env.max_fire + 1
    """
    """
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
        action = agent.action()
        action1 = agent1.action()
        
        nobs, nreward, ndone, _ = env.step([action,action1])
        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.1)

        done = np.all(ndone)
        #done = False
    # print(env.players[0].score, env.players[1].score)

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
    agent = RandomAgent(6)
    agent1 = RandomAgent(6)

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
