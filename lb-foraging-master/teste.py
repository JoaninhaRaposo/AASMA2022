"""import lbforaging
from gym.envs.registration import register

env = gym.make("Foraging-8x8-2p-1f-v2")


register(
    id="Foraging-{0}x{0}-{1}p-{2}f{3}-v2".format(s, p, f, "-coop" if c else ""),
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": p,
        "max_player_level": 3,
        "field_size": (s, s),
        "max_food": f,
        "sight": s,
        "max_episode_steps": 50,
        "force_coop": c,
    },
)"""

import distutils.spawn

import lbforaging
import gym

env = gym.make("Foraging-8x8-2p-1f-v2")
observation, x = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset()
env.close()
