import numpy as np
from gym import Env
from rlgym_ppo import Learner


def run(env: Env, agent: Learner, n_steps: int = -1):
    obs = env.reset()

    i = 0

    while i < n_steps:
        actions, _ = agent.agent.policy.get_action(obs)
        actions = actions.numpy().astype(np.float32)
        actions = actions.reshape((*actions.shape, 1))
        obs, reward, terminated, truncated, _ = env.step(actions)

        if terminated or truncated:
            obs = env.reset()

        i += 1

    env.close()



