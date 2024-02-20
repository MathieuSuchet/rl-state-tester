import numpy as np
from stable_baselines3 import PPO

from rl_state_tester.ui.ui_handling import UIHandler
from rl_state_tester.utils.envs import HarvestableEnv
from rl_state_tester.utils.orchestrator import Observer, Distributor, Orchestrator


def run(env: HarvestableEnv, agent: PPO, callbacks, n_steps: int = -1, agent_tick_skip: int = 8, with_ui: bool = True):
    obs = env.reset()
    t = 0
    current_actions = None

    others = [
        Observer(env)
    ]
    handler = None
    if with_ui:
        handler = UIHandler(callbacks)
        handler.serve()

        # Dependency injection I guess ?
    distrib = Distributor(np.array(callbacks).copy().tolist(), others)

    all_commands = []
    for h in callbacks:
        all_commands.extend(h.commands.commands)

    for h in distrib.others:
        if hasattr(h, "commands"):
            all_commands.extend(h.commands.commands)

    Orchestrator(
        commands=all_commands
    )

    while n_steps < 0 or t < n_steps:
        obs = np.array(obs)
        predicted_actions = agent.predict(obs)[0]
        predicted_actions = predicted_actions.reshape((*predicted_actions.shape, 1))
        if t % agent_tick_skip == 0:
            current_actions = predicted_actions
        obs, reward, terminal, info = env.step(current_actions)

        if terminal:
            obs = env.reset()

        t += 1

    env.close()
    if handler:
        handler.quit()

