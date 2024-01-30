import cloudpickle
import numpy as np
import multiprocessing as mp

class RolloutBuffer:
    def __init__(self):
        self.rewards = []

    def on_rollout_end(self):
        final_rewards = np.mean(self.rewards)
        return final_rewards


def _process_target(remote: mp.connection.Connection, parent_remote: mp.connection.Connection, env):
    parent_remote.close()
    env = cloudpickle.load(env)()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                if done:
                    # save final observation where user can get it, then reset
                    info["terminal_observation"] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
