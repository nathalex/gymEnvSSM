import manual_control
from gymEnvSSM import env
import SSM
import supersuit as ss
import gym

env = SSM.parallel_env(n_elements=256, local_ratio=0, time_penalty=-0.1, continuous=True, max_cycles=125)
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env,x_size=84, y_size=84, linear_interp=True)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class='stable_baselines3')

#train model here

# env = SSM.env()
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()