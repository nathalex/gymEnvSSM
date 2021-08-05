import os
import SSM
import supersuit as ss
import GershbergSaxton as GS
import pickle
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
import time

# TODO: training

phase_maps = []
new_data = False #whether or not the GS data needs updating

if os.path.exists('GSTestData/testdata.data') and not new_data:
    with open('GSTestData/testdata.data', 'rb') as filehandle:
        # read the data as binary data stream
        phase_maps = pickle.load(filehandle)

else:
    phase_maps.append(GS.main('GSTestData/UCL16.png', 16, 50))
    phase_maps.append(GS.main('GSTestData/U16.png', 16, 50))
    phase_maps.append(GS.main('GSTestData/C16.png', 16, 50))
    phase_maps.append(GS.main('GSTestData/L16.png', 16, 50))
    phase_maps.append(GS.main('GSTestData/UCLport16.png', 16, 50))
    phase_maps.append(GS.main('GSTestData/portico16.png', 16, 100))
    with open('GSTestData/testdata.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(phase_maps, filehandle)

phase_maps = tuple(phase_maps)

env = SSM.parallel_env(n_elements=256, local_ratio=0, time_penalty=-0.1, continuous=True, phasemaps=phase_maps, max_cycles=125)
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env,x_size=84, y_size=84, linear_interp=True)
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v0(env)
env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class='stable_baselines3')

#random, untrained model
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()


#train model here

# model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168,
#             learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99,
#             n_epochs=5, clip_range=0.3, batch_size=256)
# model.learn(total_timesteps=2000000)
# model.save("policy")
#
# env = SSM.env()
# env = ss.color_reduction_v0(env, mode='B')
# env = ss.resize_v0(env, x_size=84, y_size=84)
# env = ss.frame_stack_v1(env, 3)
#
# model = PPO.load("policy")
#
# pause = input("Press any key to continue\n")
#
# env.reset()
# for agent in env.agent_iter():
#    time.sleep(0.01)
#    obs, reward, done, info = env.last()
#    act = model.predict(obs, deterministic=True)[0] if not done else None
#    env.step(act)
#    env.render()
# env.close()
#
