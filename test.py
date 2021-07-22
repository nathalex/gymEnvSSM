import os
import SSM
import supersuit as ss
import GershbergSaxton as GS
import pickle

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