# gymEnvSSM:
## A Gym environment simulating Spatial Sound Modulation

This environment simulates a segmented SSM (here a phased array of transducers, or PATS) in order to optimise coalition formation of the elements using multi-agent reinforcement learning.

![MARL](/ReadMeImgs/MARL.png "Multi-Agent Reinforcement Learning")
![PATS](/ReadMeImgs/PATS.gif "Phased Array of Transducers")

Using the Gershberg Saxton algorithm to get phasemaps as input, generated from test data (see https://dl.acm.org/doi/abs/10.1145/3386569.3392492),
the goal is to optimise coalition formation for a set of images so the number of coalitions is low but the resolution of the produced images is still high.

In this case, the simulations are run on a 16x16 matrix in order to save time and for the legibility of the visualisations.

![SSM 256](/ReadMeImgs/SSM256.jpg "SSM with 256 elements")
![SSM example groupings](/ReadMeImgs/coalitions.jpg "Possible coalitions for 1024 elements")

The coalitions are visualised by numbers on each agent (the number refers to the coalition the agent belongs to).
For example, for test image C, we can see each agent is initially in its own independent coalition.

![C](/ReadMeImgs/C.png "Test image C")
![C GS](/ReadMeImgs/CGS.png "Gershberg Saxton output: phasemap for test image C")
![initial coalitions](/ReadMeImgs/Coutput.png "gymEnvSSM visualisation for initial set of coalitions, where each agent is independent")

(Place post-training results for C here)

This project is inspired by PettingZoo's Pistonball library (see https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b).
![pistonball](/ReadMeImgs/MARLtut.gif "Pistonball tutorial results")