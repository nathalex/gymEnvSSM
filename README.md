# gymEnvSSM:
## A Gym environment simulating Spatial Sound Modulation

This environment simulates a segmented SSM (here, using a phased array of transducers or PATS) in order to optimise coalition formation of the elements using multi-agent reinforcement learning.

| [<img src="/ReadMeImgs/MARL.png" width="430" title="Multi-Agent Reinforcement Learning"/>](/ReadMeImgs/MARL.png)| [<img src="/ReadMeImgs/PATS.gif" width="455" title="Phased Array of Transducers"/>](/ReadMeImgs/PATS.gif)|
|:--------|:----------|
| *Multi-Agent Reinforcement Learning* | *Phased Array of Transducers* |

Using the Gershberg-Saxton (GS) algorithm to get phasemaps as input, generated from test data (see https://dl.acm.org/doi/abs/10.1145/3386569.3392492),
the goal is to optimise coalition formation for a set of images so the number of coalitions is low but the resolution of the produced images is still high.

[<img src="/ReadMeImgs/SSMimage.jpg" width="400" title="SSM with output phasemap"/>](ReadMeImgs/SSMimage.jpg)

*SSM with output phasemap*

In this case, the simulations are run on a 16x16 matrix in order to save time and for the legibility of the visualisations.

| [<img src="/ReadMeImgs/SSM256.jpg" width="430" title="SSM with 256 elements"/>](/ReadMeImgs/SSM256.jpg)| [<img src="/ReadMeImgs/coalitions.jpg" width="430" title="Possible coalitions for 1024 elements"/>](/ReadMeImgs/coalitions.jpg)|
|:--------|:------------|
| *SSM with 256 elements* | *Possible coalitions for 1024 elements* |

The coalitions are visualised by numbers on each agent (the number refers to the coalition the agent belongs to).
For example, for test image C, we can see each agent is initially in its own independent coalition.

|[<img src="/ReadMeImgs/C.png" width="500" title="Test image C"/>](/ReadMeImgs/C.png)| [<img src="/ReadMeImgs/CGS.png" width="500" title="Gershberg-Saxton output: phasemap for test image C"/>](/ReadMeImgs/CGS.png)|
|:--------|:------------|
| *Test image C* | *Gershberg-Saxton output: phasemap for test image C* |
|[<img src="/ReadMeImgs/Coutput.png" width="500" title="gymEnvSSM visualisation for initial set of coalitions, where each agent is independent"/>](/ReadMeImgs/Coutput.png)|(Place post-training results for C here)|
|*gymEnvSSM visualisation for initial set of coalitions*|*Caption*|

This project is inspired by PettingZoo's Pistonball library (see https://towardsdatascience.com/multi-agent-deep-reinforcement-learning-in-15-lines-of-code-using-pettingzoo-e0b963c0820b).

[<img src="/ReadMeImgs/MARLtut.gif" width="600" title="Pistonball tutorial results"/>](/ReadMeImgs/MARLtut.gif)

*Pistonball tutorial results*