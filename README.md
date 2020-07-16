![NTU logo](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/ntu_logo.png)

Nanyang Technological University, Singapore

School of Computer Science and Engineering(SCSE)
___

### Final Year Project: SCE17-0434
# Reinforcement Learning for Self-Driving Cars
___

- This project is a Final Year Project carried out by **Ho Song Yan** from Nanyang Technological University, Singapore.
- Download here: [FYP2018-Final-Report.pdf](https://github.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/blob/master/docs/FYP2018-Final%20Report-HO%20SONG%20YAN.pdf)
- Please use this identifier to cite or link to this item: https://hdl.handle.net/10356/74098

___

# Abstract

This project implements reinforcement learning to generate a self-driving car-agent with deep learning network to maximize its speed. The convolutional neural network was implemented to extract features from a matrix representing the environment mapping of self-driving car. The model acts as value functions for five actions estimating future rewards. The model is trained under Q-learning algorithm in a simulation built to simulate traffic condition of seven-lane expressway. After continuous training for 2340 minutes, the model learns the control policies for different traffic conditions and reaches an average speed 94 km/h compared to maximum speed of 110 km/h.

# Simulator

![screenshot](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/screenshot.png)
*Simulator running under macOS High Sierra environment*

## Requirements
- Tensorflow
- pygame
- NumPy
- PIL

# Model architecture

![architecture](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/network_architecture.png)
*High level model architecture design*

![tensorflow structure](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/graph_structure.png)
*Graph structure in Tensorflow*

# Results

## Average speed

![chart](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/average_speed_training.png)
*Average speed against number of training episode*

## Score

![chart](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/score_training.png)
*Score against number of training episode*

## Training loss

![chart](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/loss_training.png)
*Loss against number of training episode*

## Sum of Q-values

![chart](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/sum_of_q_values_training.png)
*Sum of Q-values against number of training episode*

# Evaluation

| Traffic condition | Description |
|:--|:---|
|1. Light traffic|Maximum 20 cars are simulated with plenty room for overtaking.|
|2. Medium traffic|Maximum 40 cars are simulated with lesser chance to overtake other cars.|
|3. Heavy traffic|Maximum 60 cars are simulated to simulate heavy traffic. |

![chart](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/speed_brake_test.png)
*Condition 1: Average speed against average number of emergency brake applied*

![chart](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/speed_brake_test_2.png)
*Condition 2: Average speed against average number of emergency brake applied*

![chart](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/speed_brake_test_3.png)
*Condition 3: Average speed against average number of emergency brake applied*

# Biblography

1.	Sallab, A.E., Abdou, M., Perot, E., and Yogamani, S.: ‘Deep reinforcement learning framework for autonomous driving’, Electronic Imaging, 2017, 2017, (19), pp. 70-76
2.	Sutton, R.S.: ‘Learning to predict by the methods of temporal differences’, Machine learning, 1988, 3, (1), pp. 9-44
3.	Bellemare, M.G., Veness, J., and Bowling, M.: ‘Investigating Contingency Awareness Using Atari 2600 Games’, in Editor (Ed.)^(Eds.): ‘Book Investigating Contingency Awareness Using Atari 2600 Games’ (2012, edn.), pp. 
4.	Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., and Riedmiller, M.: ‘Playing atari with deep reinforcement learning’, arXiv preprint arXiv:1312.5602, 2013
5.	Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., Jackel, L.D., Monfort, M., Muller, U., and Zhang, J.: ‘End to end learning for self-driving cars’, arXiv preprint arXiv:1604.07316, 2016
6.	Chen, C., Seff, A., Kornhauser, A., and Xiao, J.: ‘Deepdriving: Learning affordance for direct perception in autonomous driving’, in Editor (Ed.)^(Eds.): ‘Book Deepdriving: Learning affordance for direct perception in autonomous driving’ (2015, edn.), pp. 2722-2730
7.	Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., and Ostrovski, G.: ‘Human-level control through deep reinforcement learning’, Nature, 2015, 518, (7540), pp. 529-533
8.	Yu, A., Palefsky-Smith, R., and Bedi, R.: ‘Deep Reinforcement Learning for Simulated Autonomous Vehicle Control’, Course Project Reports: Winter, 2016, pp. 1-7

