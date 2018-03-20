![NTU logo](http://www.ntu.edu.sg/home/sachin.mishra/img/logo.png)

# Nanyang Technological University, Singapore
# School of Computer Science and Engineering(SCSE)
___
# Final Year Project
# SCE17-0434
# Reinforcement Learning for Self-Driving Cars
___

*This project is a Final Year Project carried out by **Ho Song Yan** from Nanyang Technological University, Singapore.*

___

# Abstract

This project implements reinforcement learning to generate a self-driving car-agent with deep learning network to maximize its speed. The convolutional neural network was implemented to extract features from a matrix representing the environment mapping of self-driving car. The model acts as value functions for five actions estimating future rewards. The model is trained under Q-learning algorithm in a simulation built to simulate traffic condition of seven-lane expressway. After continuous training for 2340 minutes, the model learns the control policies for different traffic conditions and reaches an average speed 94 km/h compared to maximum speed of 110 km/h.

# Simulator

![screenshot](https://raw.githubusercontent.com/songyanho/Reinforcement-Learning-for-Self-Driving-Cars/master/images/screenshot.png)
*Simulator running under macOS High Sierra environment*

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

