# Banana Navigation Unity ML-Agents 
__Project Overview:__
This project applies Deep Reinforcement Learning on the 
[Unity ML agents](https://github.com/Unity-Technologies/ml-agents) Banana environment.

![banana](results/banana.gif)

The goal is to find yellow bananas and avoid blue bananas. It is an episodic environment, the agent has to maximize reward in a fixed number 
of steps.

The environment is considered to be solved if an agent gets an average reward of at least 13 over 100 episodes.

### Reward
A reward of +1 is provided for collecting a yellow banana, and a penalty of -1 is provided for collecting a blue banana. For other times, reward is 0.

### State space

The state space is continuous. It consists of vectors of size 37, specifying the agent's velocity and a ray-traced 
representation of the agent's local field of vision. It specifies presence of any objects under a number of fixed angles 
in front of the agent.

### Action space

The action space is discrete and consists of four options:
* go forward (0)
* go backward (1)
* go left (2)
* go right (3)


The Environment Follow the instructions below to explore the environment on your own machine! You will also learn how to use the Python API to control your agent.

__Step 1: Clone the DRLND Repository__ 

Follow the instructions in the [DRLND GitHub repository to set up your Python environment](https://github.com/udacity/deep-reinforcement-learning#dependencies) . These instructions can be found in README.md at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.


__Step 2: Download the Unity Environment__

For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip )

Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip )

Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip )

Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) 

Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.


__Step 3: Explore the Environment__
After you have followed the instructions above, open Navigation.ipynb (located in the p1_navigation/ folder in the DRLND GitHub repository) and experiment to learn how to use the Python API to control the agent.






