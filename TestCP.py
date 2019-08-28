import gym
from CartPole import DQNSolver
import numpy as np

env = gym.make("CartPole-v1")
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
model = DQNSolver(observation_space, action_space)
model.load('CartPoleWeights.h5')
print("good so far")

state = env.reset()
state = np.reshape(state, [1, observation_space])
step = 0
for _ in range(1000):
	step += 1
	env.render()
	state = np.reshape(state, [1, observation_space])
	action = model.act(state)
	state, _, terminal, _ = env.step(action)
	if terminal:
		print("Steps: " + str(step))
		break