import random
import gym
from gym.spaces import Box, Discrete

import numpy as np
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json



# from scores.score_logger import ScoreLogger

# ENV_NAME = "CartPole-v1"
ENV_NAME = "MountainCar-v0"
# ENV_NAME = "CarRacing-v0"
# ENV_NAME = "Pong-ram-v0"
# ENV_NAME = "Pendulum-v0"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        print("observation_space:", observation_space)
        self.model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(32, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            h = self.model.fit(state, q_values, verbose=0)
            # # if 'loss' in h.history:
            #   # print(h.history['loss'])
            # # else:
            #   # print(h.history)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
    def save(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(ENV_NAME + "_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(ENV_NAME + "_model.h5")
        print("Saved model to disk")
    def load_from_file(self):
        if os.path.exists(ENV_NAME + '_model.json') and os.path.exists(ENV_NAME + '_model.h5'):
            json_file = open(ENV_NAME + '_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(ENV_NAME + "_model.h5")
            print("Loaded model from disk")
            self.model = loaded_model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))


def cartpole():
    env = gym.make(ENV_NAME)
    # score_logger = ScoreLogger(ENV_NAME)


    observation_space = env.observation_space.shape[0]
    # print(env.observation_space)
    print(type (env.action_space))
    if isinstance(env.action_space, Discrete):
      action_space = env.action_space.n
    else:
      print("shape ", env.action_space.dtype)
      action_space = env.action_space.shape

    dqn_solver = DQNSolver(observation_space, action_space)
    # dqn_solver.load_from_file()
    run = 0
    while True:
        # if run % 10 == 0:
        #     dqn_solver.save()
        run += 1

        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        total_reward = 0
        while True:
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            total_reward += reward
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print ("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", total_reward: " + str(total_reward))
                # score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()
