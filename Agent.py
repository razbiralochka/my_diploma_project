import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from collections import deque
import numpy as np

loss_object = tf.keras.losses.Huber()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)


class DQN(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(5, activation='linear')
        self.d2 = Dense(50, activation='relu')
        self.d3 = Dense(300, activation='linear')
        self.d4 = Dense(10, activation='relu')
        self.d5 = Dense(3, activation='linear')

    def call(self, x):
        x = tf.reshape(x, [1, 5])
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return x


class Agent():
    def __init__(self):
        self.HAL9000 = DQN()
        self.HAL9000_target = DQN()
        self.memory = deque(maxlen=3000)
        self.epsilon = 0
        self.th = 0
        self.best = np.pi

    def select_action(self, state):
        if state[0] > self.best:
            self.best = state[0]
        predictions = self.HAL9000(state, training=False).numpy()
        res = np.argmax(predictions[0])

        if np.random.rand() < 0.89 * np.exp(-100 * (state[0] - self.best) ** 2) + 0.01:
            res = random.randint(0, 2)

        if abs(state[3]) > 1.7:
            res = round(-np.sign(state[3])) + 1

        return res

    def remember(self, state, action, state_, reward, done):
        self.memory.append((state, action, state_, reward, done))

    @tf.function
    def train_step(self, state, target):
        with tf.GradientTape() as tape:
            predictions = self.HAL9000(state, training=True)
            loss = loss_object(target, predictions)
        gradients = tape.gradient(loss, self.HAL9000.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.HAL9000.trainable_variables))
        train_loss(loss)
        train_accuracy(target, predictions)

    def target_train(self):
        tau = 0.95
        weights = self.HAL9000.get_weights()
        target_weights = self.HAL9000_target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
        self.HAL9000_target.set_weights(target_weights)

    def learn(self):
        if self.epsilon > 0.1:
            self.epsilon *= 0.95
        batch_size = 200
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for state, action, state_, reward, done in samples:

            target = self.HAL9000_target(state, training=False).numpy()
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.HAL9000_target(state_, training=False).numpy()[0])
                target[0][action] = reward + Q_future
            self.train_step(state, target)
