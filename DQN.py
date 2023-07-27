import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


loss_object = tf.keras.losses.MeanSquaredError()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()


class DQN(Model):
  def __init__(self):
    super().__init__()
    self.d1 = Dense(5, activation='linear', input_shape=(5,))
    self.d2 = Dense(25, activation='relu')
    self.d3 = Dense(50, activation='relu')
    self.d4 = Dense(20, activation='relu')
    self.d5 = Dense(3)

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.d5(x)
    return x

class Agent():
    def __init__(self):
        self.HAL9000 = DQN()
        self.replay_memoty = list()
    @tf.function
    def select_action(self,state):
        predictions = self.HAL9000(state, training=False)
        res = tf.argmax(predictions)
        return predictions

    def add_to_replay(self, arg):
        self.replay_memoty.append(arg)

    @tf.function
    def train_step(self,state, target):
        with tf.GradientTape() as tape:
            predictions = self.HAL9000(state, training=True)
            loss = loss_object(target, predictions)
        gradients = tape.gradient(loss, self.HAL9000.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.HAL9000.trainable_variables))
        train_loss(loss)
        train_accuracy(target, predictions)

    def train(self):
        print("Training: ")
        train_loss.reset_states()
        train_accuracy.reset_states()
        for line in self.replay_memoty:
            state = np.array([line[0]])
            action = line[1]
            reward = line[2]
            next_state = np.array([line[3]])
            target = reward + 0.99 * np.amax(self.HAL9000(next_state)[0])
            target_f = self.HAL9000(state).numpy()
            target_f[0][action] = target
            self.train_step(state, target_f)


        self.replay_memoty.clear()