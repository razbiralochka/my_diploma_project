from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class DQN1(Model):
  def __init__(self):
    super().__init__()
    self.d1 = Dense(5, activation='linear')
    self.d2 = Dense(128, activation='relu')
    self.d3 = Dense(3)

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    return x

class Agent():
    def __init__(self):
        networ = DQN1