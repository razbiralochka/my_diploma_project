import tensorflow as tf
import math

class Enviroment():
    def __init__(self):
        self.end_point = tf.constant([46, math.radians(28.6)])
        self.end_norm = tf.norm(self.end_point)

        self.acc = tf.constant(0.00351598)
        self.gain = tf.constant(0.0024262)

    def reset(self):
        radius = 1
        inclination = math.radians(51)
        azimuth = 0
        psi =  0
        beta = 0
        self.state = tf.Variable([azimuth, radius, inclination, psi, beta])
        self.Time = 0
        return self.state, False

   

    def step(self, action):

        done = False


        az, radius, inclination, psi, beta = tf.unstack(self.state)
        dpsi = beta
        dbeta = tf.cast(action - 1, float) * self.gain
        daz = (1 / (radius ** 1.5))
        dr = 2 * self.acc * (radius ** 1.5) * tf.cos(psi)
        di = self.acc * tf.sin(psi) * (radius ** 0.5) * tf.cos(az)
        dstate = tf.stack([daz, dr, di, dpsi, dbeta])



        err_1 = tf.norm(self.state[1:3] - self.end_point)

        self.state.assign_add(dstate*0.1)

        err_2 = tf.norm(self.state[1:3] - self.end_point)


        self.Time+=0.1

        reward = (err_1-err_2)*100

        if self.Time > 250:
            done = True

        if abs(self.state[3]) > 1.3:
            done = True

        if self.state[2] > math.radians(51):

            done = True

        return self.state, reward, done, self.Time
