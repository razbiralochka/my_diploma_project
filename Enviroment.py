import numpy as np
import math

class Enviroment():
    def __init__(self):
        self.end_point =np.array([46, math.radians(28.6)])
        self.acc = 0.0001
        self.gain = 0.1

    def reset(self):
        radius = 19
        inclination =0.685008
        azimuth = np.pi
        psi =  0.521268
        beta = 0
        self.sum_rewerd = 0
        self.state = np.array([azimuth, radius, inclination, psi, beta])
        self.Time = 0
        return self.state, False

    def reward(self, r,i):
       a = -0.102334
       b = 0.8918
       ic = a*np.log(r)+b
       res = 1/abs(ic-i)
       return res

    def step(self, action):

        done = False
        h = 0.1

        az = self.state[0]
        radius = self.state[1]
        psi = self.state[3]
        beta = self.state[4]


        dpsi = beta
        dbeta = float(action-1) * self.gain

        daz = (1 / (radius ** 1.5))
        dr = 2 * self.acc * (radius ** 1.5) * np.cos(psi)
        di = self.acc * np.sin(psi) * (radius ** 0.5) * np.cos(az)
        dstate = np.array([daz, dr, di, dpsi, dbeta])


        err1 = np.linalg.norm(self.state[1:3] - self.end_point)

        self.state += dstate*h

        err2 = np.linalg.norm(self.state[1:3] - self.end_point)


        self.Time+=h

        reward = 10*(err1-err2)/err2

        if self.state[1] > 46:
            done = True
            reward += 5000/abs(self.state[2]-0.5)

        if abs(self.state[3]) > 1.2:
            done = True
            reward = -100


        return self.state, reward, done, self.Time
