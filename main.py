import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 3)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x




model = DQN1()


class Calcs():
    def __init__(self):
        self.inc_list = list()
        self.r_list = list()
        self.l_list = list()
        self.p_list = list()
        self.time_list = list()
        self.inc0 = 51
        self.inc_k = 28.6
        self.r_k = 46

    def get_radius(self):
        return self.r_list

    def get_angles(self):
        return self.p_list
    def get_inc(self):
        return self.inc_list
    def get_control(self):
        return self.l_list
    def get_time(self):
        return self.time_list

    def equs(self):
        r = 1
        phi = 0
        beta = 0
        lam = 0
        u = 0
        v = 0
        omega = 0
        inc = math.radians(self.inc0)
        inc_k = math.radians(self.inc_k)
        r_k = self.r_k
        self.r_list.clear()
        self.inc_list.clear()
        self.p_list.clear()
        self.l_list.clear()
        self.time_list.clear()
        T=0
        acc = 0.00351598
        net = DQN()
        while T < 250:

            self.r_list.append(r)#*8371)
            self.time_list.append(v)#*6900.5)
            self.p_list.append(math.degrees(phi))
            self.l_list.append(math.degrees(u))
            self.inc_list.append(math.degrees(inc))

            dt = np.pi/1000
            inp = torch.tensor([r,inc,omega,lam,beta])
            u = torch.argmax(net(inp.float())).numpy()


            dlam = beta
            dbeta = u*0.001
            dp = (1 / (r ** 1.5))
            dr = 2*acc*(r**1.5)*math.cos(lam)
            di = acc*math.sin(lam)*(r**0.5)*np.cos(phi)
            dom = acc*math.sin(lam)*(r**0.5)*np.sin(phi)
            dV = acc

            lam += dlam*dt
            beta += dbeta*dt
            phi += dp*dt
            r += dr * dt
            inc += di * dt
            omega += dom * dt
            v += dV*dt
            T += dt
        err =np.sqrt((inc-inc_k)**2+(r-r_k)**2)
        print("err: ", err)
        print("omega: ", math.degrees(omega))
        print("Time: ",T)



calcs = Calcs()



calcs.equs()

angle = calcs.get_angles()
time = calcs.get_time()
radius = calcs.get_radius()
inclination = calcs.get_inc()

plt.plot(time, radius)
plt.xlabel("Характеристическая скорость")
plt.ylabel("Восота орбиты, км")
plt.show()

plt.plot(time, inclination)
plt.xlabel("Характеристическая скорость")
plt.ylabel("Наклонение орбиты, град")
plt.show()

plt.plot(time, calcs.get_control())
plt.xlabel("Характеристическая скорость")
plt.ylabel("Управляющий сигнал, град")
plt.show()


