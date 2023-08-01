from Enviroment import Enviroment
from  Agent import Agent
env = Enviroment()
agent = Agent()


state, done = env.reset()



while not done:
    action = agent.select_action(state)
    state_, reward, done, time = env.step(action)
    agent.remember(state,action,state_,reward,done)
    state = state_
    print(done)

