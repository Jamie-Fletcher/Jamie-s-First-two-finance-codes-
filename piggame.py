import numpy as np
import matplotlib.pyplot as plt

def gameturn(threshold):
    turn=0
    while threshold>turn:
        roll = np.random.randint(1,7)
        if roll == 1:
            return 0 
        else:
            turn +=roll 
    return turn

def Simulation(threshold,num_simulations):
    scores = []
    for i in range(num_simulations):
        score = gameturn(threshold)
        scores.append(score)
    return np.mean(scores)

strats = range(18,23)
num_sims = 1000000
expectedvalues=[]
for i in strats:
    ev = Simulation(i,num_sims)
    expectedvalues.append(ev)
    

best=max(expectedvalues)
bestindex= expectedvalues.index(best)
beststrat=strats[bestindex]
print(beststrat)
plt.figure(figsize=(10,6))
plt.plot(strats, expectedvalues, marker='o', color='b')
plt.axvline(beststrat, color='r', linestyle='--', label=f'Optimal ({beststrat})')
plt.title('Optimization of "Pig" Dice Game Strategy')
plt.xlabel('Hold Threshold')
plt.ylabel('Expected Value (Points per Turn)')
plt.legend()
plt.grid(True)
plt.show()