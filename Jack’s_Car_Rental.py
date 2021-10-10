import numpy as np
# lr1,lr2 = [int(x) for x in input().strip().split()]
# lrr1,lrr2 = [int(x) for x in input().strip().split()]
#this is modified versionof the jack's Car rental from the book reinforcement learning by sutton and barto
from scipy.stats import poisson
gamma = 0.9
V = np.zeros([20+1,20+1])
pie = np.zeros([20+1,20+1],dtype=int)

class samples:
    def __init__(self,l1,l2,ep = 0.01):
        self.ep = ep
        self.probs1 = self.sample(l1)
        self.probs2 = self.sample(l2)
        
    def sample(self,lam):
        d = []
        i = 0
        flag=False
        s = 0
        while True:
            prob =  poisson.pmf(i,lam)
            if prob>=self.ep:
                flag=True
                s+=prob
                d.append([i,prob])
            i+=1
            if prob<self.ep and flag:
                break
        
        d = [(i,j/s) for i,j in d]
        return d


locA = samples(3,3)
locB = samples(4,2)

def max_cars():
	return 20


def state_action_value(state1,state2,action,V,gamma):
    new_state = [max(0,min(state1-action,max_cars())),max(0,min(state2+action,max_cars()))]
    s_park = sum([int(x>10) for x in new_state ])
    if action>0:
       val = -2*abs(action-1) -4*s_park
    else:
      val = -2*abs(action) -4*s_park

    count=0
    for kA1,vA1 in locA.probs1:
        for kB1,vB1 in locB.probs1:
            for kA2,vA2 in locA.probs2:
                for kB2,vB2 in locB.probs2:
                    count+=1
                    val_req = [min(new_state[0],kA1),min(new_state[1],kB1)]
                    probs = vA1*vB1*vA2*vB2
                    reward = sum(val_req)*10
                    new_s = [max(0,min(new_state[0]-val_req[0]+kA2,max_cars())),max(0,min(new_state[1]-val_req[1]+kB2,max_cars()))]#;print(new_state)
                    val+= probs*(reward+gamma*V[new_s[0],new_s[1]])

    return val

# def transition(state1,state2,action,r):
# 	s_ = state1-action
# 	s2_= state2+action
# 	s_,s2_ = state(s_,s2_)
# 	r += -2*abs(action)
# 	return s_,r,s2_
def policy_evaluation(V,pie,theta):
    
    while True:
        delta=0
        for s1 in range(0,21):
            for s2 in range(0,21):
                    v = V[s1,s2]
                    action = pie[s1,s2]
                    V[s1,s2] = state_action_value(s1,s2,action,V,0.9)
                    delta = max(delta,np.abs(v-V[s1,s2]))
        print(delta)
        if delta<theta:
            break

def policy_iteration(V,pie):
    stable = True
    for i in range(21):
      for j in range(21):
        old = pie[i,j]
        val = []
        t1 = min(i,5)
        t2 = -min(j,5)
        for action in range(t2,t1+1):
          v = state_action_value(i,j,action,V,0.9)
          val.append([v,action])
        action = sorted(val,key=lambda x: x[0])[-1][1]
        pie[i,j] = action
        if old!=pie[i,j] and V[i,j]!=val[-1][0]: stable = False
    return stable
import seaborn as sns


def policy_improvement(V,pie,theta):
    i = 0
    while True:
        policy_evaluation(V,pie,theta)
        theta = theta/10
        print("ok")
        stable = policy_iteration(V,pie)
        plt.figure()
        ax = sns.heatmap(pie)
        ax.invert_yaxis()
        plt.savefig("policy_var{}.png".format(i+1))
        plt.close()
        plt.figure()
        ax = sns.heatmap(V)
        ax.invert_yaxis()
        plt.savefig("value_var{}.png".format(i+1))
        plt.close()
        i+=1
        if stable: 
            break

		
policy_improvement(V,pie,50)


