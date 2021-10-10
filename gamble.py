import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ph = .4
V = np.zeros(100+1)
V[100]=1
pi = np.zeros(100+1)

def state_action_value(state,action,V,gamma):
      val = 0
      for i in range(2):
        if i==1:
          new_state = state+action
          val+=ph*(V[new_state])
        else:
          new_state = state-action
          val+=(1-ph)*(gamma*V[new_state])
      return val

def plot_save(x,name,count):
  plt.figure()
  sns.set_style("darkgrid")
  plt.plot(x)
  plt.savefig("{0}{1}.png".format(name,count))
  plt.close()
def bar_save(x,name):
  plt.figure()
  sns.set_style("darkgrid")
  plt.bar(np.arange(0,x.shape[0]),x)
  plt.savefig("{0}.png".format(name))
  plt.close()

def value_iteration(V,pi,gamma=1,theta=0.01):
    count=0
    while True:
      delta=0
      for i in range(1,100):
        v = V[i]
        max_action_val=0
        c = 0
        for j in range(0,min(i,100-i)+1):
          c = state_action_value(i,j,V,gamma)
          if c<max_action_val:
            c = max_action_val
        V[i] = c
        delta = max(delta,abs(v-V[i]))
      count+=1
      plot_save(V,"value_gamble"+str(ph),count)
      print(delta)
      if delta<=theta:
        break
	# policy
    for i in range(1,100):
      v = V[i]
      action_val = np.zeros(min(i,100-i)+1)
      for j in range(0,min(i,100-i)+1):
        action_val[j] = state_action_value(i,j,V,gamma)
      print(action_val,i)
      pi[i] = np.argmax(action_val)
    bar_save(pi,"policy_gamble"+str(ph))
value_iteration(V,pi,theta=0)