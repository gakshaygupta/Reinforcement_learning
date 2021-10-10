import pandas
class Model:
    def __init__(self,num_states,num_actions,b,terminal_prob=0.1):
        self.num_states = num_states
        self.b = b
        self.start_state = 0   
        self.num_actions = num_actions
        self.rewards = self.set_reward()
        self.transitions = self.set_transitions()
        self.terminal_prob = terminal_prob
        self.terminal_r = np.random.normal(0,1)
        
    def ActionVal_init(self):
        return np.zeros([self.num_states,self.num_actions])
    
    def reset_state(self):
        self.state = self.start_state
    
    def stating_state(self):
        return self.start_state
    
    def set_transitions(self):
        T = np.random.randint(low=0,high=self.num_states,size=[self.num_states,self.num_actions,self.b])
        return T
        
    def set_reward(self):
        reward = np.random.normal(0,1,size=[self.num_states,self.num_actions,self.b])
        return reward
    
    def sample_index(self):
        probT = np.random.uniform()
        if probT<self.terminal_prob:
            return -1
        index = np.random.randint(low=0,high=self.b)
        return index
    
    def on_policy_sim(self,action):
        index = self.sample_index()
        if index>=0:
            s,r = self.transitions[self.state,action,index],self.rewards[self.state,action,index]
            self.state = s
            return r,s
        self.state = -1
        return self.terminal_r,-1
    
    def uniform_sim(self):
        state = np.random.randint(low=0,high=self.num_states)
        action = np.random.randint(low=0,high=self.num_actions)
        return state,action

class Agent:
    
    def __init__(self,model,num_updates,gamma=1,epsilon=0.1):
        self.model = model
        self.num_updates = num_updates
        self.epsilon = epsilon
        self.gamma = gamma
        
    def ValPolicy_init(self):
        self.q = self.model.ActionVal_init()
        self.policy = np.argmax(self.q,axis=-1)
        
    def action(self,state):
        
        if self.epsilon>0:
            probs = self.epsilon/self.model.num_actions
            rand = np.random.uniform()
            if rand<=self.epsilon:
                action = np.random.choice(range(self.model.num_actions))
            else:
                action = self.policy[state]
            if action==self.policy[state]:
                return action,1-self.epsilon+probs
            else:
                return action,probs
        else:
            return self.policy[state],1
        
    def calc_v_naught(self):
        max_action = self.policy[self.model.start_state]
        exp = (1-self.epsilon+self.epsilon/self.model.num_actions)*self.q[self.model.start_state,max_action]
        for i in range(self.model.num_actions):
            if i!= max_action:
                exp+=self.epsilon*self.q[self.model.start_state,i]/self.model.num_actions
        return exp
    
    def exp_update(self,state,action):
        exp = 0
        T = self.model.transitions
        for i in range(self.model.b):
            s = T[state,action,i]
            r = self.model.rewards[state,action,i]
            exp+=(1-self.model.terminal_prob)*(r+self.gamma*np.max(self.q[s]))
        exp /= self.model.b
        exp += self.model.terminal_prob*(self.model.terminal_r)
        return exp
            
    def on_policy_updates(self):
        self.ValPolicy_init()
        self.model.reset_state()
        v_naught =[]
        for i in range(self.num_updates):
            state = self.model.state
            action,_ =  self.action(state)
            r,s = self.model.on_policy_sim(action)
            self.q[state,action] = self.exp_update(state,action)
            if s==-1:
                self.model.reset_state()
            v_naught.append(self.calc_v_naught())
        return v_naught
            
    def uniform_updates(self):
        self.ValPolicy_init()
        self.model.reset_state()
        v_naught =[]
        for i in range(self.num_updates):
            state,action = self.model.uniform_sim()
            self.q[state,action] = self.exp_update(state,action)
            v_naught.append(self.calc_v_naught())
        return v_naught

df = pandas.DataFrame()  
l = []
lk = []  
for i in range(200):
    model = Model(num_states=1000,num_actions=2,b=1)
    agent = Agent(model,num_updates=20000,epsilon=0.1)
    l.append(agent.on_policy_updates())
    lk.append(agent.uniform_updates())
df["on_policy,b=1"]=np.sum(np.array(l),axis=0)
df["uniform,b=1"] =np.sum(np.array(lk),axis=0)

l = []
lk = []  
for i in range(200):
    model = Model(num_states=1000,num_actions=2,b=3)
    agent = Agent(model,num_updates=20000,epsilon=0.1)
    l.append(agent.on_policy_updates())
    lk.append(agent.uniform_updates())
df["on_policy,b=3"]=np.sum(np.array(l),axis=0)
df["uniform,b=3"] =np.sum(np.array(lk),axis=0)

l = []
lk = []  
for i in range(200):
    model = Model(num_states=1000,num_actions=2,b=10)
    agent = Agent(model,num_updates=20000,epsilon=0.1)
    l.append(agent.on_policy_updates())
    lk.append(agent.uniform_updates())
df["on_policy,b=10"]=np.sum(np.array(l),axis=0)
df["uniform,b=10"] =np.sum(np.array(lk),axis=0)

df.plot()

df = pandas.DataFrame()  
l = []
lk = []  
for i in range(200):
    model = Model(num_states=10000,num_actions=2,b=1)
    agent = Agent(model,num_updates=200000,epsilon=0.1)
    l.append(agent.on_policy_updates())
    lk.append(agent.uniform_updates())
df["on_policy,b=3"]=np.sum(np.array(l),axis=0)
df["uniform,b=3"] =np.sum(np.array(lk),axis=0)

l = []
lk = []  
for i in range(200):
    model = Model(num_states=10000,num_actions=2,b=3)
    agent = Agent(model,num_updates=200000,epsilon=0.1)
    l.append(agent.on_policy_updates())
    lk.append(agent.uniform_updates())
df["on_policy,b=10"]=np.sum(np.array(l),axis=0)
df["uniform,b=10"] =np.sum(np.array(lk),axis=0)

df.plot()