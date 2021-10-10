import numpy as np
from scipy.stats import poisson
#lr1,lr2 = [int(x) for x in input().strip().split()]
#lrr1,lrr2 = [int(x) for x in input().strip().split()]
#reward = [10,-2]

gamma = 0.9
V = np.zeros([20+1,20+1])
pie = np.zeros([20+1,20+1])

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
        s = (1-s)/len(d)
        d = [(i,j+s) for i,j in d]
        return d


locA = samples(3,3)
locB = samples(4,2)

def max_cars():
	return 20


def state_action_value(state1,state2,action,V,gamma):
    new_state = [max(0,min(state1-action,max_cars())),max(0,min(state2+action,max_cars()))]

    val = -2*abs(action)
    for kA1,vA1 in locA.probs1:
        for kB1,vB1 in locB.probs1:
            for kA2,vA2 in locA.probs2:
                for kB2,vB2 in locB.probs2:
                    val_req = [min(new_state[0],kA1),min(new_state[1],kB1)]
                    probs = vA1*vB1*vA2*vB2
                    reward = sum(val_req)*10
                    new_s = [max(0,min(new_state[0]-val_req[0]+kA2,max_cars())),max(0,min(new_state[1]+val_req[1]+kB2,max_cars()))]#;print(new_state)
                    val+= probs*(reward+gamma*V[int(new_s[0]),int(new_s[1])])

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
        print("one",delta)
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
        if old!=pie[i,j] and abs(V[i,j]-val[-1][0])>0.1: stable = False
    return stable
import seaborn as sns


def policy_improvement(V,pie,theta):
    while True:
        policy_evaluation(V,pie,theta)
        print("ok")
        theta = theta/2
        stable = policy_iteration(V,pie)
        if stable: 
            break

		
policy_improvement(V,pie,50)








import cv2
import numpy as np
img =np.clip(np.load(r"C:\Users\Akshay\Desktop\img.npy"),0,1)
imgcorr =np.clip(np.load(r"C:\Users\Akshay\Desktop\imgcorr.npy"),0,1)
cv2.imshow("ff",2*img)
ax = sns.heatmap(imgcorr)
import matplotlib.pyplot as plt
plt.savefig("av.png")









track=np.zeros(shape=[1000,1000])
track[500:,500:] = 1
plt.imshow(track)
import cv2
class Track_simulator:
    
    def __init__(self,size,s_len,f_len,max_speed):
        self.size = size
        self.s_len = s_len
        self.f_len = f_len
        self.max_speed = max_speed
        self.real_track = self.make_track() 

    def make_track(self):
        track = np.zeros(self.size)
        #track[self.s_len:,self.f_len:] = 1
        real_track = np.ones([self.size[0]+4*self.max_speed,self.size[1]+4*self.max_speed])
        real_track[2*self.max_speed:-2*self.max_speed,2*self.max_speed:-2*self.max_speed] = track
        return real_track
    
    def reset_speed(self):
        self.Vx = 0
        self.Vy = 0
        
    def is_boundry(self,i,j):
        return self.real_track[i][j]==1
    
    def show_track(self):
        plt.imshow(self.real_track)
    
    def is_finish(self,i,j):
        f_i = self.size[0]+2*self.max_speed-1
        f_j_min = 2*self.max_speed
        f_j_max = f_j_min+self.f_len-1
        return i>=f_i and f_j_min<=j<=f_j_max
    
#    def rand_start(self):
#        start = np.random.randint(low=0,high=self.s_len)
#        start_index = start+2*self.max_speed
#        return start_index,self.size[0]+2*self.max_speed-2
    
    def is_start(self,i,j):
        s_j = self.size[1]+2*self.max_speed-1
        s_i_min = 2*self.max_speed
        s_i_max = s_i_min+self.s_len-1
        return j==s_j and s_i_min<=i<=s_i_max
    
    def take_action(self,action,i,j):
        vx = self.Vx+action[0]
        vy = self.Vy+action[1]
        if (vx<=0 or vy<=0) and not self.is_start(i,j):
            return self.Vx,self.Vy
        self.Vx = max(min(vx,self.max_speed-1),0)
        self.Vy = max(min(vy,self.max_speed-1),0)
        return self.Vx,self.Vy
    
    def s_point_loc(self,x):
        i = 2*self.max_speed+x
        j = self.size[1]+2*self.max_speed-1
        return i,j
    
    def rand_start(self):
        x = np.random.randint(low=0,high=self.s_len)
        return self.s_point_loc(x)

class MonteCarlo:
    def __init__(self,num_sim,gamma=1):
        self.track = Track_simulator([50]*2,10,20,5)
        self.policy = np.zeros([*self.track.real_track.shape],dtype=int)
        self.action_value = np.random.uniform(size=[*self.track.real_track.shape,9])
        self.index_to_action = [[i,j] for i in range(-1,2) for j in range(-1,2)]
        self.action_to_index = {str(self.index_to_action[i]):i for i in range(0,len(self.index_to_action))}
        self.policy = np.argmax(self.action_value,axis=2)
        self.num_sim = num_sim
        self.b = 1/9
        self.gamma = gamma
        
        
    def rand_action(self):
        x = np.random.choice([-1,0,1])
        y = np.random.choice([-1,0,1])
        return x,y
    
    def create_chain(self,start):
        self.track.reset_speed()
        chain = []
        mini = []
        i,j=start
        while True:
            x,y = self.rand_action()
            mini.append([i,j])
            mini.append([x,y])
            mini.append(-1)
            Vx,Vy = self.track.take_action([x,y],i,j)
            i+=Vx
            j-=Vy   
            chain.append(mini)
            mini = []
            if self.track.is_finish(i,j):
                break
            if self.track.is_boundry(i,j):
                self.track.reset_speed()
                i,j = self.track.rand_start()
        
        return chain
    def arg_max(self,arr,i,j):
        index = []
        val = np.max(arr[i,j])
        for k in range(len(arr[i,j])):
            if arr[i,j,k]==val:
                index.append(k)
        return index
    
    def Off_policy(self):
        # This off policy Monte Carlo sampling uses per decision importance sampling
        s = 0
        count = np.zeros_like(self.action_value)
        for episode in range(self.num_sim):
            G = 0
            s = s%self.track.s_len
            i,j = self.track.s_point_loc(s)
            chain = self.create_chain([i,j])
            
            for stateT,actionT,rewardT in chain[::-1]:
                G = self.gamma*G+rewardT
                actionT = self.action_to_index[str(actionT)]
                count[stateT[0],stateT[1],actionT]+=1
                t=count[stateT[0],stateT[1],actionT]
                self.action_value[stateT[0],stateT[1],actionT]+=(G-self.action_value[stateT[0],stateT[1],actionT])/t
                index = self.arg_max(self.action_value,stateT[0],stateT[1])
                max_action = actionT if actionT in index else index[0]
                self.policy[stateT[0],stateT[1]] = max_action
                W = 1/self.b if max_action==actionT else 0
                G = W*G
            s+=1
            print("Episode:",episode)
                
    def show(self,i,j):
        temp = self.track.real_track.copy()
        temp[i,j]  = 10
        cv2.imshow("playing with the policy",temp)
        return cv2.waitKey(10) & 0xFF == ord('q')
    
    def play(self,rand_start=True,pos=0):
        self.track.reset_speed()
        if rand_start:
            loc = self.track.rand_start()
        else:
            loc = self.track.s_point_loc(pos)
        while True:
            close = self.show(*loc)
            if close:
                break
            action = self.policy[loc[0],loc[1]]
            action = self.index_to_action[action]
            Vx,Vy = self.track.take_action(action,*loc)
            #print(action,Vx,Vy)
            loc = loc[0]+Vx,loc[1]-Vy
            if self.track.is_finish(*loc):
                print("FINISHED")
                break
            if self.track.is_boundry(*loc):
                self.track.reset_speed()
                loc = self.track.rand_start()
                #print("Boundry Hit. New location is",loc,self.track.real_track[loc[0],loc[1]])
#    def MCES_Every_Visit(self,):
#
#t = Track_simulator([10**3]*2,100,500,5)
#t.show_track()

d = MonteCarlo(100000)
ipolicy = d.policy.copy()
#f = d.create_chain(d.track.rand_start())
d.Off_policy()
d.play(False,pos=11)



#import gc
#gc.collect()

def pl(chain):
    for i in chain:
        temp = d.track.real_track.copy()
        temp[i[0],i[1]]=10
        
        cv2.imshow("track",temp)
        cv2.waitKey(0)
        yield 1
k = pl(f)
while True:
    next(k)
    
# =============================================================================
# 
# =============================================================================
import pygame
import sys
import time

class Grid_WorldSim:
    def __init__(self,height,width,start_loc,finish_loc,actions,reward=-1,shift=None,kings_move=False):
        self.shift = [0]*width if shift==None else shift
        self.height = height
        self.width = width
        self.start_loc = start_loc
        self.finish_loc = finish_loc
        self.grid = self.make_grid()
        self.r = reward
        self.actions = actions
        self.num_actions = len(self.actions)
        self.reset_loc()
        self.kings_move=kings_move
        
    def reset_loc(self):
        self.x_loc,self.y_loc = self.start_loc[0]+1,self.start_loc[1]+1
    
    def ActionVal_init(self):
        action_val = 0*np.random.uniform(low = 0,high = 1,size = [self.height+2,self.width+2,self.num_actions])
        action_val[self.finish_loc[0]+1,self.finish_loc[1]+1] = 0
        return action_val
    
    def make_grid(self):
        grid = np.zeros([self.height,self.width])
        grid[self.finish_loc[0],self.finish_loc[1]]=-1
        sudo_grid = np.ones([self.height+2,self.width+2])
        sudo_grid[1:self.height+1,1:self.width+1] = grid
        return sudo_grid
    
    def is_finished(self,i,j):
        return self.grid[i,j]==-1
    
    def is_boundry(self,i,j):
        return self.grid[i,j]==1
    
    def apply_wind(self,x,y):
        stoc_move = 0
        if self.kings_move:
            stoc_move = np.random.choice([-1,0,1])
        x_ = x
        x_ -= self.shift[y-1]+stoc_move
        if 0<x_<=self.height and 0<y<self.width:
            x = x_
        return x,y
    
    def starting_state(self):
        return self.start_loc[0]+1,self.start_loc[1]+1
    
    def simulate(self,action):
        action = self.actions[action]
        x_temp,y_temp = self.apply_wind(self.x_loc,self.y_loc)
        if not self.is_boundry(x_temp,y_temp):
            self.x_loc,self.y_loc = x_temp,y_temp
        x_temp,y_temp=self.x_loc+action[0],self.y_loc+action[1]
        if not self.is_boundry(x_temp,y_temp):
            self.x_loc,self.y_loc = x_temp,y_temp
        return self.r,[self.x_loc,self.y_loc]

class TDZero:
    def __init__(self,simulation,num_episodes,epsilon=0.1,alpha=0.5,gamma=1):
        self.simulation = simulation
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.action_val = self.simulation.ActionVal_init()
        self.policy = np.argmax(self.action_val,axis=2)
        self.num_action = self.simulation.num_actions
    
    def action(self,state):
        
        if self.epsilon>0:
            probs = self.epsilon/self.num_action
            rand = np.random.uniform()
            if rand<=self.epsilon:
                action = np.random.choice(range(self.num_action))
            else:
                action = self.policy[state[0],state[1]]
            if action==self.policy[state[0],state[1]]:
                return action,1-self.epsilon+probs
            else:
                return action,probs
        else:
            return self.policy[state[0],state[1]],1
        
    def Learn(self):
        t = 0
        for episode in range(self.num_episodes):
            self.simulation.reset_loc()
            state = self.simulation.starting_state()
            action = self.action(state)[0]
            
            while True:
                r,new_state = self.simulation.simulate(action)
                new_action = self.action(new_state)[0]
                Q = self.action_val[state[0],state[1],action]
                Q_next = self.action_val[new_state[0],new_state[1],new_action]
                self.action_val[state[0],state[1],action]+=self.alpha*(r+self.gamma*Q_next-Q)
                self.policy[state[0],state[1]] = np.argmax(self.action_val[state[0],state[1]])
                state = new_state
                action = new_action
                t+=1
                if self.simulation.is_finished(*state):
                    break
            print("Episode:",episode,"Time Steps Taken",t)
                
    
    def play(self,rand_start=True,pos=0):
        global SCREEN, CLOCK, GRID, HEIGHT, WIDTH, blockSize, BLACK, WHITE, GREEN, RED
        BLACK = (0, 0, 0)
        WHITE = (200, 200, 200)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        pygame.init()
        GRID = self.simulation.grid.copy()
        blockSize = 20
        WINDOW_HEIGHT, WINDOW_WIDTH = GRID.shape[0]*blockSize, GRID.shape[1]*blockSize
        SCREEN = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        CLOCK = pygame.time.Clock()
        SCREEN.fill(BLACK)
        HEIGHT,WIDTH = GRID.shape[0], GRID.shape[1]
        
        self.simulation.reset_loc()
        state = self.simulation.starting_state()
        count=0
        while True:
            GRID = self.simulation.grid.copy()
            GRID[state[0],state[1]]  = 10
            self.main()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            pygame.display.update()
            SCREEN.fill(BLACK)
            action = self.action(state)[0]
            print(state,action)
            _,state = self.simulation.simulate(action)
            count+=1
            if self.simulation.is_finished(*state):
                print("FINISHED")
                print("Steps Taken",count)
                pygame.quit()
                sys.exit()
                break
    
    def main(self):
        time.sleep(.5)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                color=WHITE
    
                rect = pygame.Rect(x*(blockSize), y*(blockSize),
                                   blockSize, blockSize)
                if GRID[y][x]==1:
                    color=GREEN
                    SCREEN.fill(color,rect)
                if GRID[y][x]==10:
                    color=RED
                    SCREEN.fill(color,rect)
                if GRID[y][x]==-1:
                    color = WHITE
                    SCREEN.fill(color,rect)
                pygame.draw.rect(SCREEN, color, rect, 1)
        
        


        
s = "0 0 0 1 1 1 2 2 1 0"
shift = [int(x) for x in s.strip().split()]
action = [[i,j] for i in range(-1,2) for j in range(-1,2) if not abs(i)==abs(j)==0]
grid = Grid_WorldSim(height=7,width=10,start_loc=[3,0],finish_loc=[3,7],shift = shift,actions = action,kings_move=True)

TD = TDZero(grid,1000,alpha=0.5)
TD.epsilon=0
import cv2
TD.Learn()

TD.play()
    















# =============================================================================
# EXPERIMENTS
# =============================================================================

class ChainSim:
    def __init__(self,chain_len=5,reward=1):
        self.chain_len = chain_len
        self.pos = 1
        self.reward = reward
        
    def actual_val(self):
        trans_mat = np.zeros(shape=(self.chain_len+2,self.chain_len+2)) 
        for i in range(1,trans_mat.shape[0]-1):
            for j in range(1,trans_mat.shape[1]-1):
                if i==j:
                    trans_mat[i][j-1]=0.5
                    trans_mat[i][j+1]=0.5
        ones = self.reward*np.ones(self.chain_len)
        const= np.zeros(self.chain_len+2)
        const[1:-1]=ones
        val = np.linalg.solve(np.identity(self.chain_len+2)-trans_mat,const)
        return val[1:-1]
    
    def is_finished(self,state):
        return state<=0 or state>=self.chain_len+1
    
    def take_step(self,e):
        return 1 if np.random.uniform()>e else -1
    
    def set_pos(self,pos):
        self.pos = pos
        
    def StateVal_init(self):
        return np.zeros(self.chain_len+2)
    
    def num_state(self):
        return self.chain_len
    
    def simulate(self,e):
        a = self.take_step(e)
        if self.is_finished(self.pos):
            return 0,self.pos,a
        
        self.pos += a
        return self.reward,self.pos,a
    
        
        
    


class TDnStep:
    def __init__(self,sim,num_episodes,average_over,alpha,gamma,n_steps):
        self.sim = sim
        self.num_states = self.sim.num_state()
        self.num_episodes = num_episodes
        self.average_over = average_over
        self.alpha = alpha
        self.n_steps = n_steps
        self.state_val = self.sim.StateVal_init()
        self.gamma = gamma
        self.actual_val = self.sim.actual_val()
        
    def Learn(self):
        c = 0
        store=[]
        b = 0.5
        for episode in range(self.num_episodes):
            state = c%self.num_states +1
            self.sim.set_pos(state)
            t = 0
            chain=[]
            #print('New_episode')
            while True:
                #print(state)
                if not self.sim.is_finished(state):
                    r,s,a = self.sim.simulate(b)
                    chain.append([r,s,a])
                else:
                    break
                #print(chain)
                tou = t-self.n_steps+1
                if tou>=0:
                    G = 0
                    for r,s,a in chain[::-1]:
                        G=self.gamma*G+r
                    G+=(self.gamma**(self.n_steps))*self.state_val[chain[-1][-2]]
                    self.state_val[state] += self.alpha*(G-self.state_val[state])
                    state = chain.pop(0)[-2]
                t+=1
            c+=1
            store.append(self.rms(self.state_val[1:-1],self.actual_val))
        return sum(store)/len(store)
            
    def rms(self,V,v):
        rms = V-v
        rms = rms**2
        rms = np.sum(rms)
        rms = rms**0.5
        return rms
    
    def TDerrorLearn(self):
        c = 0
        store = []
        b=0.5
        for episode in range(self.num_episodes):
            state = c%self.num_states +1
            self.sim.set_pos(state)
            t = 0
            chain=[]
            #print('New_episode')
            while True:
                #print(state)
                if not self.sim.is_finished(state):
                    r,s,a = self.sim.simulate(b)
                    chain.append([r,s,a])
                else:
                    break
                #print(chain)
                tou = t-self.n_steps+1
                if tou>=0:
                    G = 0
                    for i in range(0,self.n_steps-1):
                        _,s_,_ = chain[-2-i]
                        r,s,_ = chain[-1-i]
                        TDerror= r+self.gamma*self.state_val[s]-self.state_val[s_]
                        G=self.gamma*G+TDerror
                    r,s,a = chain[0]
                    TDerror = r+self.gamma*self.state_val[s]-self.state_val[state]
                    G = self.state_val[state]+self.gamma*G+TDerror
                    #G+=(self.gamma**(self.n_steps))*self.state_val[chain[-1][-1]]
                    self.state_val[state] += self.alpha*(G-self.state_val[state])
                    state = chain.pop(0)[-2]
                t+=1
            c+=1
            store.append(self.rms(self.state_val[1:-1],self.actual_val))
        return sum(store)/len(store)
    
    def off_policyLearn(self):
        c = 0
        store=[]
        b=0.4
        for episode in range(self.num_episodes):
            state = c%self.num_states +1
            self.sim.set_pos(state)
            t = 0
            chain=[]
            sigma = 1
            #print('New_episode')
            while True:
                #print(state)
                if not self.sim.is_finished(state):
                    r,s,a = self.sim.simulate(b)
                    chain.append([r,s,a])
                else:
                    break
                #print(chain)
                tou = t-self.n_steps+1
                if tou>=0:
                    for i in len(self.no_chain):
                        if a==1:
                            sigma = sigma*(0.5/(1-b))
                        else:
                            sigma = sigma*(0.5/b)
                    G = 0
                    for r,s in chain[::-1]:
                        G=self.gamma*G+r
                    G+=(self.gamma**(self.n_steps))*self.state_val[chain[-1][-1]]
                    self.state_val[state] += self.alpha*sigma*(G-self.state_val[state])
                    state = chain.pop(0)[-2]
                t+=1
            c+=1
            store.append(self.rms(self.state_val[1:-1],self.actual_val))
        
    def off_policyLearn_contVariate(self):
        c = 0
        store=[]
        b=0.4
        for episode in range(self.num_episodes):
            state = c%self.num_states +1
            self.sim.set_pos(state)
            t = 0
            chain=[]
            sigma = 1
            #print('New_episode')
            while True:
                #print(state)
                if not self.sim.is_finished(state):
                    r,s,a = self.sim.simulate(b)
                    chain.append([r,s,a])
                else:
                    break
                #print(chain)
                tou = t-self.n_steps+1
                if tou>=0:
                    for i in len(self.no_chain):
                        if a==1:
                            sigma = sigma*(0.5/(1-b))
                        else:
                            sigma = sigma*(0.5/b)
                    G = self.state_val[chain[-1][-2]]
                    for i in range(0,self.no_chain-1):
                        r,s,a = chain[-i-1]
                        _,s_,_ = chain[-i-2]
                        if a==1:
                            sigma = (0.5/(1-b))
                        else:
                            sigma = (0.5/b)
                        G=sigma*(self.gamma*G+r) + (1-sigma)*self.state_val[s_]
                    r,s,a = chain[0]
                    if a==1:
                        sigma = (0.5/(1-b))
                    else:
                        sigma = (0.5/b)
                    G+=sigma*(r+self.gamma*G)+(1-sigma)*self.state_val[state]
                    self.state_val[state] += self.alpha*(G-self.state_val[state])
                    state = chain.pop(0)[-2]
                t+=1
            c+=1
            store.append(self.rms(self.state_val[1:-1],self.actual_val))
    
        



import pickle
d = {}
alpha=0
inc = 0.05

while alpha<1:
    alpha+=inc
    n = 1
    while n<256:
        n = n*2
        d[alpha,n] = []
        store1=store2=0
        for j in range(0,100):
            sim = ChainSim(chain_len=10,reward=0.01)
            td = TDnStep(sim,num_episodes=100,average_over=1,alpha=alpha,n_steps=n,gamma=1)
            store1 += td.Learn()

            td = TDnStep(sim,num_episodes=100,average_over=1,alpha=alpha,n_steps=n,gamma=1)
            store2 += td.TDerrorLearn()
            
        d[alpha,n].append(store1/100)
        d[alpha,n].append(store2/100)


with open("data.plk",'wb') as a:
    pickle.dump(d,a)
import seaborn as sns
from  collections import Counter

alpha = [x[0] for x in d]
alpha = [x for x in Counter(sorted(alpha))]
n=1
ratings = np.unique(list(range(20)))
palette = iter(sns.husl_palette(len(ratings)))
while n<256:
    n*=2
    rms = []
    for i in range(0,len(alpha)):
        rms.append(d[alpha[i],n][0])
    sns.pointplot(alpha,rms,color=next(palette))
    rms = []
    for i in range(0,len(alpha)):
        rms.append(d[alpha[i],n][1])
    sns.pointplot(alpha,rms)

    



class Grid_WorldSim2:
    def __init__(self,height,width,start_loc,finish_loc,actions,phase_change=3000,reward=1):
        self.height = height
        self.width = width
        self.start_loc = start_loc
        self.finish_loc = finish_loc
        self.grid = self.make_grid()
        self.r = reward
        self.actions = actions
        self.num_actions = len(self.actions)
        self.reset_loc()
        self.phase_change=phase_change
        self.t = 0
        
    def reset_loc(self):
        self.x_loc,self.y_loc = self.start_loc[0]+1,self.start_loc[1]+1
    
    def ActionVal_init(self):
        action_val = 0*np.random.uniform(low = 0,high = 1,size = [self.height+2,self.width+2,self.num_actions])
        action_val[self.finish_loc[0]+1,self.finish_loc[1]+1] = 0
        return action_val
    
    def make_grid(self):
        grid = np.zeros([self.height,self.width])
        grid[self.finish_loc[0],self.finish_loc[1]]=-1
        grid[-2,0:-1]=1
        sudo_grid1 = np.ones([self.height+2,self.width+2])
        sudo_grid1[1:self.height+1,1:self.width+1] = grid
        
        grid = np.zeros([self.height,self.width])
        grid[self.finish_loc[0],self.finish_loc[1]]=-1
        grid[-2,1:-1]=1
        sudo_grid2 = np.ones([self.height+2,self.width+2])
        sudo_grid2[1:self.height+1,1:self.width+1] = grid
        return sudo_grid1,sudo_grid2
    
    def is_finished(self,i,j):
        phase = 0 if self.t<self.phase_change else 1
        return self.grid[phase][i,j]==-1
    
    def is_boundry(self,i,j):
        phase = 0 if self.t<self.phase_change else 1
        return self.grid[phase][i,j]==1
    
    def starting_state(self):
        return self.start_loc[0]+1,self.start_loc[1]+1
    
    def simulate(self,action):
        self.t +=1
        action = self.actions[action]
        x_temp,y_temp=self.x_loc+action[0],self.y_loc+action[1]
        if not self.is_boundry(x_temp,y_temp):
            self.x_loc,self.y_loc = x_temp,y_temp
        return 0 if not self.is_finished(self.x_loc,self.y_loc) else self.r,[self.x_loc,self.y_loc]


class DyanQ:
    def __init__(self,sim,alpha,num_loops,epsilon,model_learn_steps,gamma=1,k=0):
        self.sim = sim
        self.alpha =alpha 
        self.num_loops = num_loops
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.action_val = self.sim.ActionVal_init()
        self.policy = np.argmax(self.action_val,axis=2)
        self.num_action = self.sim.num_actions
        self.model = {}   
        self.model_learn_steps = model_learn_steps
        self.T = {}
        self.k = k
    def action(self,state):
        
        if self.epsilon>0:
            probs = self.epsilon/self.num_action
            rand = np.random.uniform()
            if rand<=self.epsilon:
                action = np.random.choice(range(self.num_action))
            else:
                action = self.policy[state[0],state[1]]
            if action==self.policy[state[0],state[1]]:
                return action,1-self.epsilon+probs
            else:
                return action,probs
        else:
            return self.policy[state[0],state[1]],1
        
    def Learn(self):
        t = 0
        self.sim.reset_loc()
        state = self.sim.starting_state()
        cumm_r = [0]
        for episode in range(self.num_loops):
            t+=1
            action = self.action(state)[0]
            self.set_time(state,action,t)
            r,new_state = self.sim.simulate(action)
            Q = self.action_val[state[0],state[1],action]
            Q_next = np.max(self.action_val[new_state[0],new_state[1]])
            self.action_val[state[0],state[1],action]+=self.alpha*(r+self.gamma*Q_next-Q)
            self.model_input(state,action,r,new_state)
            self.Model_Learn(t)
            self.policy[state[0],state[1]] = np.argmax(self.action_val[state[0],state[1]])
            state = new_state

            if self.sim.is_finished(*state):
                self.sim.reset_loc()
            cumm_r.append(cumm_r[-1]+r)
#            print("Episode:",episode,"Time Steps Taken",t)
        return cumm_r
    
    def set_time(self,state,action,t):
        state = tuple(state)
        key = (state,action)
        self.T[key]=t
        
    def get_time(self,state,action):
        state = tuple(state)
        key = (state,action)
        return 0 if key not in self.T else self.T[key] 
    
    def model_input(self,state,action,reward,state_):
        state = tuple(state)
        key = (state,action)
        self.model[key]=[reward,state_]
        
    def model_output(self,state,action):
        state = tuple(state)
        key = (state,action)
        return self.model[key]
    
    def sample_state_action(self):
        state_actions = self.model.keys()
        if len(state_actions)==1:
            return list(state_actions)[0]
        state,action = list(state_actions)[np.random.choice(range(len(list(state_actions))))]
        return state,action
    
    def Model_Learn(self,t):
        
        for i in range(self.model_learn_steps):
            state,action = self.sample_state_action()
            r,new_state = self.model_output(state,action)
            Q = self.action_val[state[0],state[1],action]
            Q_next = np.max(self.action_val[new_state[0],new_state[1]])
            T = t-self.get_time(state,action)
            extra_r = self.k*(T**0.5)
            self.action_val[state[0],state[1],action]+=self.alpha*(r+extra_r+self.gamma*Q_next-Q)
            
    def play(self,rand_start=True,pos=0):
        global SCREEN, CLOCK, GRID, HEIGHT, WIDTH, blockSize, BLACK, WHITE, GREEN, RED
        BLACK = (0, 0, 0)
        WHITE = (200, 200, 200)
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        pygame.init()
        GRID = self.sim.grid[1].copy()
        blockSize = 20
        WINDOW_HEIGHT, WINDOW_WIDTH = GRID.shape[0]*blockSize, GRID.shape[1]*blockSize
        SCREEN = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
        CLOCK = pygame.time.Clock()
        SCREEN.fill(BLACK)
        HEIGHT,WIDTH = GRID.shape[0], GRID.shape[1]
        
        self.sim.reset_loc()
        state = self.sim.starting_state()
        count=0
        while True:
            GRID = self.sim.grid.copy()
            GRID[state[0],state[1]]  = 10
            self.main()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
            pygame.display.update()
            SCREEN.fill(BLACK)
            action = self.action(state)[0]
            print(state,action)
            _,state = self.sim.sim(action)
            count+=1
            if self.sim.is_finished(*state):
                print("FINISHED")
                print("Steps Taken",count)
                pygame.quit()
                sys.exit()
                break
    
    def main(self):
        time.sleep(.5)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                color=WHITE
    
                rect = pygame.Rect(x*(blockSize), y*(blockSize),
                                   blockSize, blockSize)
                if GRID[y][x]==1:
                    color=GREEN
                    SCREEN.fill(color,rect)
                if GRID[y][x]==10:
                    color=RED
                    SCREEN.fill(color,rect)
                if GRID[y][x]==-1:
                    color = WHITE
                    SCREEN.fill(color,rect)
                pygame.draw.rect(SCREEN, color, rect, 1)
        
        


        

action = [[i,j] for i in range(-1,2) for j in range(-1,2) if not abs(i)==abs(j)]
grid = Grid_WorldSim2(height=6,width=9,start_loc=[-1,3],finish_loc=[0,-1],actions = action)

TD = DyanQ(grid,0.1,num_loops=10000,model_learn_steps=100,epsilon=0.1,k=0.2)
#TD.epsilon=0
import cv2
cumm = TD.Learn()

#TD.play()
            
import seaborn as sns
sns.pointplot(x=list(range(len(cumm))),y=cumm)


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
    
    def __init__(self,model,num_updates,epsilon=0.1):
        self.model = model
        self.num_updates = num_updates
        self.epsilon = epsilon
        
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
            exp+=(1-self.model.terminal_prob)*(r+np.max(self.q[s]))
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
df.to_csv("1000states:ex8.8.csv")

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
df.to_csv("1000states:ex8.8.csv")
    

# =============================================================================
# Exercise 11.3
# =============================================================================

class Biard:
    def __init__(self,num_states):
        self.num_states = num_states
        self.action = {'solid':0,"dashed":1}
        
    def starting_state(self):
        self.state = np.random.randint(low=0,high=self.num_states)
        return self.state
    
    def simulate(self,action):
        action = self.action[action]
        if action==0:
            self.state = self.num_states-1
        else:
            self.state = np.random.randint(low=0,high=self.num_states-1)
        return 0,self.state
    
        
class SemiGrad_QLearning:
    def __init__(self,sim,num_steps,alpha,gamma):
        self.sim = sim
        self.num_steps = num_steps
        self.alpha = alpha
        self.gamma = gamma 
        self.features = self.create_features()
        self.params = np.ones(self.sim.num_states+1+1)
        #self.params[6]=10
        #probs for solid action
        self.b = 1/self.sim.num_states
        self.pie = 1
        
    def create_features(self):
        solid = [1]
        dashed = [0]
        from collections import defaultdict
        features_dict = defaultdict(dict)
        for i in range(self.sim.num_states):
            zero_model = [0]*(self.sim.num_states+1)
            if i ==self.sim.num_states-1:
                zero_model[i] = 1
                zero_model[self.sim.num_states] = 2
            else:
                zero_model[i] = 2
                zero_model[self.sim.num_states] = 1
            features_dict[i]['solid'] = np.array(zero_model+solid)
            features_dict[i]['dashed'] =np.array(zero_model+dashed)
        return features_dict
    
    def action(self):
        rand = np.random.uniform()
        if rand<=self.b:
            return "solid",self.b,self.pie
        return "dashed",1-self.b,1-self.pie
    
    def action_val(self,state,action):
        params = self.params
        xs = self.features[state][action]
        return np.sum(params*xs)
    
    def max_action_val(self,state):
        params = self.params
        xs1,xs2 = self.features[state].values()
        return max(np.sum(params*xs1),np.sum(params*xs2))
        
    def updates(self):
        state = self.sim.starting_state()
        p= [self.params]
        for iteration in range(self.num_steps):
            action,b_probs,pie_probs = self.action()
            r,next_state = self.sim.simulate(action)
            self.params = self.params+self.alpha*(r+self.gamma*self.max_action_val(next_state)-self.action_val(state,action))*self.features[state][action]
            state=next_state
            p.append(self.params)
        return p
k=5
sim = Biard(k)
agent = SemiGrad_QLearning(sim,10,0.01,0.99)
p = []  
for i in range(1000):
    p += agent.updates()
import pandas
df = pandas.DataFrame()
for i in range(1,k+3):
    k = []
    for j in p:
        k.append(j[i-1])
    df["W"+str(i)] = k
df.plot()



l = 2**0.5-1
val = -2*(1+l)/(l*(1-l))
print(val)