import cv2
import matplotlin.pyplot as plt

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