import pygame
import sys
import time

class Grid_WorldSim:
    def __init__(self,height,width,start_loc,finish_loc,actions,reward=-1,shift=None):
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
        x_ = x
        x_ -= self.shift[y-1]
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
action = [[i,j] for i in range(-1,2) for j in range(-1,2) if not abs(i)==abs(j)]
grid = Grid_WorldSim(height=7,width=10,start_loc=[3,0],finish_loc=[3,7],shift = shift,actions = action)

TD = TDZero(grid,1000,alpha=0.5)
TD.epsilon=0
import cv2
TD.Learn()

TD.play()
    