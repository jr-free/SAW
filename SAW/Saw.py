import matplotlib.pyplot as plt
import numpy as np
from numpy import nan, mean
from numpy.random import randint
from numpy.random import choice
from collections import deque
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

class Saw():
    
    ###############################################
    # General methods for simulation and animation
    ###############################################
    def __sample(M, dim=(10,10), space=2, method = 1):
        '''
        Used for calculating number of SAWs in a specified grid.
        
            dim -- the size of grid for the walk. Can be 2 or 3D. Just input
                    tuple. i.e. (10,10) for 10x10 2D lattice, or
                    (10,10,10) for 10x10x10 lattice in 3D.
        '''
        SAWc = deque([])
        
        # Conditions to check for 2D
        if space == 2 and method == 1:
            for i in range(M):
                SAWc.append( Saw.Saw2D(dim).run1() )
            return SAWc
        if space == 2 and method == 2:
            for i in range(M):
                SAWc.append( Saw.Saw2D(dim).run2() )
            return SAWc
        if space == 2 and method == 3:
            for i in range(M):
                SAWc.append( Saw.Saw2D(dim).run3() )
            return SAWc
        
        # Conditions to check for 3D
        if space == 3 and len(dim)==3 and method == 1:
            for i in range(M):
                SAWc.append( Saw.Saw3D(dim).run1() )
            return SAWc       
        if space == 3 and len(dim)==3 and method == 2:
            for i in range(M):
                SAWc.append( Saw.Saw3D(dim).run2() )
            return SAWc        
        if space == 3 and len(dim)==3 and method == 3:
            for i in range(M):
                SAWc.append( Saw.Saw3D(dim).run3() )
            return SAWc

    def simulate2D(n, dim=(10,10), method = 1):
        '''
        
        Parameters
        ----------
        n : int
            Number of SAWs to generate.
        dim : 2-Tuple, optional
            The default is (10,10). This specifies a 10x10 lattice.
        method : int, optional
            The default is 1. Range is 1-3. Specifies the type of monte carlo
            involved in the simulation.

        Returns
        -------
        Large int
            Returns the estimated number of SAWs in 2d.

        '''
        return np.mean(Saw.__sample(n, dim, method=method))
    
    def animate2D(n, dim=(10,10), method=1):
        
        saws = deque([])
        for i in range(n):
            s = Saw.Saw2D(dim=dim)
            s.run2()
            saws.append(s)
            del s
            
        fig, ax = plt.subplots(111, projection='3d')
        saw_plots = [saw.draw() for saw in saws]
        my_anim = animation.ArtistAnimation(fig, 
                                            saw_plots, 
                                            interval = 500, 
                                            blit=False, 
                                            repeat_delay=2000)
        plt.show()
    
    def simulate3D(n, dim=(10,10,10), method = 1):
        '''
        
        Parameters
        ----------
        n : int
            Number of SAWs to generate.
        dim : 3-tuple, optional
            The default is (10,10,10). This specifies a 10x10x10 lattice.
        method : int, optional
            The default is 1. Range is 1-3. Specifies the type of monte carlo
            involved in the simulation.

        Returns
        -------
        Large int
            Returns the estimated number of SAWs in 3d.

        '''
        return np.mean(Saw.__sample(n, dim, space=3, method=method))
    
    def animate3D(n, dim=(10,10,10), method=1):
        saws = deque([])
        for i in range(n):
            s = Saw.Saw3D(dim=dim)
            s.run2()
            saws.append(s)
            del s
            
        fig, ax = plt.subplots()
        saw_plots = [saw.draw() for saw in saws]
        my_anim = animation.ArtistAnimation(fig, 
                                            saw_plots, 
                                            interval = 500, 
                                            blit=False, 
                                            repeat_delay=2000)
        plt.show()
    
    
    ############################################
    # Self-avoiding walk class in two dimensions
    ############################################
    class Saw2D():
        '''
        This class can be used to generate and count self-avoiding walks in two
        dimensions. 
        
        There are three different run options that correspond to three 
        separate monte carlo estimation methods:
            
            run1 -- this method runs until there are no more options.
            run2 -- this method randomly generates a maximum walk length and
                    runs until that length is achieved or the walk is trapped.
            run3 -- this method runs until there is a collision, or it will
                    terminate randomly with small probability.
        '''
        
        def __init__(self, dim = (10,10)):
            self.dim = dim
            
            # False = not occupied, True = occupied.
            self.grid = np.zeros((dim[0],dim[1]), dtype=bool)
            
            # Start at origin
            self.start = (0,0)
            
            # Set sentinel
            self.current = self.start
            
            # No established end point.
            self.end = np.nan
            
            # Number of choices at each step. Max of dimension-1 steps.
            self.num_choices = []
            
            # Length of the walk.
            self.length = 0
            
            # Nodes
            self.nodes = deque([self.start])
            
            # Total points on the lattice.
            self.total_points = dim[0]*dim[1]
    
        def __repr__(self):
            return "Self-avoiding walk of length {}".format(self.length)
        
        def __eq__(self, other):
            '''
            Walks are equal precisely if they pass through the same nodes.
            '''
            return self.nodes == other.nodes
        
        def __hash__(self):
            return hash(self.length)
        
        def __len__(self):
            return len(self.nodes)
        
        def check_collision(self, node):
            '''
            Determine whether a given node on the lattice is part of the
            SAW or boundary (if specified).
            '''
            
            # If the point of interest is outside of boundary, return false.
            if node[0]<0 or node[1]<0:
                return False
            if node[0] >= self.dim[0] or node[1] >= self.dim[1]:
                return False
            
            # Otherwise, find the value on the grid and determine if occupied.
            return not self.grid[node[0]][node[1]]
            
            # if self.restricted == False:
            #     return node not in self.nodes
            # else:
            #     return node not in self.nodes and node not in self.boundary
        
        def find_movement_options(self):
            '''
            Check for collisions around current node and determine 
            viable movement options.
            '''
            
            options = []
            if self.check_collision((self.current[0], self.current[1]+1)):
                options.append(0)
            if self.check_collision((self.current[0], self.current[1]-1)):
                options.append(1)
            if self.check_collision((self.current[0]-1, self.current[1])):
                options.append(2)
            if self.check_collision((self.current[0]+1, self.current[1])):
                options.append(3)
            
            if options == []:
                return -1
            else:
                return options
        
        
        def move(self, direction):
            '''
            Move in a given direction. 
            
            0 - Up
            1 - Down
            2 - Left
            3 - Right
            '''
    
            if direction == 0:
                self.current = (self.current[0], self.current[1]+1)
                self.grid[self.current[0]][self.current[1]] = True
                self.nodes.append(self.current)
            elif direction == 1:
                self.current = (self.current[0], self.current[1]-1)
                self.grid[self.current[0]][self.current[1]] = True
                self.nodes.append(self.current)
    
            elif direction == 2:
                self.current = (self.current[0]-1, self.current[1])
                self.grid[self.current[0]][self.current[1]] = True
                self.nodes.append(self.current)
    
            elif direction == 3:
                self.current = (self.current[0]+1, self.current[1])
                self.grid[self.current[0]][self.current[1]] = True
                self.nodes.append(self.current)
            else:
                return False
            
            self.length = self.length+1
            return True
    
        
    
        def run1(self, index = None):
            '''
            run1: SAW keeps walking until out of options
            '''
            
            while(True):
                
                options = self.find_movement_options()
                
                if options == -1:
                    break
                _LEN = len(options)
                self.num_choices.append(_LEN)
                probs = [1/_LEN]*_LEN
                direction = choice(options, size=1, p = probs)
                self.move(direction)
                
                            
            #self.nodes = tuple(self.nodes)
            
            result = 1
            for item in self.num_choices:
                result = result*item
            
            return result
            #return self
        
        
        def run2(self, index = None):
            '''
            run2: SAW starts by picking a maximum length, then walks until 
            either that length is achieved or it runs out of possible moves 
            (i.e. gets trapped).
            '''
            
            # Pick a length at random. Either move until you can't move anymore
            # or until you reach desired length.
            total_length = np.random.randint(1,self.total_points)
            while(self.length < total_length):
                
                options = self.find_movement_options()
                if options == -1:
                    break
                _LEN = len(options)
                self.num_choices.append(_LEN)
                probs = [1/_LEN]*_LEN
                direction = choice(options, size=1, p = probs)
                self.move(direction)
                
                            
            #self.nodes = tuple(self.nodes)
            result = 1
            for item in self.num_choices:
                result = result*item
            
            return result
            #return self
        
        def run3(self, stop_prob=0.1, index=None):
            '''
            run3: Run until collision or stop abruptly with small probability.
            '''
            
            while(True):
                
                U = np.random.uniform(0,1,1)
                options = self.find_movement_options()
                if options == -1 or U < stop_prob:
                    break
                
                self.num_choices.append(len(options))
                probs = [1/len(options)]*len(options)
                direction = choice(options, size=1, p = probs)
                self.move(direction)
                
            #self.nodes = tuple(self.nodes)
            #del self.boundary
            
            result = 1
            for item in self.num_choices:
                result = result*item
            
            return result
            #return self
            
            
        
        def draw(self, show=False):
            '''
            Draw the SAW.
            '''
            
            x = [node[0] for node in self.nodes]
            y = [node[1] for node in self.nodes]
            
            p1, = plt.plot(x,y, '-o')
            p2 = plt.scatter(x,y,s=50)
            
            # Set integer ticks and window size.
            plt.xlim((0, self.dim[0]))
            plt.xticks(range(0,self.dim[0]))
            plt.ylim((0,self.dim[1]))
            plt.yticks(range(0,self.dim[1]))            
            
            if show == True:
                plt.show()
            
            return [p1, p2]
    
    ##############################################
    # Self-avoiding walk class in three dimensions
    ##############################################
    class Saw3D():
        '''
        This class can be used to generate and count self-avoiding walks in
        three dimensions. Note: this is all just modified boilerplate code
        from Saw2D -- the machinary is all the same. Not the most elegant
        solution, but its easy.
        '''
        
        def __init__(self, dim = (10,10,10)):
            self.dim = dim
            
            # False = not occupied, True = occupied.
            self.grid = np.zeros((dim[0],dim[1],dim[2]), dtype=bool)
            
            # Start at origin
            self.start = (0,0,0)
            
            # Set sentinel
            self.current = self.start
            
            # No established end point.
            self.end = np.nan
            
            # Number of choices at each step. Max of dimension-1 steps.
            self.num_choices = []
            
            # Length of the walk.
            self.length = 0
            
            # Nodes
            self.nodes = deque([self.start])
            
            # Total points on the lattice.
            self.total_points = dim[0]*dim[1]*dim[2]
    
        def __repr__(self):
            return "Self-avoiding walk of length {}".format(self.length)
        
        def __eq__(self, other):
            '''
            Walks are equal precisely if they pass through the same nodes.
            '''
            return self.nodes == other.nodes
        
        def __hash__(self):
            return hash(self.length)
        
        def __len__(self):
            return len(self.nodes)
        
        def check_collision(self, node):
            '''
            Determine whether a given node on the lattice is part of the
            SAW or boundary (if specified).
            '''
            
            # If the point of interest is outside of boundary, return false.
            if node[0]<0 or node[1]<0 or node[2]<0:
                return False
            if node[0] >= self.dim[0] or node[1] >= self.dim[1] or \
            node[2] >= self.dim[2]:
                return False
            
            # Otherwise, find the value on the grid and determine if occupied.
            return not self.grid[node[0]][node[1]][node[2]]
            
            # if self.restricted == False:
            #     return node not in self.nodes
            # else:
            #     return node not in self.nodes and node not in self.boundary
        
        def find_movement_options(self):
            '''
            Check for collisions around current node and determine 
            viable movement options.
            '''
            
            options = []
            if self.check_collision((self.current[0], self.current[1]+1, self.current[2])):
                options.append(0)
            if self.check_collision((self.current[0], self.current[1]-1, self.current[2])):
                options.append(1)
            if self.check_collision((self.current[0]-1, self.current[1], self.current[2])):
                options.append(2)
            if self.check_collision((self.current[0]+1, self.current[1], self.current[2])):
                options.append(3)
            if self.check_collision((self.current[0], self.current[1], self.current[2]+1)):
                options.append(4)
            if self.check_collision((self.current[0], self.current[1], self.current[2]-1)):
                options.append(5)
            
            if options == []:
                return -1
            else:
                return options
        
        
        def move(self, direction):
            '''
            Move in a given direction. 
            
            0 - Up y
            1 - Down y
            2 - Left x
            3 - Right x
            4 - Elevation+ z
            5 - Elevation- z
            '''
    
            if direction == 0:
                self.current = (self.current[0], self.current[1]+1, self.current[2])
                self.grid[self.current[0]][self.current[1]][self.current[2]] = True
                self.nodes.append(self.current)
                
            elif direction == 1:
                self.current = (self.current[0], self.current[1]-1, self.current[2])
                self.grid[self.current[0]][self.current[1]][self.current[2]] = True 
                self.nodes.append(self.current)
    
            elif direction == 2:
                self.current = (self.current[0]-1, self.current[1], self.current[2])
                self.grid[self.current[0]][self.current[1]][self.current[2]] = True
                self.nodes.append(self.current)
    
            elif direction == 3:
                self.current = (self.current[0]+1, self.current[1], self.current[2])
                self.grid[self.current[0]][self.current[1]][self.current[2]] = True
                self.nodes.append(self.current)
                
            elif direction == 4:
                self.current = (self.current[0], self.current[1], self.current[2]+1)
                self.grid[self.current[0]][self.current[1]][self.current[2]] = True
                self.nodes.append(self.current)
                
            elif direction == 5:
                self.current = (self.current[0], self.current[1], self.current[2]-1)
                self.grid[self.current[0]][self.current[1]][self.current[2]] = True
                self.nodes.append(self.current)
            else:
                return False
            
            self.length = self.length+1
            return True
    
        
    
        def run1(self, index = None):
            '''
            run1: SAW keeps walking until out of options
            '''
            
            while(True):
                
                options = self.find_movement_options()
                
                if options == -1:
                    break
                _LEN = len(options)
                self.num_choices.append(_LEN)
                probs = [1/_LEN]*_LEN
                direction = choice(options, size=1, p = probs)
                self.move(direction)
                
                            
            #self.nodes = tuple(self.nodes)
            
            result = 1
            for item in self.num_choices:
                result = result*item
            
            return result
            #return self
        
        
        def run2(self, index = None):
            '''
            run2: SAW starts by picking a maximum length, then walks until 
            either that length is achieved or it runs out of possible moves 
            (i.e. gets trapped).
            '''
            
            # Pick a length at random. Either move until you can't move anymore
            # or until you reach desired length.
            total_length = np.random.randint(1,self.total_points)
            while(self.length < total_length):
                
                options = self.find_movement_options()
                if options == -1:
                    break
                _LEN = len(options)
                self.num_choices.append(_LEN)
                probs = [1/_LEN]*_LEN
                direction = choice(options, size=1, p = probs)
                self.move(direction)
                
                            
            #self.nodes = tuple(self.nodes)
            result = 1
            for item in self.num_choices:
                result = result*item
            
            return result
            #return self
        
        def run3(self, stop_prob=0.1, index=None):
            '''
            run3: Run until collision or stop abruptly with small probability.
            '''
            
            while(True):
                
                U = np.random.uniform(0,1,1)
                options = self.find_movement_options()
                if options == -1 or U < stop_prob:
                    break
                
                self.num_choices.append(len(options))
                probs = [1/len(options)]*len(options)
                direction = choice(options, size=1, p = probs)
                self.move(direction)
                
            #self.nodes = tuple(self.nodes)
            #del self.boundary
            
            result = 1
            for item in self.num_choices:
                result = result*item
            
            return result
            #return self
            
            
        
        def draw(self, show=True):
            '''
            Draw the SAW.
            '''
            
            x = [node[0] for node in self.nodes]
            y = [node[1] for node in self.nodes]
            z = [node[2] for node in self.nodes]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            p1 = ax.plot(x,y,z)
            p2 = ax.scatter(x,y,z,s=50)
            
            # Set integer ticks and window size.
            plt.xlim((0, self.dim[0]))
            plt.xticks(range(0,self.dim[0]))
            plt.ylim((0,self.dim[1]))
            plt.yticks(range(0,self.dim[1]))    
            plt.ylim((0,self.dim[1]))
            plt.yticks(range(0,self.dim[1])) 
            ax.set_zlim(0,self.dim[2])
            ax.set_zticks(range(0,self.dim[2]))            
            
            if show == True:
                plt.show()
            
            return [p1, p2]
        
            