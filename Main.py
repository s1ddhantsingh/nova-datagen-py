'''
Date: 4/2/2023

Do animations with a react-app

'''

from Simulator import *
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd
import time

import mysql.connector as mc

# import streamlit

class Main:
    def __init__(self, filename):
        self.file_name=filename
        '''the simulate class will load each object simulator'''
        simulate = Simulate('input.json')
        
        '''
        parameters are set as
        @param:
        1. generate_per_cycle |--> 60 (simulates 60 hz system)
        2. number_of_cycles |--> 5 (five * time_delta TOTAL TIME)
        3. time_delta |--> 0.5sec pause per cycle
        '''
        # simulate.start(60, 5, 0.5)
        self.start(simulate, 100, 60, 0)

    '''run the the overall simulator function'''
    def start(self, simulate, number_of_cycles, generate_per_cycle, time_delta):
        for _ in range(number_of_cycles):
            time.sleep(time_delta)
            simulate.generate(generate_per_cycle, True) # appends to the end of the file

            # self.animate('electricity', 'datasources.csv')

    # def animate(self, stream, filename):
    #     fig, ax = plt.subplots()

    #     # with open(self.file_name, 'r') as file:
    #     print('reached')
    #     dataframe = pd.read_csv(filename)
    #     print('reached 2')
    #     x = [i for i in range(len(dataframe))]
    #     y = dataframe.loc[:, stream] # value
    
    #     print(x, y)

    #     plt.plot(x, y)
    #     plt.show()
    #     time.sleep(3)

Main('input.json')