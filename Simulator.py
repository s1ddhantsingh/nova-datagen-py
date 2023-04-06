'''
Date: 4/2/2023
Program: NOVA RVG
    i. 3+ Data Modalities 
        a. Rotor Bearing Systems 
        b. Electricity outputs
        c. Haas Engines
    ii. init inputs:
        a. PASS IN A JSON OBJECT
            JSON Object Elements:
                1. generate_datastream_for --> includes the titles of ALL types of data information
                2. for each datastream component:
                    a. type --> what type of data is this
                    b. min --> what is the minimum produced for the average range level?
                    c. step --> how OFTEN should this data change (seconds)
        b. write to a file UNTIL process is closed, this file will be the background process in the Main.py python file
'''
import random
import json
import time

# import mysql.connector as mc
# db = mc.connect(host = "localhost", user = "root", password = "root")

class Simulator:
    '''initializer function to setup the stream simulator'''
    def __init__(self, seed, mean, std_dev, step_sz, err_rate, err_len, smin, smax) -> None:
        self.mean = mean
        self.std_dev = std_dev
        self.step_sz = step_sz
        self.val = self.mean - random.random()
        self.default_err_rate = err_rate
        self.default_err_len = err_len
        self.current_err_rate = err_rate
        self.current_err_len = err_len
        self.min = smin
        self.max = smax
        self.is_curr_err = False

        self.last_none_err_val = 0.0
        self.factors = [-1, 1]

        self.val_cnt = 0
        self.err_cnt = 0

        random.seed(seed)

    '''calculate the next value in the stream'''
    def calc_next_val(self):
        self.current_err_len -= 1
        if self.current_err_rate == self.current_err_len and self.current_err_rate < 0.01:
            self.current_err_rate = 0.01
        nextIsError = random.random() < self.current_err_rate
        if nextIsError:
           if not self.is_curr_err:
               self.last_none_err_val = self.val
           self.new_err_val()
        else:
           self.new_val()
           if self.is_curr_err:
               self.is_curr_err = False
               self.current_err_rate = self.default_err_rate
               self.current_err_len = self.default_err_len
        return self.val

    '''helper function <--> generate & output a new value'''
    def new_val(self):
        self.val_cnt += 1
        delta_val = random.random() * self.step_sz
        factor = self.factors[0 if random.random() < 0.5 else 1]
        if self.is_curr_err:
            self.val = self.last_none_err_val
        self.val += delta_val * factor

    '''generate a random error value <--> generate & output a new random error value'''
    def new_err_val(self):
        self.err_cnt += 1
        if not self.is_curr_err:
            if self.val < self.mean:
                self.val = random.random() * (self.mean - 3 * self.std_dev - self.min) + self.min
            else:
                self.val = random.random() * (self.max - self.mean - 3 * self.std_dev) + self.mean + self.std_dev
        else:
            delta_val = random.random() * self.step_sz
            factor = self.factors[0 if random.random() < 0.5 else 1]
            self.val += delta_val * factor()

class Simulate:
    '''initialize the simulate function, the OVERARCHING FUNCTION TO CALL'''
    def __init__(self, filename) -> None:
        '''initial state'''
        file=open(filename) # need to create an input.json input for ANY queries, 
        data=json.load(file)

        '''outputting all generated data to:'''
        wfile=open("datasources.csv","w")

        '''initialize stream Simulators'''
        self.simulators=[]
        for stream in data['generate_datastreams_for']:

            err_rate = 1 # error_rate of 10%

            randseed, mean, std_dev, step_sz, err_len, smin, smax = \
                random.randint(0, 10e99),\
                (( data[stream]['max'] - data[stream]['min'] )/2), \
                2,\
                data[stream]['step'],\
                5,\
                ( data[stream]['min'] ),\
                ( data[stream]['max'] )
            s = Simulator(randseed, mean, std_dev, step_sz, err_rate, err_len, smin, smax)
            self.simulators.append(s)
            print(data[stream]['abbrev'])

        str_title=','.join(str(e) for e in data['generate_datastreams_for'])
        wfile.write(str_title)

    '''generate data randomly with random epoch_size + bool_append'''
    def generate(self, epoch_size, bool_append):
        
        analytics = Analytics('analytics.csv', True)

        '''start generating data with a predefined epoch_size'''    
        wfile=open("datasources.csv", 'a+' if bool_append else 'w')

        for second in range(epoch_size):
            wfile.write('\n')
            current_data = []
            for simulator in self.simulators:
                current_data.append(simulator.calc_next_val())
            wfile.write(','.join(str(e) for e in current_data))
        wfile.close()

        analytics.generate_analytics()


'''

TODO: Finish analytics class -- python implementation

'''
class Analytics:
    def __init__(self, filename, bool_append):
        self.file=filename
        wfile=open("datasources.csv", 'a+' if bool_append else 'w')

    # store analytics measures in this file...
    def generate_analytics(self):
        # create data-saving analytics pipline
        pass




