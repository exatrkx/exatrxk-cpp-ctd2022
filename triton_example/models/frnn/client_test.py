#import frnn
import numpy as np
import csv
from numpy import loadtxt
file = open('data/in_e.csv','rb')
input_data = loadtxt(file,delimiter=" ")
with open('data/in_e.csv', newline='') as csvfile:
    input_data0 = list(csv.reader(csvfile))
#new=np.array(input_data0)

print(input_data)
