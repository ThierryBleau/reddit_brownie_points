import numpy as np
import matplotlib.pyplot as plt
import data_preparation
import main_regression
import time




hyperparameters=[(10**-6,10**-4,False),(10**-8,10**-5,False),(10**-9,10**-5,False),(10**-6,10**-4,True),(10**-7,10**-5,True)]


for i in hyperparameters:
    error, runtime = main_regression.testing_function(i[0],i[1],i[2])
    #print('parameters',i)
    print(error,runtime)

	# with open ('benchmark.txt' ,'a',encoding='utf8') as text_file:
	#     text_file.write(' hypermarameter:{0},{1},{2}'.format(i[0],i[1],i[2]))
	#     text_file.write('\n')
	#     text_file.write('error: {0}'.format(error))
	#     text_file.write('\n')
	#     text_file.write('runtime: {0}'.format(runtime))
	#     text_file.write('\n')
	#     text_file.write('###########################################################')
	#     text_file.write('\n')
	#     text_file.write('\n')