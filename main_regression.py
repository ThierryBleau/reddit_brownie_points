import numpy as np
import matplotlib.pyplot as plt
import data_preparation

raw = data_preparation.main()
def get_y_vector(data):
    y = []
    for i in range(len(data)):
        y.append(data[i]['popularity_score'])
    return(y)

y = np.array(get_y_vector(raw))
print(y)
print(raw[-1])
ones = np.ones(len(raw))
x = np.column_stack((ones,raw))
w = np.dot(np.linalg.pinv(x),y)

regression = w[0] + w[1]*raw
