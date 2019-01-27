import numpy as np
import matplotlib.pyplot as plt
import data_preparation

raw, hfwords = data_preparation.main()
#potential feature: proximity of popular word to the beginning, number of words in comment, 
def get_vector(data,feature):
    y = []
    for i in range(len(data)):
        y.append(data[i][feature])
    return(y)

y = np.array(get_vector(raw,'popularity_score'))

def data_ironing(data):
    new_matrix = np.ones(len(data))
    for col in data[0]:
        vector = np.array([])
        print(col)
        if col == 'is_root':
            vector = get_vector(data,'is_root')
            print(col)
        elif col == 'controversiality':
            vector = get_vector(data,'controversiality')
            print(col)
        elif col == 'children':
            vector = get_vector(data,'children')
            print(col)
        else:
            break
        print(vector)
        new_matrix = np.hstack((new_matrix,vector))
    new_matrix = np.hstack((new_matrix,get_occurence_matrix(data)))
    return(new_matrix)

def get_occurence_matrix(data):
    occurence_matrix = []
    for i in range(len(data)):
        occurence_matrix.append([])
        for hfword in data[i]['matrix']:
            occurence_matrix[i].append(hfword[1])
    return(np.array(occurence_matrix))



def closed_form():
    x = np.column_stack((ones,raw))
    w = np.dot(np.linalg.pinv(x),y)
    regression = w[0] + w[1]*raw
    return(regression)


#print(get_occurence_matrix(raw).shape)
print(data_ironing(raw)[0])