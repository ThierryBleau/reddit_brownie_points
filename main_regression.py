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
        if col == 'is_root':
            vector = get_vector(data,'is_root')
            
        elif col == 'controversiality':
            vector = get_vector(data,'controversiality')
            
        elif col == 'children':
            vector = get_vector(data,'children')
            
        else:
            continue
        new_matrix = np.vstack((new_matrix,vector))
    occ_matrix = get_occurence_matrix(data)
    new_matrix = np.vstack((new_matrix,occ_matrix.T))
    print(new_matrix.shape)
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

def matrix_gradient(x,y,weight_vector):
    term1 = np.dot(x.T,x).dot(weight_vector)
    term2 = np.dot(x.T,y)
    return(term1-term2)

def main_gradient_function():
    x = data_ironing(raw)
    num_features = x.shape[0]
    diff = 0
    beta = 0
    eta_0 = 0.001
    epsilon = 0.01
    weight_vector = np.zeros(num_features)
    alpha = eta_0/(1+beta)
    while diff < epsilon:
        new_weight_vector = weight_vector - 2*alpha*matrix_gradient(x,y,weight_vector)
        diff = new_weight_vector - weight_vector
        weight_vector = new_weight_vector
        print(weight_vector)
    return(weight_vector)


print(main_gradient_function())