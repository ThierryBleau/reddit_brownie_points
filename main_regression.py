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
        #print(vector)
        new_matrix = np.vstack((new_matrix,vector))
    print(new_matrix.shape)
    occ_matrix = get_occurence_matrix(data)
    print(occ_matrix.shape)
    new_matrix = np.vstack((new_matrix,occ_matrix))
    print(new_matrix.shape)
    return(new_matrix)

def get_occurence_matrix(data):
    occurence_matrix = []
    for i in range(len(data)):
        occurence_matrix.append([])
        for hfword in data[i]['matrix']:
            occurence_matrix[i].append(hfword[1])
    return(np.array(occurence_matrix).T)



def closed_form():
    x = np.column_stack((ones,raw))
    w = np.dot(np.linalg.pinv(x),y)
    regression = w[0] + w[1]*raw
    return(regression)

def matrix_gradient(XTX,XTy,weight_vector):
    
    XTXw = np.dot(XTX,weight_vector)
    
    return(XTXw-XTy)

def main_gradient_function():
    x = data_ironing(raw)
    num_features = x.shape[0]
    diff = 0
    norm_diff = 100000
    beta = 0
    eta_0 = 0.000001
    epsilon = 0.01
    weight_vector = np.ones(num_features)
    alpha = eta_0/(1+beta)
    XTX = np.dot(x,x.T)
    XTy = np.dot(x,y)
    counter = 0
    while norm_diff > epsilon:
        new_weight_vector = weight_vector - 2*alpha*matrix_gradient(XTX,XTy,weight_vector)
        diff = new_weight_vector - weight_vector
        norm_diff = np.linalg.norm(diff)
        print(norm_diff)
        weight_vector = new_weight_vector
        #print(weight_vector.shape)
        #print(weight_vector)
        counter = counter +1
    print(weight_vector)
    print(counter)
    return(weight_vector)

def prediction(x,w):
    pop_pred = np.dot(x,w)
    return(pop_pred)



main_gradient_function()