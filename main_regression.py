import numpy as np
import matplotlib.pyplot as plt
import data_preparation

training_data, hfwords = data_preparation.main2() 
'''training_data, hfwords = data_preparation.main2() ''' #use this for the new features
training_data, validation, testing = data_preparation.partition(training_data,10000)
#potential feature: proximity of popular word to the beginning, number of words in comment, 
def get_vector(data,feature):
    y = []
    for i in range(len(data)):
        y.append(data[i][feature])
    return(y)

y = np.array(get_vector(training_data,'popularity_score'))

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
            
        elif col == 'avg_char_len':
            vector = get_vector(data,'avg_char_len')
            
        else:
            continue
        #print(vector)
        new_matrix = np.vstack((new_matrix,vector))
    occ_matrix = get_occurence_matrix(data)
    new_matrix = np.vstack((new_matrix,occ_matrix))
    return(new_matrix)

def get_occurence_matrix(data):
    occurence_matrix = []
    for i in range(len(data)):
        occurence_matrix.append([])
        for hfword in data[i]['matrix']:
            occurence_matrix[i].append(hfword[1])
    return(np.array(occurence_matrix).T)



def closed_form():
    x = np.column_stack((ones,training_data))
    w = np.dot(np.linalg.pinv(x),y)
    regression = w[0] + w[1]*training_data
    return(regression)

def matrix_gradient(XTX,XTy,weight_vector):
    XTXw = np.dot(XTX,weight_vector)
    return(XTXw-XTy)

x = data_ironing(training_data)

def main_gradient_function():
    num_features = x.shape[0]
    diff = 0
    norm_diff = 100000
    beta = 0
    eta_0 = 0.0000001
    epsilon = 0.001
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
    pop_pred = np.dot(x.T,w)
    print(pop_pred)
    return(pop_pred)

def least_squares(w,validation_data):
    y = get_vector(validation_data,'popularity_score')
    print(y)
    pred = data_ironing(validation)
    print(pred)
    mse = np.mean((y - pred)**2)
    return(mse)

def testing_function():
    w = main_gradient_function()
    pred = prediction(x,w)
    validity = least_squares(pred,validation)
    print(validity)
    return

testing_function()
