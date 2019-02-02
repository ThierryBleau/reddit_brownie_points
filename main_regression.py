import numpy as np
import matplotlib.pyplot as plt
import data_preparation
import time
#np.set_printoptions(threshold=np.nan)

training_data, hfwords = data_preparation.main2() 
'''training_data, hfwords = data_preparation.main2() ''' #use this for the new features
#traning data, vector with popular words
#traning data is a list of dictionaries, dictionaries have everything+ a 2x160 matrix called 'matrix'3
training_data, validation, testing = data_preparation.partition(training_data,10000)
#potential feature: proximity of popular word to the beginning, number of words in comment, 

def get_vector(data,feature):
    #makes a list with only the features in it
    y = []
    for i in range(len(data)):
        y.append(data[i][feature])
    return(y)

#convert it to a np vector
y = np.array(get_vector(training_data,'popularity_score'))

def data_ironing(data):
    #makes a vector with only 1-s.lenght is length of data
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

        elif col == 'non_linear':
            vector = get_vector(data,'non_linear')
            
        else:
            continue
        #print(vector)
        new_matrix = np.vstack((new_matrix,vector))
    occ_matrix = get_occurence_matrix(data)         #160x10000 matrix with occurence_vectors in  the row
    occ_matrix = occ_matrix[0:59]
    new_matrix = np.vstack((new_matrix,occ_matrix))
    return(new_matrix)


def get_occurence_matrix(data):
    #makes a 160x10000 matrix with the 'matrix' dictionary entries in the rows
    occurence_matrix = []
    for i in range(len(data)):
        occurence_matrix.append([])
        for hfword in data[i]['matrix']:
            occurence_matrix[i].append(hfword[1])
    return(np.array(occurence_matrix).T)

def closed_form(x,y): 
    # note to group b4 submittion: the shape of the input x must be given in the form
    # (data,features) i.e (10000, 65) and not (65, 10000)... bc it makes more 
    # sense to me... thierrys data_ironing returns (features, data)
    # ### will delet comment b4 submittion ###
    start_time = time.time()
    ones = np.ones((x.shape[0],1)) 
    X = np.column_stack((x, ones))
    xTy = np.dot(X.T, y)
    xTx = np.dot(X.T, X)
    xTxinv = np.linalg.pinv(xTx) 

    w_est = np.dot(xTxinv,xTy)
    runtime = time.time()- start_time
    return (w_est, runtime)


def matrix_gradient(XTX,XTy,weight_vector):
    XTXw = np.dot(XTX,weight_vector)
    #def of gradient
    return(XTXw-XTy)

x = data_ironing(training_data)

def main_gradient_function(in_eta_0,in_epsilon,dynamic):
    num_features = x.shape[0] #163 coming from data + bias term
    diff = 0
    norm_diff = 100000
    beta = 0
    eta_0 = in_eta_0
    epsilon = in_epsilon
    weight_vector = np.ones(num_features)
    XTX = np.dot(x,x.T)
    XTy = np.dot(x,y)
    counter = 0

    start_time = time.time()
    while norm_diff > epsilon:
        if dynamic == True:
            beta = 0.01 * counter
        alpha = eta_0/(1+beta)
        new_weight_vector = weight_vector - 2*alpha*matrix_gradient(XTX,XTy,weight_vector)
        diff = new_weight_vector - weight_vector
        norm_diff = np.linalg.norm(diff)
        #print(norm_diff)
        weight_vector = new_weight_vector
        #print(weight_vector.shape)
        #print(weight_vector[0:4])
        counter = counter +1
    #print(weight_vector)
    #print(counter)
    #print(weight_vector[0:4])
    runtime = time.time()-start_time
    return(weight_vector, runtime)

def prediction(x,w):
    pop_pred = np.dot(x.T,w)
    return(pop_pred)

def prediction_closed(x, w):
    ones = np.ones((x.shape[0],1)) 
    X = np.column_stack((x, ones))

    p = np.dot(X,w)
    return p

def least_squares(pred,validation_data):
    y = get_vector(validation_data,'popularity_score')
    #print(y)
    #print(pred)
    mse = np.mean((y - pred)**2)
    return(mse)

def testing_function(in_eta_0,in_epsilon,dynamic):
    w , runtime= main_gradient_function(in_eta_0,in_epsilon,dynamic)
    pred = prediction(data_ironing(validation),w)
    error = least_squares(pred,validation)
    return(error,runtime)

def test_closed():
    w, runt = closed_form(x.T, y)
    x_bar = data_ironing(validation).T
    pred = prediction_closed(x_bar, w)
    err = least_squares(pred, validation)
    return(err,runt)
