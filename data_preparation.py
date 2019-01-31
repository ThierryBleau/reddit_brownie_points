import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
from collections import Counter 

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

# Example:
data_point = data[0] # select the first data point in the dataset

# Now we print all the information about this datapoint
#for info_name, info_value in data_point.items():
    #print(info_name + " : " + str(info_value))
    

def partition(data,split_point):
    #partitions the data into train, valid and test
    train = data[0:split_point]
    valid = data[split_point:split_point+((len(data)-split_point)//2)]
    test = data[split_point+((len(data)-split_point)//2):len(data)]
    return train, valid, test



def prep(data_input):
    #makes text to lowercase and splits into a list
    #question: should we remove end of sentence ? or !
    for i in range(0,len(data_input)):
        data_input[i]['text'] = data_input[i]['text'].lower().split()
    return data_input

def prep2(data):
    for i in range(0,len(data)):
        data[i]['char_lenght']=len(data[i]['text'])
        data[i]['text'] = data[i]['text'].lower().split()
        data[i]['n_o_words'] = len(data[i]['text'])  
        #NOTE if we put this to the regression and the next one, the error will be 18
        #maybe because they have a high covariance
        data[i]['avg_char_len']=data[i]['char_lenght']/data[i]['n_o_words']  
        data[i]['non_linear']=data[i]['is_root']*data[i]['avg_char_len']      
    return data


def pooled_words(data,n_words):
    #makes a list containing every word
    from collections import Counter
    all_words=[]
    for i in range(0,len(data)):
        all_words.append(data[i]['text'])
    #first makes a list of lists, then a 'flat list'
    flat_list = [item for sublist in all_words for item in sublist]

    filtered = filter(lambda x: len(x)>2, flat_list)
    
    #calculates word count
    Counter = Counter(filtered)
    most_occur = Counter.most_common(n_words)

    #creates a list the most occured words
    most_occured_n_words = []
    for i in range(n_words):
        most_occured_n_words.append(most_occur[i][0])
        
    return(most_occured_n_words)




def feature_most_occured(data,n_words, most_occured_words):
    #creates a matrix for every dict member; size is 2 rows, 160 colums
    #1st row is index, 2nd is word's number of occurence in the comment
    for i in range(len(data)):
        data[i]['index']=[]
        data[i]['occurence']=[]
        for j in range(n_words):
            data[i]['index'].append(j)
            data[i]['occurence'].append(data[i]['text'].count(most_occured_words[j]))
        data[i]['matrix'] = np.column_stack((data[i]['index'], data[i]['occurence']))
        del(data[i]['index'])
        del(data[i]['occurence'])
    return data




#NOT USED
def order_of_most_occured(data,n_words, most_occured_words):
    #creates a matrix for every dict member; size is 2 rows, 160 colums
    #1st row is index, 2nd is word's number of occurence in the comment
    for i in range(len(data)):
        data[i]['index']=[]
        data[i]['occurence']=[]
        data[i]['ordering']=[]

        for j in range(n_words):
            try:
                data[i]['index'].append(j)
                data[i]['occurence'].append(data[i]['text'].count(most_occured_words[j]))
                data[i]['ordering'].append(data[i]['text'].index(most_occured_words[j])+1)
            except ValueError:
                data[i]['ordering'].append(0)
        data[i]['matrix'] = np.column_stack((data[i]['index'], data[i]['occurence'], data[i]['ordering']))
        del(data[i]['index'])
        del(data[i]['occurence'])
        del(data[i]['ordering'])

    return data

def main():
    with open("proj1_data.json") as fp:
        data = json.load(fp)
    
    data_prepped = prep(data)
    most_occured_words = pooled_words(data_prepped,160)
    data = feature_most_occured(data_prepped,160, most_occured_words)   
    data = text_lenght_feature(data)
    return(data_prepped,most_occured_words)




def main2():
    with open("proj1_data.json") as fp:
        data = json.load(fp)
    
    data_prepped = prep2(data)
    most_occured_words = pooled_words(data_prepped,160)
    data = feature_most_occured(data_prepped,160,most_occured_words)
    #call the next line of you want to add the ordering of the most occured words or distance
    #data = order_of_most_occured(data_prepped,160, most_occured_words)
    return(data,most_occured_words)


