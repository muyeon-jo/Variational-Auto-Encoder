import pickle
import numpy as np

def pickle_load(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def pickle_save(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def makeCateData():
    f = pickle_load("./content/POI(philadelphia)/philadelphia10/visitedCategoryPerArea.pkl")
    r = pickle_load("./content/POI(philadelphia)/philadelphia10/cate2Index.pkl")
    label = []
    data = []
    #total = []
    print("category length = "+str(len(r)))
    for i,row in f.items():
        for j, v in row.items():
            if len(v) <= 1:
                continue
            li = np.zeros(len(r))
            sum = 0
            for key, value in v.items():
                sum+=value
            for key, value in v.items():
                li[key] = value/sum
            
            data.append(li)
            label.append([i,j])
    return data , label, len(r)        

def makeUserData():
    f = pickle_load("./content/POI(philadelphia)/philadelphia10/userVisitDataPerArea.pkl")
    r = pickle_load("./content/POI(philadelphia)/philadelphia10/user_id2Index.pkl")
    label = []
    data = []
    print("user length = "+str(len(r)))
    for i,row in f.items():
        for j, v in row.items():
            if len(v) <= 1:
                continue
            li = np.zeros(len(r))
            for key, value in v.items():
                li[key] = value
            
            data.append(li)
            label.append([i,j])
    return data , label, len(r)