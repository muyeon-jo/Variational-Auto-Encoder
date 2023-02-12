from numpy.linalg import norm
import numpy as np
import pickleData
import multiprocessing as mp
import torch

def cosineSimilarity(x,y):
    return np.dot(x,y)/(norm(x)*norm(y))
def multi(embed, start, end, pn):
    print("start "+str(pn))
    sim = []
    count = 0
    for i in range(start, end):
        if count %100 == 0:
            print("{pn} : {count}".format(pn=pn, count=count))
        sim.append([])
        for j in range(len(embed)):
            sim[count].append(cosineSimilarity(embed[i],embed[j]))

        count += 1
    pickleData.pickle_save(sim,"temp"+pn+".pkl")
def merge():
    tt = pickleData.pickle_load("temp0.pkl")
    t1 =pickleData.pickle_load("temp1.pkl")
    t2=pickleData.pickle_load("temp2.pkl")
    t3=pickleData.pickle_load("temp3.pkl")
    t4=pickleData.pickle_load("temp4.pkl")
    t5=pickleData.pickle_load("temp5.pkl")

    tt.extend(t1)
    tt.extend(t2)
    tt.extend(t3)
    tt.extend(t4)
    tt.extend(t5)
    del(t1)
    del(t2)
    del(t3)
    del(t4)
    del(t5)
    print("saving..")
    pickleData.pickle_save(tt,"./content/Embeddings/similarity.pkl")
def getSim():
    userEmbed = pickleData.pickle_load("./content/Embeddings/userEmbed.pkl")
    userLabel = pickleData.pickle_load("./content/Embeddings/userlabel.pkl")
    cateEmbed = pickleData.pickle_load("./content/Embeddings/categoryEmbed.pkl")
    cateLabel = pickleData.pickle_load("./content/Embeddings/categorylabel.pkl")

    fusionEmbed = []
    for i in range(len(userEmbed)):
        fusionEmbed.append(torch.cat([userEmbed[i],cateEmbed[i]]).to("cpu"))
        
    pickleData.pickle_save(fusionEmbed,"./content/Embeddings/fusionEmbed.pkl")
    
    tl = int(len(fusionEmbed)/6)

    #multi(fusionEmbed,0,tl,"0")
    work = [[fusionEmbed,0,tl,"0"],
    [fusionEmbed,tl,tl*2,"1"],
    [fusionEmbed,tl*2,tl*3,"2"],
    [fusionEmbed,tl*3,tl*4,"3"],
    [fusionEmbed,tl*4,tl*5,"4"],
    [fusionEmbed,tl*5,len(fusionEmbed),"5"]]

    pool = mp.Pool(processes=6)
    pool.starmap(multi,work)
    pool.close()
    pool.join()

if __name__ == "__main__":
    merge()