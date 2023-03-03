from numpy.linalg import norm
import numpy as np
import pickleData
import multiprocessing as mp
import torch
import matplotlib.pyplot as plt
def cosineSimilarity(x,y):
    if norm(x) == 0 or norm(y) == 0:
        return 0
    return np.dot(x,np.transpose(y))/(norm(x)*norm(y))
def saveSimilarity_mp(embed, start, end, pn):
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
    pickleData.pickle_save(sim,"./temp/temp"+pn+".pkl")
def mergeTempData():
    tt = pickleData.pickle_load("./temp/temp0.pkl")
    t1 =pickleData.pickle_load("./temp/temp1.pkl")
    t2=pickleData.pickle_load("./temp/temp2.pkl")
    t3=pickleData.pickle_load("./temp/temp3.pkl")
    t4=pickleData.pickle_load("./temp/temp4.pkl")
    t5=pickleData.pickle_load("./temp/temp5.pkl")

    tt.extend(t1)
    tt.extend(t2)
    tt.extend(t3)
    tt.extend(t4)
    tt.extend(t5)
    print(len(tt))
    del(t1)
    del(t2)
    del(t3)
    del(t4)
    del(t5)
    # print("saving..")
    # pickleData.pickle_save(tt,"./content/Embeddings/similarity.pkl")
    return tt
def getSimilarityEmbed():
    userEmbed = pickleData.pickle_load("./content/Embeddings/userEmbed.pkl")
    cateEmbed = pickleData.pickle_load("./content/Embeddings/categoryEmbed.pkl")

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
    pool.starmap(saveSimilarity_mp,work)
    pool.close()
    pool.join()
def getSimilarityData(dataDirectory:str):
    data = pickleData.pickle_load(dataDirectory)
    tt = data[:4999]
    data=[]
    for i in tt:
        data.append(i[0].to("cpu"))
        #data.append(i)
    tl = int(len(data)/6)
    temp = cosineSimilarity(data[0],data[1])
    work = [[data,0,tl,"0"],
    [data,tl,tl*2,"1"],
    [data,tl*2,tl*3,"2"],
    [data,tl*3,tl*4,"3"],
    [data,tl*4,tl*5,"4"],
    [data,tl*5,len(data),"5"]]

    pool = mp.Pool(processes=6)
    pool.starmap(saveSimilarity_mp,work)
    pool.close()
    pool.join()
def getSortedEmbed():
    cateEmbed = pickleData.pickle_load("./content/Embeddings/mnist latent_label.pkl")
    result = mergeTempData()
    simResult = []
    count = 1
    for row in result:
        temp = []
        for i in range(len(row)):
            temp.append((row[i],cateEmbed[i]))
        
        sor = sorted(temp,key = lambda s: s[0], reverse = True)
        simResult.append(sor)
        if count %1000 == 0:
            print(count)
            pickleData.pickle_save(simResult, "./content/Embeddings/sortedFusionEmbedding"+ str(count)+".pkl")
            simResult = []
        count+=1
        
    pickleData.pickle_save(simResult, "./content/Embeddings/sortedFusionEmbedding_else.pkl")
def getFusionData(index, embedSize:str):
    if index <1000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(1000)+".pkl")
        data = temp[index]
    elif index <2000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(2000)+".pkl")
        data = temp[index-1000]
    elif index <3000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(3000)+".pkl")
        data = temp[index-2000]
    elif index <4000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(4000)+".pkl")
        data = temp[index-3000]
    elif index <5000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(5000)+".pkl")
        data = temp[index-4000]
    elif index <6000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(6000)+".pkl")
        data = temp[index-5000]
    elif index <7000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(7000)+".pkl")
        data = temp[index-6000]
    elif index <8000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(8000)+".pkl")
        data = temp[index-7000]
    elif index <9000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(9000)+".pkl")
        data = temp[index-8000]
    elif index <10000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(10000)+".pkl")
        data = temp[index-9000]
    elif index <11000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(11000)+".pkl")
        data = temp[index-10000]
    elif index <12000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(12000)+".pkl")
        data = temp[index-11000]
    elif index <13000:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding"+ str(13000)+".pkl")
        data = temp[index-12000]
    else:
        temp = pickleData.pickle_load("./content/Embeddings/"+embedSize+"/sortedFusionEmbedding_else.pkl")
        data = temp[index-13000]

    return data
def visualize(index:int,embedSize:str):
    data = getFusionData(index, embedSize)
    visitedArea = pickleData.pickle_load("philadelphia10_visitedArea")
    visitedX = []
    visitedY = []
    area = []
    color = []    
    for i in range(len(visitedArea)):
        for j in range(len(visitedArea[0])):
            if(visitedArea[i][j] > 0.0):
                area.append(visitedArea[i][j]/19453 * 500)
                visitedX.append(j/2)
                visitedY.append(i/2)
                color.append(visitedArea[i][j])
    # plt.matshow(visitedArea)
    # plt.colorbar()
    # plt.show()
    # print(max(area))
    plt.scatter(visitedX, visitedY,s = area, alpha = 0.3, c=color)
    plt.colorbar()

    xplot = []
    yplot = []
    plt.plot(data[0][1][1], data[0][1][0], 'ro')
    plt.text(data[0][1][1],data[0][1][0],"("+str(data[0][1][1])+","+str(data[0][1][0])+")")
    for i in range(1,11):
        xplot.append(data[i][1][1])
        yplot.append(data[i][1][0])
        plt.text(data[i][1][1],data[i][1][0],"("+str(data[i][1][1])+","+str(data[i][1][0])+")")
    plt.plot(xplot, yplot, 'co')
    xplot = []
    yplot = []
    for i in range(11,51):
        xplot.append(data[i][1][1])
        yplot.append(data[i][1][0])
        #plt.text(data[i][1][1],data[i][1][0],"("+str(data[i][1][1])+","+str(data[i][1][0])+")")
    plt.plot(xplot, yplot, 'yo')
    plt.xlabel('X-Index(100m)')
    plt.ylabel('Y-Index(100m)')
    plt.show()

def compareSimilarArea(key1, key2, N:int, src1, src2):
    posDict = pickleData.pickle_load("./content/Embeddings/posDict.pkl")
    index = posDict[key1]
    sim1 = getFusionData(index, src1)

    index = posDict[key2]
    sim2 = getFusionData(index, src2)

    set1 = set()
    set2 = set()
    
    for i in sim1[:N]:
        set1.add(str(i[1]))
    
    for i in sim2[:N]:
        set2.add(str(i[1]))

    print(len(set1&set2))
    print(set1&set2)
def getPositionDict():
    label = pickleData.pickle_load("./content/Embeddings/userlabel.pkl")
    data = dict()
    count= 0
    for i in label:
        data[str(i[0])+","+str(i[1])] = count
        count+=1

    pickleData.pickle_save(data,"./content/Embeddings/posDict.pkl")
def showMnistLatent(data:str):
    latent_list = pickleData.pickle_load(data)
    x = [[],[],[],[],[],[],[],[],[],[]]
    y = [[],[],[],[],[],[],[],[],[],[]]
    sym = ["kv","ro","bo","co","mo","yo","go","ko","rv","gv"]
    lb = ["0","1","2","3","4","5","6","7","8","9"]
    for i, l in latent_list:
        if l == 1:
            symbol = "ro"
            x[1].append((i[0][0].to("cpu")))
            y[1].append(i[0][1].to("cpu"))
        elif l == 2:
            symbol = "bo"
            x[2].append((i[0][0].to("cpu")))
            y[2].append(i[0][1].to("cpu"))
        elif l == 3:
            symbol = "co"
            x[3].append((i[0][0].to("cpu")))
            y[3].append(i[0][1].to("cpu"))
        elif l == 4:
            symbol = "mo"
            x[4].append((i[0][0].to("cpu")))
            y[4].append(i[0][1].to("cpu"))
        elif l == 5:
            symbol = "yo"
            x[5].append((i[0][0].to("cpu")))
            y[5].append(i[0][1].to("cpu"))
        elif l == 6:
            symbol = "go"
            x[6].append((i[0][0].to("cpu")))
            y[6].append(i[0][1].to("cpu"))
        elif l == 7:
            symbol = "ko"
            x[7].append((i[0][0].to("cpu")))
            y[7].append(i[0][1].to("cpu"))
        elif l == 8:
            symbol = "rv"
            x[8].append((i[0][0].to("cpu")))
            y[8].append(i[0][1].to("cpu"))
        elif l == 9:
            symbol = "gv"
            x[9].append((i[0][0].to("cpu")))
            y[9].append(i[0][1].to("cpu"))
        elif l == 0:
            symbol = "kv"
            x[0].append((i[0][0].to("cpu")))
            y[0].append(i[0][1].to("cpu"))
        else:
            continue
    for i in range(10):
        plt.plot(x[i],y[i], sym[i],label = lb[i], markersize = 3)
    plt.legend()
    plt.show()

def similarityMnistLatent(data:str):
    latent_list = pickleData.pickle_load(data)
    x = [[],[],[],[],[],[],[],[],[],[]]
    y = [[],[],[],[],[],[],[],[],[],[]]
    sym = ["kv","ro","bo","co","mo","yo","go","ko","rv","gv"]
    lb = ["0","1","2","3","4","5","6","7","8","9"]
    sim = []
    for i, l in latent_list:
        i

    for i, l in latent_list:
        if l == 1:
            symbol = "ro"
            x[1].append((i[0][0].to("cpu")))
            y[1].append(i[0][1].to("cpu"))
        elif l == 2:
            symbol = "bo"
            x[2].append((i[0][0].to("cpu")))
            y[2].append(i[0][1].to("cpu"))
        elif l == 3:
            symbol = "co"
            x[3].append((i[0][0].to("cpu")))
            y[3].append(i[0][1].to("cpu"))
        elif l == 4:
            symbol = "mo"
            x[4].append((i[0][0].to("cpu")))
            y[4].append(i[0][1].to("cpu"))
        elif l == 5:
            symbol = "yo"
            x[5].append((i[0][0].to("cpu")))
            y[5].append(i[0][1].to("cpu"))
        elif l == 6:
            symbol = "go"
            x[6].append((i[0][0].to("cpu")))
            y[6].append(i[0][1].to("cpu"))
        elif l == 7:
            symbol = "ko"
            x[7].append((i[0][0].to("cpu")))
            y[7].append(i[0][1].to("cpu"))
        elif l == 8:
            symbol = "rv"
            x[8].append((i[0][0].to("cpu")))
            y[8].append(i[0][1].to("cpu"))
        elif l == 9:
            symbol = "gv"
            x[9].append((i[0][0].to("cpu")))
            y[9].append(i[0][1].to("cpu"))
        elif l == 0:
            symbol = "kv"
            x[0].append((i[0][0].to("cpu")))
            y[0].append(i[0][1].to("cpu"))
        else:
            continue
    for i in range(10):
        plt.plot(x[i],y[i], sym[i],label = lb[i], markersize = 3)
    plt.legend()
    plt.show()

def visualizeMNISTLatent(index, folder, di):
    if index <1000:
        temp = pickleData.pickle_load("./content/Embeddings/"+folder+"/sortedFusionEmbedding"+ str(1000)+".pkl")
        data = temp[index]
    elif index <2000:
        temp = pickleData.pickle_load("./content/Embeddings/"+folder+"/sortedFusionEmbedding"+ str(2000)+".pkl")
        data = temp[index-1000]
    elif index <3000:
        temp = pickleData.pickle_load("./content/Embeddings/"+folder+"/sortedFusionEmbedding"+ str(3000)+".pkl")
        data = temp[index-2000]
    elif index <4000:
        temp = pickleData.pickle_load("./content/Embeddings/"+folder+"/sortedFusionEmbedding"+ str(4000)+".pkl")
        data = temp[index-3000]
    elif index <5000:
        temp = pickleData.pickle_load("./content/Embeddings/"+folder+"/sortedFusionEmbedding"+ str(5000)+".pkl")
        data = temp[index-4000]
    else:
        temp = pickleData.pickle_load("./content/Embeddings/"+folder+"/sortedFusionEmbedding_else.pkl")
        data = temp[index-5000]

    xplot = []
    yplot = []
    
    plt.plot(data[0][1][1], data[0][1][0], 'ro')
    plt.text(data[0][1][1],data[0][1][0],"("+str(data[0][1][1])+","+str(data[0][1][0])+")")
    for i in range(1,11):
        xplot.append(data[i][1][1])
        yplot.append(data[i][1][0])
        plt.text(data[i][1][1],data[i][1][0],"("+str(data[i][1][1])+","+str(data[i][1][0])+")")
    plt.plot(xplot, yplot, 'co')
    xplot = []
    yplot = []
    for i in range(11,51):
        xplot.append(data[i][1][1])
        yplot.append(data[i][1][0])
        #plt.text(data[i][1][1],data[i][1][0],"("+str(data[i][1][1])+","+str(data[i][1][0])+")")
    plt.plot(xplot, yplot, 'yo')
    plt.xlabel('X-Index(100m)')
    plt.ylabel('Y-Index(100m)')
    plt.show()

if __name__ == "__main__":
    #getSimilarityData("./content/Embeddings/200m/cate_4/categoryEmbed.pkl")
    getSimilarityData("mnist latent_list.pkl")
    # getSimilarityEmbed()
    getSortedEmbed()
    #getPositionDict()
    # dd = [[10,20,30,40,50],
    # [1,2,3,4,5],
    # [7,2,8,4,1],
    # [88,99,77,44,55]]
    # np.random.shuffle(dd)

    #visualize(2120, "200m/128+128")
    #visualize(2120, "200m/128+128_2")
    #visualize(4000, "userData similarity")
    #visualize(4000, "KLD 0.5/128")
    # 81,244 : 4000 
    # 0,43 : 0
    # 44,212 : 2120
    #compareSimilarArea("0,43","0,43",100,"200m/64+64","200m/64+64_2")
    #showMnistLatent("mnist latent_list.pkl")
    #showMnistLatent("mnist latent_list_0.5.pkl")
