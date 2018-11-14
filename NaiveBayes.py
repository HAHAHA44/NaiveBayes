import numpy as np
import math
class NaiveBayes():
    def __init__(self):
        pass

    def createDict(self,data,vec):
        ret = [[],[]]
        ret[0] = [0]*14
        ret[1] = [0]*14
        #discrete values 
        for i in [1,3,5,6,7,8,9,13]:
            ret[0][i] = dict()
            ret[1][i] = dict()
            sum = [0,0]
            for j in range(0,len(data)):
                if data[j][i] != '?':
                    sum[vec[j]] = sum[vec[j]] + 1
                    if data[j][i] not in ret[vec[j]][i]:
                        ret[vec[j]][i][data[j][i]] = 0
                    ret[vec[j]][i][data[j][i]] = ret[vec[j]][i][data[j][i]] + 1
            for k in ret[0][i]:
                ret[0][i][k] = ret[0][i][k] / sum[0]
            for k in ret[1][i]:
                ret[1][i][k] = ret[1][i][k] / sum[1]

        #continuous values
        for i in [0,2,4,10,11,12]:
            ret[0][i] = [0,0]
            ret[1][i] = [0,0]
            a = [[],[]]
            for j in range(0,len(data)):
                a[vec[j]].append(int(data[j][i]))
            ret[0][i][0] = np.mean(a[0])
            ret[0][i][1] = np.var(a[0])
            ret[1][i][0] = np.mean(a[1])
            ret[1][i][1] = np.var(a[1])
        return ret

    def predict(self, testset, voclist):
        ret = []
        for data in testset:
            y = [1,1]
            for i in [1,3,5,6,7,8,9,13]:
                for k in [0,1]:
                    if data[i] in voclist[k][i]:
                        y[k] = voclist[k][i][data[i]] * y[k]
            
            for i in [0,2,4,10,11,12]:
                for k in [0,1]:
                    m = pow((int(data[i])-voclist[k][i][0]),2)
                    n = 2*(voclist[k][i][1])
                    a = math.exp(-1 * m/n)
                    b = math.sqrt(2*math.pi*(voclist[k][i][1]))
                    p = a/b
                    y[k] = y[k] * p
            if y[0] > y[1]:
                ret.append(0)
            else:
                ret.append(1)
        
        return ret

    def showres(self,myans,testans):
        c = 0
        for i in range(0,len(myans)):
            if myans[i] == testans[i]:
                c = c+1
        print(len(myans),' ',len(testans))
        print('accuracy = ',c/len(myans))



def loadDataSet(fileName):
    data = []
    vec = []
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.split(', ')
        data.append(curLine[:14])
        if curLine[14] == '>50K\n' or curLine[14] == '>50K' :
            vec.append(1)
        else:
            vec.append(0)
    return data,vec

def loadTestSet(fileName):
    data = []
    vec = []
    fr=open(fileName)
    for line in fr.readlines():
        curLine=line.split(', ')
        data.append(curLine[:14])
        if curLine[14] == '>50K.\n' or curLine[14] == '>50K.' :
            vec.append(1)
        else:
            vec.append(0)
    return data,vec



trainx,trainy = loadDataSet('adult.data')
testx,testy = loadTestSet('Cadult.test')
nb = NaiveBayes()
voc = nb.createDict(trainx,trainy)
ans = nb.predict(testx,voc)
nb.showres(ans,testy)


