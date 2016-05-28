from numpy import *
def loadSimpData():
    dataMat = matrix([[1., 2.1],[2., 1.1],[1.3, 1.],[1.,1.],[2.,1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen]<=threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen]>=threshVal] = -1.0
    return retArray

def dif(matIn1, matIn2):
    n = shape(matIn1)[0]
    ret = 0
    for i in range(n):
        if(matIn1[i] != matIn2[i]):
           ret += 1
    return ret

def buildStrump(dataArr, classLabels):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    mindif = m;
    ret = {}
    numStep = 10.0
    for i in range(n):
        minI = min(dataMatrix[:,i])
        maxI = max(dataMatrix[:, i])
        ofstep = (maxI-minI)/numStep
        threshIneq = ['lt','bt']
        j = 0
        while j < maxI:
            for k in threshIneq:
                threshVal = minI+j*ofstep
                retlabel = stumpClassify(dataMatrix, i, threshVal, 'lt')
                retdif = dif(classLabels,retlabel)
                if retdif < mindif:
                    mindif = retdif
                    ret['dim'] = i;ret['val'] = threshVal;ret['neq'] = k
            j += 1
    return ret

data, label = loadSimpData()
print buildStrump(data, label)
print data
print label