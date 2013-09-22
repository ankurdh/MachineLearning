import math
from collections import Counter
from heapq import nsmallest

class LetterData:
    def __init__(self, letterData=[]):
        self.lettr = letterData[0]
        self.xbox = int(letterData[1])
        self.ybox = int(letterData[2])
        self.width = int(letterData[3])
        self.high = int(letterData[4])
        self.onpix = int(letterData[5])
        self.xbar = int(letterData[6])
        self.ybar = int(letterData[7])
        self.x2bar = int(letterData[8])
        self.y2bar = int(letterData[9])
        self.xybar = int(letterData[10])
        self.x2ybr = int(letterData[11])
        self.xy2br = int(letterData[12])
        self.xege = int(letterData[13])
        self.xegvy = int(letterData[14])
        self.yege = int(letterData[15])
        self.yegvx = int(letterData[16])
        
        self.dataMap = {
                            0: self.xbox,
                            1: self.ybox,
                            2: self.width,
                            3: self.high,
                            4: self.onpix,
                            5: self.xbar,
                            6: self.ybar,
                            7: self.x2bar,
                            8: self.y2bar,
                            9: self.xybar,
                            10: self.x2ybr,
                            11: self.xy2br,
                            12: self.xege,
                            13: self.xegvy,
                            14: self.yege,
                            15: self.yegvx            
                        }
        
    def getAsPoints(self):
        return (self.xbox, self.ybox, self.width, self.high, self.onpix, self.xbar, self.ybar, self.x2bar, self.y2bar, self.xybar, self.x2ybr, self.xy2br,
                self.xege, self.xegvy, self.yege, self.yegvx)
    
    def __len__(self):
        return 16
    
    def __getitem__(self, i):
        return self.dataMap[i] 
#         return (self.xbox, self.ybox, self.width, self.high, self.onpix, self.xbar, self.ybar, self.x2bar, self.y2bar, self.xybar, self.x2ybr, 
# self.xy2br,
#                 self.xege, self.xegvy, self.yege, self.yegvx)
        
def findEuclidianDifference(point1, point2):
    p1 = pow(point1.xbox - point2.xbox, 2)
    p2 = pow(point1.ybox - point2.ybox, 2)
    p3 = pow(point1.width - point2.width, 2)
    p4 = pow(point1.high - point2.high, 2)
    p5 = pow(point1.onpix - point2.onpix, 2)
    p6 = pow(point1.xbar - point2.xbar, 2)
    p7 = pow(point1.ybar - point2.ybar, 2)
    p8 = pow(point1.x2bar - point2.x2bar, 2)
    p9 = pow(point1.y2bar - point2.y2bar, 2)
    p10 = pow(point1.xybar - point2.xybar, 2)
    p11 = pow(point1.x2ybr - point2.x2ybr, 2)
    p12 = pow(point1.xy2br - point2.xy2br, 2)
    p13 = pow(point1.xege - point2.xege, 2)
    p14 = pow(point1.xegvy - point2.xegvy, 2)
    p15 = pow(point1.yege - point2.yege, 2)
    p16 = pow(point1.yegvx - point2.yegvx, 2)
    
    sumOfSquares = p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14 + p15 + p16 
    return math.sqrt(sumOfSquares)

trainingData = []
counter = 0

def getTrainingData():
    return trainingData

def getDataElementFrom(line):
    splitLine = line.split(",")
    return LetterData(splitLine)
    
def addTrainingData(line):
    trainingData.append(getDataElementFrom(line))
    
def evaluateLine(line, k):
    
    class DistanceLetterType:
        def __init__(self, distance, letter):
            self.euclidianDistance = distance
            self.lettr = letter
            
    minEuclidianDistanceList = []
    
    currentElement = getDataElementFrom(line)
    for x in trainingData:
        currentDistance = findEuclidianDifference(currentElement, x)
        obj = DistanceLetterType(currentDistance, x.lettr)
        minEuclidianDistanceList.append(obj)
    
    """
    create a heap out of the list. The python documentation says the heapify method creates the heap
    in-place and in linear time.
    """    
#     heapify(minEuclidianDistanceList)
    currentLetter = currentElement.lettr
    nKKs = nsmallest(k, minEuclidianDistanceList, key=lambda obj: obj.euclidianDistance)
    
    if k == 1:
        
#         print "Letter '%s' has the closest match to : '%s', at distance of '%0.2f' on the feature space." % (currentElement.lettr, nKKs[0].lettr, nKKs[0].euclidianDistance)
        matchedLetter = nKKs[0].lettr
    else:
        counter = Counter()
        for i in range(k):
            counter[nKKs[i].lettr] += 1
        
#         print "Letter '" + currentElement.lettr + "' has the closest match to : " + counter.most_common(1)[0][0] + " with " + str(counter.most_common(1)[0][1]) + " out of " + str(k) + " occurences."
        matchedLetter = counter.most_common(1)[0][0]
        
    return [currentLetter, matchedLetter]
