'''
Created on Sep 4, 2013

@author: Ankur
'''

import sys
import time

from kNNfunctions import addTrainingData
from kNNfunctions import evaluateLine
from kNNfunctions import getTrainingData
from kNNfunctions import getDataElementFrom
from kdtree import KDTree

#have a counter for keeping track of number of lines read from training data.
counter = 0

#Have variable for calculating accuracy.
properMatches = 0

#metadata variables. 
isPrinted = False
isFirst = True

# declare a 2D array for confusion matrix
confusionMatrix = [[0 for x in xrange(26)] for x in xrange(26)] 
condensedData = []

with open('training_data.txt', 'r') as f:
    print "Building condensed training data.."
    start = time.clock()
    for line in f:
        if counter < 15000:
            ''' Add the first line blindly '''
            if isFirst :
                addTrainingData(line)
                isFirst = False
                counter += 1
                continue
            
            '''
            Check each line. If the match result is wrong, then add it to the training set. 
            '''
            matchResults = evaluateLine(line, 1)
            if matchResults[0] != matchResults[1]:
                addTrainingData(line)
            
            counter += 1
            if (counter % 3000) == 0:
                print "\tEvaluated %d rows for condensed training set in %.2fs" %(counter, time.clock() - start)
                
            continue
        
        elif isPrinted == False:
            elapsed = (time.clock() - start)
            print ("Condensed Training data mapped to feature space in %.4fmin." % (elapsed/60))
            print ("Boundary points evaluated: %d" % getTrainingData().__len__())
            classificationTimeStart = time.clock()
            isPrinted = True
            
            # Implement the search of the next 5000 elements using a KDTree
            searchStartTime = time.clock()
            
            #test: Trying with KDTree
            kdTree = KDTree.construct_from_data(getTrainingData())
            
        else:
            currentLine = getDataElementFrom(line)
#             nearest = evaluateLine(line, 1)
            nearest = kdTree.query(currentLine)
            confusionMatrix[ord(currentLine.lettr) - 65][ord(nearest[0].lettr) - 65] += 1
            
            if currentLine.lettr == nearest[0].lettr:
                properMatches += 1
#             confusionMatrix[ord(nearest[0]) - 65][ord(nearest[1]) - 65] += 1
#             
#             if nearest[0] == nearest[1]:
#                 properMatches += 1
            
            counter += 1
            if counter % 500 == 0:
                print "\tEvaluated %d entries in %.2fs." % (counter - 15000, time.clock() - searchStartTime)

print "Evaluation of letters with condensed training set done in: %.2fmin" % ((time.clock() - searchStartTime) / 60)

print "Accuracy of algorithm: %.2f%%" %((float(properMatches)/5000.0)*100)

print "Confusion Matrix:"
print "%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c" % ('-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
for row in range(26):
    sys.stdout.write("%-5c" %chr(row + 65)) 
    for col in range(26):
        sys.stdout.write("%-5d" %confusionMatrix[row][col])
    print "\n"
