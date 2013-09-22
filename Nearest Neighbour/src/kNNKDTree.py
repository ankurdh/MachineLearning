'''
Created on Sep 4, 2013

@author: Ankur
'''

import sys
import time

from kdtree import KDTree
from kNNfunctions import getDataElementFrom

#Have variable for calculating accuracy.
properMatches = 0

counter = 0
isPrinted = False
isKDTreeConstructed = False

# declare a 2D array for confusion matrix
confusionMatrix = [[0 for x in xrange(26)] for x in xrange(26)]

listOfPoints = []

with open('training_data.txt', 'r') as f:
    for line in f:
        if counter < 15000:
            listOfPoints.append(getDataElementFrom(line))
            counter += 1
            continue
        
        if isKDTreeConstructed == False:
            start = time.clock()
            kdTree = KDTree.construct_from_data(listOfPoints)
            elapsedForKDTreeConstruction = (time.clock() - start)
            isKDTreeConstructed = True
            print "KDTree constructed in %.2fs" % (elapsedForKDTreeConstruction)
            searchStartTime = time.clock()
            print "Evaluating input data..."
            
        currentLine = getDataElementFrom(line)
        nearest = kdTree.query(currentLine)
        confusionMatrix[ord(currentLine.lettr) - 65][ord(nearest[0].lettr) - 65] += 1
            
        if currentLine.lettr == nearest[0].lettr:
            properMatches += 1
            
        counter += 1
        if counter % 500 == 0:
            print "\tEvaluated %5d entries in %.2fs." % (counter - 15000, time.clock() - searchStartTime)

print "Evaluation of letters with KDTree done in: %.2fmin" % ((time.clock() - searchStartTime)/60)
print "Accuracy of Classification: %.2f%%" %((float(properMatches)/5000.0)*100)

# The file will be closed now. 
print "Confusion Matrix:"
print "%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c" % ('-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
for row in range(26):
    sys.stdout.write("%-5c" %chr(row + 65))
    for col in range(26):
        sys.stdout.write("%-5d" %confusionMatrix[row][col])
    print "\n"
