'''
Created on Sep 4, 2013

@author: Ankur
'''

import sys
import time

from kNNfunctions import addTrainingData
from kNNfunctions import evaluateLine

k = int(sys.argv[1])

#Have variable for calculating accuracy.
properMatches = 0

print 'Evaluating kNN algorithm for k:', str(k)

print "Mapping training data to feature space..."

counter = 0
isPrinted = False

#declare a 2D array for confusion matrix
confusionMatrix = [[0 for x in xrange(26)] for x in xrange(26)] 

with open('training_data.txt', 'r') as f:
    start = time.clock()
    for line in f:
        if counter < 15000:
            addTrainingData(line)
            counter += 1
            
            if (counter % 3000) == 0:
                print "\tMapped %d rows to feature space." %counter
            continue
        if isPrinted == False:
            elapsed = (time.clock() - start)
            print ("Training data mapped to feature space in %.4fs." % elapsed)
            print "Evaluating input data .."
            classificationTimeStart = time.clock()
            isPrinted = True
        matchResults = evaluateLine(line, k)
        confusionMatrix[ord(matchResults[0]) - 65][ord(matchResults[1]) - 65] += 1
        
        if matchResults[0] == matchResults[1]:
            properMatches += 1
        
        counter += 1
        
        if counter % 500 == 0:
            print "\tEvaluated %d rows in %.2fmins" %(counter - 15000, (time.clock() - classificationTimeStart)/60)
    
    classificationTime = (time.clock() - classificationTimeStart)
    print "Classification time: %0.2fmin" %(classificationTime/60)

#The file will be closed now. 

print "Accuracy: %.2f%%" %((float(properMatches)/5000.0) * 100)

print "Confusion Matrix:"
print "%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c%-5c" % ('-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')
for row in range(26):
    sys.stdout.write("%-5c" %chr(row + 65)) 
    for col in range(26):
        sys.stdout.write("%-5d" %confusionMatrix[row][col])
    print "\n"
