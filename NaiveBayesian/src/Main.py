'''
Created on Sep 27, 2013

@author: Ankur
'''
from SMSData import SMSData, AttributeCountsAndProbabilites, SPAM, HAM
from random import shuffle
import heapq
import math
import random

properClassifications = 0
improperClassifications = 0

smsData = []
attributeData = []
vocabularyCount = 0
totalSMSRead = 0;

def contains(iterable, wordFilter):
    for x in iterable:
        if wordFilter(x):
            return x
    return None

NMostFrequentWordsToRemove = [10, 25, 50, 100, 500]

def createFiveChunks(seq):
    avg = len(seq) / 5.0
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

# read from the file and construct the random SMSdata training set.
def init():
    print "Reading training data..."
    x = 0
    with open('SMSSpamCollection', 'r') as f:
        for line in f:
            smsData.append(SMSData(line))
            x += 1      
    print "Finished reading traning data... Randomizing before constructing learners.."
    # now we've read in all the Training data. Choose to traing with only 2/3rds of it.
    indexArray = []
    # initialize an array of count of read in elements, shuffle them randomly and then use them to construct the training data.
    for i in range(x):
        indexArray.append(i);
        
    # shuffle the smsData to ensure random chunks are generated.
    r = random.seed(3243234)
    shuffle(smsData, r)
    
def trainNaiveBayes(chunks, validationChunkIndex):
    
    # use the global variables
    global attributeData
    attributeData = []
    global vocabularyCount
    vocabularyCount = 0
    
    # local variable set
    totalTrainingDataCount = 0
    classCounts = [0, 0]
    
    
    # Now we have created 5 chunks of randomly shuffled SMSData. Do a 5 fold cross validation.
    for i in range(5):
        
        # do not use the data in the validation chunk to train the learner.
        if i == validationChunkIndex:
            continue
        
        # iterate over all the sms' in each chunk.
        smsList = chunks.__getitem__(i)
#         print 'Evaluating chunk %d' % (i+1)
        counter = 0
        for sms in smsList:
            counter += 1
            for word in sms.attributes:
                entry = contains(attributeData, lambda x: x.word == word)
                
                if entry is not None:
                    entry.addToClass(sms.hypClass)
                    if sms.hypClass == SPAM:
                        classCounts.insert(SPAM, classCounts.__getitem__(SPAM) + 1)
                    else:
                        classCounts.insert(HAM, classCounts.__getitem__(HAM) + 1)
                else:
                    entry = AttributeCountsAndProbabilites(word)
                    entry.addToClass(sms.hypClass)
                    attributeData.append(entry)
                    
                vocabularyCount += 1
        totalTrainingDataCount += counter
        
    # print "Total words read: %d\n" % vocabularyCount
    validationSMSSet = chunks.__getitem__(validationChunkIndex)
    
    for i in range(NMostFrequentWordsToRemove.__len__()):
        print "\tRemoving top %d occuring words" %NMostFrequentWordsToRemove[i]
        topNWords = heapq.nlargest(NMostFrequentWordsToRemove[i], attributeData, key = lambda x: x.totalOccurances)
        
        for topOccuringWord in topNWords:
            elementToDelete = contains(attributeData, lambda x: x.word == topOccuringWord.word)
            if elementToDelete is not None:
                attributeData.remove(elementToDelete)
    
    #   test with 'n' top occuring words deleted..
        testNaiveBayes(validationSMSSet, attributeData, vocabularyCount, totalTrainingDataCount, classCounts)
    
# the testNaiveBayes function
def testNaiveBayes(validationSet, parameterTable, vocabularyCount, totalTrainingDataCount, classCounts=[]):
    
    properClassifications = 0.0
    improperClassifications = 0.0
    
    truePositives = 0.0
    falsePositives = 0.0
    trueNegatives = 0.0
    falseNegatives = 0.0
    
    for sms in validationSet:
        #=======================================================================
        # The probability that this SMS is HAM/SPAM is evaluated using the Naive
        # Bayesian estimate:
        #
        # P(word | SPAM/HAM) = (nk + 1)/(n + |V|)
        #    nk  -> No. of SMS' with word as SPAM/HAM
        #    n   -> Total no. of training data.
        #    |V| -> Vocabulary Size
        #=======================================================================
        
        # for each class, get the probabilities of each word.
        
        # keep a variable for keeping track of the probabilities for both classes. 
        classProbabilities = []        
        
        for hypClass in [SPAM, HAM]:
            currentClassProbability = 0.0 #initialize this to 1.0 if we're doing product of probabilities.
            for word in sms.attributes:
                
                # first check if this is a word present in our vocabulary
                entry = contains(parameterTable, lambda x: x.word == word)
                
                nk = entry.getCountForClass(hypClass) if entry is not None else 0
                
#                 currentClassProbability *= ((nk + 1.0) / (totalTrainingDataCount + vocabularyCount))
                currentClassProbability += math.log((nk + 1.0) / (totalTrainingDataCount + vocabularyCount), 2)
                
#            The below function is to be used to use "Product of probabilities". 
#           classProbabilities.insert(hypClass, currentClassProbability * (classCounts.__getitem__(hypClass)/totalTrainingDataCount))

#         The below function is to be used for "SUM OF LOGS". A good idea is to scale the probabilities by logs. 
            classProbabilities.insert(hypClass, currentClassProbability + math.log((classCounts.__getitem__(hypClass) / totalTrainingDataCount), 2))
        
        # Now we have the probabilities for both the classes. Classify the new example.
        classifiedClass = SPAM if classProbabilities.__getitem__(SPAM) > classProbabilities.__getitem__(HAM) else HAM
        
        #Check if the classified value is a TP or FP. Consider HAM to be true and SPAM to be false.
        if classifiedClass == HAM and sms.hypClass == HAM:
            truePositives += 1
        elif classifiedClass == HAM and sms.hypClass == SPAM:
            falsePositives += 1
        elif classifiedClass == SPAM and sms.hypClass == HAM:
            falseNegatives += 1
        elif classifiedClass == SPAM and sms.hypClass == SPAM:
            trueNegatives += 1
            
        #Add to the proper/improper counts
        if classifiedClass == sms.hypClass:
            properClassifications += 1 
        else:
            improperClassifications += 1
    
    print "\t==================================================================================================="
    print "\tTotal Proper Classifications: %d Total Improper Classifications: %d" %(properClassifications, improperClassifications)
    print "\tTotal True Positives Rate: %.2f%% Total False Positives Rate: %.2f%% \n" %(100 * (truePositives/totalTrainingDataCount) , 100 * (falsePositives/totalTrainingDataCount))
    print "\tAccuracy: %.2f%% Error: %.2f%% " %(100 * (truePositives + trueNegatives)/(truePositives + trueNegatives + falseNegatives + falsePositives), 100 * (improperClassifications/totalTrainingDataCount))
    print "\tSpecifity: %.2f%% Sensitivity: %.2f%%" %(100 * trueNegatives /(trueNegatives + falsePositives), 100 * truePositives / (falseNegatives + truePositives))
    print "\tRecall: %.2f%%, Precision: %.2f%%" %(truePositives * 100 /(falseNegatives + truePositives), 100 * truePositives / (falsePositives + truePositives))
    print "\t==================================================================================================="
            
#===============================================================================
# Naive Bayes Classification implementation.
# Outline: 
# - Read the input data. 
# - Randomize the data.
# - Create 5 chunks.
# - Do a 5 fold validation.
#===============================================================================
            
init()

print "Creating chunks for 5-stage cross validation"
chunks = createFiveChunks(smsData)

for i in range(5):
    print "Iteration %d of 5 stage cross validation" % (i + 1)
    trainNaiveBayes(chunks, i)
