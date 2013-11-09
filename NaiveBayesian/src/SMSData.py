'''
Created on Sep 27, 2013

@author: Ankur
'''
import string

SPAM = 0
HAM = 1

abbrReplacements = {
                " y ": " why ",
                " d ": " the ",
                " n ": " and ",
                " u ": " you ",
                " c ": " see ",
                " tho ": " though ",
                " k ": " ok ", 
                " okay ": " ok ",
                " pls ": " please ",
                " ur ": " your ",
                " nd ": " and ",
                " r ": " are "
                }

class AttributeCountsAndProbabilites:
        
    def __init__(self, word):
        self.word = word
        self.spam = 0
        self.ham = 0
        self.pSpam = 0.0
        self.pHam = 0.0
        self.totalOccurances = 0;
        
    def addToClass(self, smsClass):
        if smsClass == SPAM :
            self.spam += 1
            self.totalOccurances += 1;
#             self.calculateSpamProbability()
        elif smsClass == HAM:
            self.ham += 1
            self.totalOccurances += 1;
#             self.calculateHamProbability()
        
    def calculateHamProbability(self):
        self.pHam = self.getProbabilityForClass(HAM)
        
    def calculateSpamProbability(self):
        self.pSpam = self.getProbabilityForClass(SPAM)
    
    def getCountForClass(self, hypClass):
        return self.spam if hypClass == SPAM else self.ham
    
    def getProbabilityForClass(self, x):
        if self.totalOccurances == 0:
            return -1;
        
        if x == SPAM:
            return self.spam * 1.0 / self.totalOccurances
        elif x == HAM:
            return self.ham * 1.0/ self.totalOccurances
        else:
            return -1
    
    def __repr__(self):
        return "Word: %s, p(Spam) = %.2f, p(Ham) = %.2f Total Count: %d\n" %(self.word, self.getProbabilityForClass(SPAM), self.getProbabilityForClass(HAM), self.totalOccurances)
    
class SMSData:
    
    def __init__(self, smsLine):
        smsLineBreakup = smsLine.split("\t");
        
        if smsLineBreakup[0] == "spam":
            self.hypClass = SPAM
        else: 
            self.hypClass = HAM
        
        #cleanup some stuff from sms body
        smsLineBreakup[1] = smsLineBreakup[1].lower()                       #make all lower case
        smsLineBreakup[1] = smsLineBreakup[1].strip()                       #remove all new lines and trailing spaces
        smsLineBreakup[1] = smsLineBreakup[1].translate(string.maketrans("", ""), string.punctuation)  #replace all punctuation marks.
        
        #replace abbrevations
        for i, j in abbrReplacements.iteritems():
            smsLineBreakup[1] = smsLineBreakup[1].replace(i, j)
        
        self.attributes = smsLineBreakup[1].split(" ")

    def __repr__(self):
        return "Class: %s, Attributes: %s\n" %('SPAM' if self.hypClass == SPAM else 'HAM', self.attributes)
