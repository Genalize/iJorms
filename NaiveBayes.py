from Dataset import getAsList
from PerformanceMeasure import performanceMeasure


def sentenceClassifierNB(trainingSet, testSet):   #Naive Bayes classifier
    hashtable, lengths = naiveBayesTrain(trainingSet)
    predictions, prob = getPredictions(hashtable, lengths, testSet)
    return predictions, prob
    # print(sorted(hashtable.items(), key = operator.itemgetter(1), reverse = True))


def naiveBayesTrain(trainingSet):
    hashtable, lengths = generateHash(trainingSet)
    return hashtable, lengths


def generateHash(trainingSet):     #generate hash table of probabilities
    education, certification, workExperience, skill =  getAsList(trainingSet)
    vocabEducation = list(set(education))
    vocabCertification = list(set(certification))
    vocabWorkExperience = list(set(workExperience))
    vocabSkill = list(set(skill))
    total = len(vocabCertification) + len(vocabSkill) + len(vocabWorkExperience) + len(vocabEducation)  #cardinality of vocabulary set

    hashTable = {}    #hashtable of probabilities generated below with add-1 laplace smoothing
    for i in vocabCertification:
        hashTable[tuple([i,'certification'])] = (sum(1 for p in certification if p == i) + 1) / (len(certification)+(len(vocabSkill)+len(vocabWorkExperience)+len(vocabEducation)+len(vocabCertification)))
    for i in vocabEducation:
        hashTable[tuple([i,'education'])] = (sum(1 for p in education if p == i)+1) / (len(education)+(len(vocabSkill)+len(vocabWorkExperience)+len(vocabEducation)+len(vocabCertification)))
    for i in vocabSkill:
        hashTable[tuple([i,'skill'])] = (sum(1 for p in skill if p == i)+1) / (len(skill)+(len(vocabSkill)+len(vocabWorkExperience)+len(vocabEducation)+len(vocabCertification)))
    for i in vocabWorkExperience:
        hashTable[tuple([i,'workExperience'])] = (sum(1 for p in workExperience if p == i)+1) / (len(workExperience)+(len(vocabSkill)+len(vocabWorkExperience)+len(vocabEducation)+len(vocabCertification)))

    # hashTable['certification'] = 0.25    #considering equiprobable class
    # hashTable['education'] = 0.25
    # hashTable['skill'] = 0.25
    # hashTable['workExperience'] = 0.25
    lengths = {'certification': len(certification), 'education': len(education), 'skill': len(skill), 'workExperience': len(workExperience), 'total': total}  #number of words in each class

    lenDataset = len(certification)+len(education)+len(skill)+len(workExperience) #Yields lesser accuracy than equiprobable assumption
    probCertification = len(certification) / lenDataset
    probEducation = len(education) / lenDataset
    probSkill = len(skill) / lenDataset
    probWorkExperience = len(workExperience) / lenDataset
    hashTable['certification'] = probCertification
    hashTable['education'] = probEducation
    hashTable['skill'] = probSkill
    hashTable['workExperience'] = probWorkExperience

    return hashTable, lengths


def getPredictions(hashtable, lengths, testSet):     #get predictions for all the sentences
    predictions = {}
    prob = {}
    predictionsCertification = []
    probCertification = []
    predictionsEducation = []
    probEducation = []
    predictionsSkill = []
    probSkill = []
    predictionsWorkExperience = []
    probWorkExperience = []

    for i in testSet['certification']:
        if(len(i) != 0):
            result, probab = predict(hashtable, lengths, i)
            predictionsCertification.append(result)
            probCertification.append(probab)
    predictions['certification'] = predictionsCertification
    prob['certification'] = probCertification

    for i in testSet['education']:
        if(len(i) != 0):
            result, probab = predict(hashtable, lengths, i)
            predictionsEducation.append(result)
            probEducation.append(probab)
    predictions['education'] = predictionsEducation
    prob['education'] = probEducation

    for i in testSet['skill']:
        if(len(i) != 0):
            result, probab = predict(hashtable, lengths, i)
            predictionsSkill.append(result)
            probSkill.append(probab)
            if result == 'workExperience':
                print(i)
    predictions['skill'] = predictionsSkill
    prob['skill'] = probSkill


    for i in testSet['workExperience']:
        if(len(i) != 0):
            result, probab = predict(hashtable, lengths, i)
            predictionsWorkExperience.append(result)
            probWorkExperience.append(probab)
    predictions['workExperience'] = predictionsWorkExperience
    prob['workExperience'] = probWorkExperience

    return predictions, prob


def predict(hashtable, lengths, inVector):      #prediction for one sentence
    probabilities = calculateClassProbabilities(hashtable, lengths, inVector)
    # if (len(set(list(probabilities.values()))) == 1):
    #     return 'Other', 0
    if(probabilities['certification']/hashtable['certification'] == probabilities['education']/hashtable['education'] and probabilities['certification']/hashtable['certification'] == probabilities['skill']/hashtable['skill'] and probabilities['certification']/hashtable['certification'] == probabilities['workExperience']/hashtable['workExperience']):
        return 'Other', 0

    bestLabel, bestProb = None, -1
    for classValue, probability in list(probabilities.items()):
        if (bestLabel is None or probability > bestProb):
            bestProb = probability
            bestLabel = classValue


    return bestLabel, bestProb


def calculateClassProbabilities(hashtable, lengths, inVector):  #calculation of probabilities that the sentence belongs to each of the four classes
    probabilities = {}
    for cls in ['certification', 'education', 'skill', 'workExperience']:
        probabilities[cls] = hashtable[cls]
        for i in inVector:
            if((i, cls) in hashtable.keys()):
                probabilities[cls] *= hashtable[(i, cls)]
            else:   #handling unknown words, i.e., those that are not seen in training set
                # probabilities[cls] *= (1 / (lengths[cls] + lengths['total'] + 1))
                probabilities[cls] *= (1 / (lengths['total'] + 1))

    return probabilities


def calculateNBPerformanceMeasure(NBPrediction):
    truePositive = {}
    falseNegative = {}
    falsePositive = {}
    trueNegative = {}

    truePositive['education'] = NBPrediction['education'].count('education')
    falseNegative['education'] = len(NBPrediction['education']) - truePositive['education']
    falsePositive['education'] = NBPrediction['certification'].count('education') + NBPrediction['skill'].count('education') + NBPrediction['workExperience'].count('education')
    trueNegative['education'] = (len(NBPrediction['certification']) + len(NBPrediction['skill']) + len(NBPrediction['workExperience'])) - falsePositive['education']

    truePositive['certification'] = NBPrediction['certification'].count('certification')
    falseNegative['certification'] = len(NBPrediction['certification']) - truePositive['certification']
    falsePositive['certification'] = NBPrediction['education'].count('certification') + NBPrediction['skill'].count('certification') + NBPrediction['workExperience'].count('certification')
    trueNegative['certification'] = (len(NBPrediction['education']) + len(NBPrediction['skill']) + len(NBPrediction['workExperience'])) - falsePositive['certification']

    truePositive['skill'] = NBPrediction['skill'].count('skill')
    falseNegative['skill'] = len(NBPrediction['skill']) - truePositive['skill']
    falsePositive['skill'] = NBPrediction['education'].count('skill') + NBPrediction['certification'].count('skill') + NBPrediction['workExperience'].count('skill')
    trueNegative['skill'] = (len(NBPrediction['education']) + len(NBPrediction['certification']) + len(NBPrediction['workExperience'])) - falsePositive['skill']

    truePositive['workExperience'] = NBPrediction['workExperience'].count('workExperience')
    falseNegative['workExperience'] = len(NBPrediction['workExperience']) - truePositive['workExperience']
    falsePositive['workExperience'] = NBPrediction['education'].count('workExperience') + NBPrediction['certification'].count('workExperience') + NBPrediction['skill'].count('workExperience')
    trueNegative['workExperience'] = (len(NBPrediction['education']) + len(NBPrediction['certification']) + len(NBPrediction['skill'])) - falsePositive['workExperience']

    return performanceMeasure(truePositive, trueNegative, falsePositive, falseNegative)