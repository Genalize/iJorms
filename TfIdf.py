import math
from Dataset import getAsList
from PerformanceMeasure import performanceMeasure
import operator


def tfidf(trainingSet, testSet):
    education, certification, workExperience, skill = getAsList(trainingSet)
    tfEducation, tfWorkExperience, tfSkill, tfCertification = calculateTfWeight(education, workExperience, skill, certification)
    Idf = calculateIdfWeight(education, workExperience, skill, certification)

    predictions, weights = getPredictionsTfIdf(testSet, tfEducation, tfWorkExperience, tfSkill, tfCertification, Idf)
    # for i in range(len(predictions)):
    #      print(tokens[i], predictions[i], weights[i])

    return predictions, weights


def calculateTfWeight(education, workExperience, skill, certification): #calculation of the TF weights
    tfEducation = {}
    tfWorkExperience = {}
    tfSkill = {}
    tfCertification = {}

    for i in education:
        if(i not in tfEducation.keys()):
            tfEducation[i] = 1 + math.log10(sum(1 for p in education if p == i))   # Log-frequency weighting, can use other (accuracy needs to be checked)
    # for i in tfEducation:
    #     tfEducation[i] = 0.5 + (0.5 * tfEducation[i])/(max(tfEducation.values()))
    for i in workExperience:
        if(i not in tfWorkExperience.keys()):
            tfWorkExperience[i] = 1 + math.log10(sum(1 for p in workExperience if p == i))
    # for i in tfWorkExperience:
    #     tfWorkExperience[i] = 0.5 + (0.5 * tfWorkExperience[i])/(max(tfWorkExperience.values()))
    for i in skill:
        if(i not in tfSkill.keys()):
            tfSkill[i] = 1 + math.log10(sum(1 for p in skill if p == i))
    # for i in skill:
    #     tfSkill[i] = 0.5 + (0.5 * tfSkill[i])/(max(tfSkill.values()))
    for i in certification:
        if(i not in tfCertification.keys()):
            tfCertification[i] = 1 + math.log10(sum(1 for p in certification if p == i))
    # for i in tfCertification:
    #     tfCertification[i] = 0.5 + (0.5 * tfCertification[i])/(max(tfCertification.values()))

    # print(tfCertification)
    # print(sorted(tfEducation.items(), key = operator.itemgetter(1), reverse = True))
    # print(sorted(tfWorkExperience.items(), key = operator.itemgetter(1), reverse = True))
    # print(sorted(tfSkill.items(), key = operator.itemgetter(1), reverse = True))
    # print(sorted(tfCertification.items(), key = operator.itemgetter(1), reverse = True))

    return tfEducation, tfWorkExperience, tfSkill, tfCertification


def calculateIdfWeight(education, workExperience, skill, certification):    #calculation of IDF weights
    temp = education + workExperience + skill + certification
    Idf = {}
    for i in temp:
        if(i not in Idf.keys()):
            Idf[i] = 0
            if(i in education):
                Idf[i] = Idf[i] + 1
            if(i in workExperience):
                Idf[i] = Idf[i] + 1
            if(i in skill):
                Idf[i] = Idf[i] + 1
            if(i in certification):
                Idf[i] = Idf[i] + 1
            Idf[i] = math.log10(4 / Idf[i])
    # print(sorted(Idf.items(), key = operator.itemgetter(1), reverse = False))

    return Idf


def getPredictionsTfIdf(testSet, tfEducation, tfWorkExperience, tfSkill, tfCertification, Idf):     #get predictions for all the sentences
    predictions = {}
    weights = {}
    predictionsCertification = []
    weightCertification = []
    predictionsEducation = []
    weightEducation = []
    predictionsSkill = []
    weightSkill = []
    predictionsWorkExperience = []
    weightWorkExperience = []

    for i in testSet['certification']:
        result, weig = predictTfIdf(i, tfEducation, tfWorkExperience, tfSkill, tfCertification, Idf)
        predictionsCertification.append(result)
        weightCertification.append(weig)
    predictions['certification'] = predictionsCertification
    weights['certification'] = weightCertification

    for i in testSet['education']:
        result, weig = predictTfIdf(i, tfEducation, tfWorkExperience, tfSkill, tfCertification, Idf)
        predictionsEducation.append(result)
        weightEducation.append(weig)
    predictions['education'] = predictionsEducation
    weights['education'] = weightEducation

    for i in testSet['skill']:
        result, weig = predictTfIdf(i, tfEducation, tfWorkExperience, tfSkill, tfCertification, Idf)
        predictionsSkill.append(result)
        weightSkill.append(weig)
    predictions['skill'] = predictionsSkill
    weights['skill'] = weightSkill

    for i in testSet['workExperience']:
        result, weig = predictTfIdf(i, tfEducation, tfWorkExperience, tfSkill, tfCertification, Idf)
        predictionsWorkExperience.append(result)
        weightWorkExperience.append(weig)
    predictions['workExperience'] = predictionsWorkExperience
    weights['workExperience'] = weightWorkExperience

    return predictions, weights


def predictTfIdf(i, tfEducation, tfWorkExperience, tfSkill, tfCertification, Idf):      #prediction for one sentence
    weight = {}
    weight['education'] = 0
    weight['workExperience'] = 0
    weight['skill'] = 0
    weight['certification'] = 0

    for word in i:
        if(word in tfEducation.keys()):
            weight['education'] = weight['education'] + tfEducation[word] * Idf[word]
        if(word in tfWorkExperience.keys()):
            weight['workExperience'] = weight['workExperience'] + tfWorkExperience[word] * Idf[word]
        if(word in tfSkill.keys()):
            weight['skill'] = weight['skill'] + tfSkill[word] * Idf[word]
        if(word in tfCertification.keys()):
            weight['certification'] = weight['certification'] + tfCertification[word] * Idf[word]
    if (len(set(list(weight.values()))) == 1):
        return 'Other', 0

    bestLabel, bestWeight = None, -1
    for classValue, weig in list(weight.items()):
        if (bestLabel is None or weight[classValue] > bestWeight):
            bestWeight = weig
            bestLabel = classValue

    return bestLabel, bestWeight


def calculateTfIdfPerformanceMeasure(TfIdfPrediction):
    truePositive = {}
    falseNegative = {}
    falsePositive = {}
    trueNegative = {}

    truePositive['education'] = TfIdfPrediction['education'].count('education')
    falseNegative['education'] = len(TfIdfPrediction['education']) - truePositive['education']
    falsePositive['education'] = TfIdfPrediction['certification'].count('education') + TfIdfPrediction['skill'].count('education') + TfIdfPrediction['workExperience'].count('education')
    trueNegative['education'] = (len(TfIdfPrediction['certification']) + len(TfIdfPrediction['skill']) + len(TfIdfPrediction['workExperience'])) - falsePositive['education']

    truePositive['certification'] = TfIdfPrediction['certification'].count('certification')
    falseNegative['certification'] = len(TfIdfPrediction['certification']) - truePositive['certification']
    falsePositive['certification'] = TfIdfPrediction['education'].count('certification') + TfIdfPrediction['skill'].count('certification') + TfIdfPrediction['workExperience'].count('certification')
    trueNegative['certification'] = (len(TfIdfPrediction['education']) + len(TfIdfPrediction['skill']) + len(TfIdfPrediction['workExperience'])) - falsePositive['certification']

    truePositive['skill'] = TfIdfPrediction['skill'].count('skill')
    falseNegative['skill'] = len(TfIdfPrediction['skill']) - truePositive['skill']
    falsePositive['skill'] = TfIdfPrediction['education'].count('skill') + TfIdfPrediction['certification'].count('skill') + TfIdfPrediction['workExperience'].count('skill')
    trueNegative['skill'] = (len(TfIdfPrediction['education']) + len(TfIdfPrediction['certification']) + len(TfIdfPrediction['workExperience'])) - falsePositive['skill']

    truePositive['workExperience'] = TfIdfPrediction['workExperience'].count('workExperience')
    falseNegative['workExperience'] = len(TfIdfPrediction['workExperience']) - truePositive['workExperience']
    falsePositive['workExperience'] = TfIdfPrediction['education'].count('workExperience') + TfIdfPrediction['certification'].count('workExperience') + TfIdfPrediction['skill'].count('workExperience')
    trueNegative['workExperience'] = (len(TfIdfPrediction['education']) + len(TfIdfPrediction['certification']) + len(TfIdfPrediction['skill'])) - falsePositive['workExperience']

    return performanceMeasure(truePositive, trueNegative, falsePositive, falseNegative)