import os
os.environ['CLASSPATH'] = "C:/Users/Manish/Desktop/MajorProject/tika-app-1.14.jar"
from jnius import autoclass # Import the Java classes we are going to need
# from nltk.tag import StanfordNERTagger
# import nltk
import copy
from Dataset import *
from NaiveBayes import sentenceClassifierNB, calculateNBPerformanceMeasure
from TfIdf import tfidf, calculateTfIdfPerformanceMeasure
from bagging import sentenceClassifierBag, calculateBagPerformanceMeasure


# def getText(filename):       # get plain text using Apache Tika
#     Tika = autoclass('org.apache.tika.Tika')
#     Metadata = autoclass('org.apache.tika.metadata.Metadata')
#     FileInputStream = autoclass('java.io.FileInputStream')
#
#     tika = Tika()
#     meta = Metadata()
#     text = tika.parseToString(FileInputStream(filename), meta)
#     return text


# def preprocess(content):   #preprocess the test resume to get tokens
#     c = content.lower()
#     c = c.replace('.\n','\n')
#     c = c.replace(',', ' ')
#     c = c.replace(':', ' ')
#     c = c.replace('|', ' ')
#     c = c.replace('/', ' ')
#     c = c.replace(';', ' ')
#     c = c.replace('-', ' ')
#     c = c.replace('–', ' ') #big dash
#     c = c.split('\n')
#     c = list(filter(lambda x: x != '', c))
#     tokens = []
#     for i in c:
#         token = i.split(' ')
#         token = list(filter(lambda x: x != '', token))
#         tokens.append(token)
#     return tokens            #improve the preprocessing: remove () maybe...


def main():
    # filename = "C:\\Users\\Manish\\Desktop\\MajorProject\\resume1.doc"  # test resume
    # content = getText(filename)
    # tokens = preprocess(content)
    data = readDataset()
    splitRatio = 0.1
    splittedDataset = tenChunks(copy.deepcopy(data), splitRatio)

    totalFmeasureNBCertification = 0
    totalFmeasureNBEducation = 0
    totalFmeasureNBSkill = 0
    totalFmeasureNBWorkExperience = 0

    totalFmeasureTfIdfCertification = 0
    totalFmeasureTfIdfEducation = 0
    totalFmeasureTfIdfSkill = 0
    totalFmeasureTfIdfWorkExperience = 0

    # totalFmeasureBagCertification = 0
    # totalFmeasureBagEducation = 0
    # totalFmeasureBagSkill = 0
    # totalFmeasureBagWorkExperience = 0

    for r in range(10):
        trainingSet = {'education':[], 'workExperience':[], 'skill':[], 'certification':[]}
        testSet = splittedDataset[r]
        for j in list(set(range(0,9))-{r}):
            trainingSet['education'].extend(splittedDataset[j]['education'])
            trainingSet['workExperience'].extend(splittedDataset[j]['workExperience'])
            trainingSet['skill'].extend(splittedDataset[j]['skill'])
            trainingSet['certification'].extend(splittedDataset[j]['certification'])

        NBPrediction, NBProbability = sentenceClassifierNB(copy.deepcopy(trainingSet), copy.deepcopy(testSet))
        fmeasureNBCertification, fmeasureNBEducation, fmeasureNBSkill, fmeasureNBWorkExperience = calculateNBPerformanceMeasure(NBPrediction)
        totalFmeasureNBCertification += fmeasureNBCertification
        totalFmeasureNBEducation += fmeasureNBEducation
        totalFmeasureNBSkill += fmeasureNBSkill
        totalFmeasureNBWorkExperience += fmeasureNBWorkExperience

        TfIdfPrediction, TfIdfWeight = tfidf(copy.deepcopy(trainingSet), copy.deepcopy(testSet))
        fmeasureTfIdfCertification, fmeasureTfIdfEducation, fmeasureTfIdfSkill, fmeasureTfIdfWorkExperience = calculateTfIdfPerformanceMeasure(TfIdfPrediction)
        totalFmeasureTfIdfCertification += fmeasureTfIdfCertification
        totalFmeasureTfIdfEducation += fmeasureTfIdfEducation
        totalFmeasureTfIdfSkill += fmeasureTfIdfSkill
        totalFmeasureTfIdfWorkExperience += fmeasureTfIdfWorkExperience

        # BagPrediction, BagProbability = sentenceClassifierBag(copy.deepcopy(trainingSet), copy.deepcopy(testSet))
        # fmeasureBagCertification, fmeasureBagEducation, fmeasureBagSkill, fmeasureBagWorkExperience = calculateBagPerformanceMeasure(BagPrediction)
        # totalFmeasureBagCertification += fmeasureBagCertification
        # totalFmeasureBagEducation += fmeasureBagEducation
        # totalFmeasureBagSkill += fmeasureBagSkill
        # totalFmeasureBagWorkExperience += fmeasureBagWorkExperience


    finalFmeasureNBCertification = totalFmeasureNBCertification * 0.1
    finalFmeasureNBEducation = totalFmeasureNBEducation * 0.1
    finalFmeasureNBSkill = totalFmeasureNBSkill * 0.1
    finalFmeasureNBWorkExperience = totalFmeasureNBWorkExperience * 0.1

    finalFmeasureTfIdfCertification = totalFmeasureTfIdfCertification * 0.1
    finalFmeasureTfIdfEducation = totalFmeasureTfIdfEducation * 0.1
    finalFmeasureTfIdfSkill = totalFmeasureTfIdfSkill * 0.1
    finalFmeasureTfIdfWorkExperience = totalFmeasureTfIdfWorkExperience * 0.1

    # finalFmeasureBagCertification = totalFmeasureBagCertification * 0.1
    # finalFmeasureBagEducation = totalFmeasureBagEducation * 0.1
    # finalFmeasureBagSkill = totalFmeasureBagSkill * 0.1
    # finalFmeasureBagWorkExperience = totalFmeasureBagWorkExperience * 0.1

    print('')


main()