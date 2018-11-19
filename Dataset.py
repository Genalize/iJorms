import csv
import random


def readDataset():  #tokenize the dataset and group them into respective class
    with open('C:\\Users\\Manish\\Desktop\\MajorProject\\Datasets\\dataset.csv') as csvfile:    #read CSV file
        dataset = list(csv.reader(csvfile, delimiter=','))
    preprocessDataset(dataset)
    data = {'education':[], 'workExperience':[], 'skill':[], 'certification':[]}
    for i in dataset:
        if (i[1] == 'education'):
            data['education'].append(i[0])
        elif (i[1] == 'workExperience'):
            data['workExperience'].append(i[0])
        elif (i[1] == 'skill'):
            data['skill'].append(i[0])
        elif (i[1] == 'certification'):
            data['certification'].append(i[0])

    return data


def preprocessDataset(content):
    for i in content:
        i[0] = i[0].replace(',', ' ')
        i[0] = i[0].replace(':', ' ')
        i[0] = i[0].replace('|', ' ')
        i[0] = i[0].replace('/', ' ')
        i[0] = i[0].replace(';', ' ')
        i[0] = i[0].replace('-', ' ')
        i[0] = i[0].replace('–', ' ')
        i[0] = i[0].replace('—', ' ')
        i[0] = i[0].replace('(', ' ')
        i[0] = i[0].replace(')', ' ')
        i[0] = i[0].lower()
        i[0] = i[0].split(' ')
        i[0] = list(filter(lambda x: x != '', i[0]))

    return content


def tenChunks(data, splitRatio):
    chunkSizeEducation = int(len(data['education']) * splitRatio)
    chunkSizeWorkExperience = int(len(data['workExperience']) * splitRatio)
    chunkSizeSkill = int(len(data['skill']) * splitRatio)
    chunkSizeCertification = int(len(data['certification']) * splitRatio)
    chunks = {0:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              1:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              2:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              3:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              4:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              5:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              6:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              7:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              8:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]},
              9:{'education':[], 'workExperience':[], 'skill':[], 'certification':[]}}

    remaining = data
    for t in range(9):
        while ((len(chunks[t]['workExperience'])) < chunkSizeWorkExperience):
            index = random.randrange(len(remaining['workExperience']))
            chunks[t]['workExperience'].append(remaining['workExperience'].pop(index))
        while ((len(chunks[t]['education'])) < chunkSizeEducation):
            index = random.randrange(len(remaining['education']))
            chunks[t]['education'].append(remaining['education'].pop(index))
        while ((len(chunks[t]['skill'])) < chunkSizeSkill):
            index = random.randrange(len(remaining['skill']))
            chunks[t]['skill'].append(remaining['skill'].pop(index))
        while ((len(chunks[t]['certification'])) < chunkSizeCertification):
            index = random.randrange(len(remaining['certification']))
            chunks[t]['certification'].append(remaining['certification'].pop(index))

    chunks[9] = remaining
    return chunks


def getAsList(trainingSet):
    education = []
    certification = []
    workExperience = []
    skill = []
    for i in trainingSet['education']:
        for z in i:
            education.append(z)
    for i in trainingSet['certification']:
        for z in i:
            certification.append(z)
    for i in trainingSet['workExperience']:
        for z in i:
            workExperience.append(z)
    for i in trainingSet['skill']:
        for z in i:
            skill.append(z)
    return education, certification, workExperience, skill