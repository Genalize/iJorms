
def performanceMeasure(truePositive, trueNegative, falsePositive, falseNegative):
    accuracyEducation = (truePositive['education'] + trueNegative['education']) / (truePositive['education'] + trueNegative['education'] + falsePositive['education'] + falseNegative['education'])
    accuracyCertification = (truePositive['certification'] + trueNegative['certification']) / (truePositive['certification'] + trueNegative['certification'] + falsePositive['certification'] + falseNegative['certification'])
    accuracySkill = (truePositive['skill'] + trueNegative['skill']) / (truePositive['skill'] + trueNegative['skill'] + falsePositive['skill'] + falseNegative['skill'])
    accuracyWorkExperience = (truePositive['workExperience'] + trueNegative['workExperience']) / (truePositive['workExperience'] + trueNegative['workExperience'] + falsePositive['workExperience'] + falseNegative['workExperience'])

    precisionEducation = (truePositive['education']) / (truePositive['education'] + falsePositive['education'])
    precisionCertification = (truePositive['certification']) / (truePositive['certification'] + falsePositive['certification'])
    precisionSkill = (truePositive['skill']) / (truePositive['skill'] + falsePositive['skill'])
    precisionWorkExperience = (truePositive['workExperience']) / (truePositive['workExperience'] + falsePositive['workExperience'])

    recallEducation = (truePositive['education']) / (truePositive['education'] + falseNegative['education'])
    recallCertification = (truePositive['certification']) / (truePositive['certification'] + falseNegative['certification'])
    recallSkill = (truePositive['skill']) / (truePositive['skill'] + falseNegative['skill'])
    recallWorkExperience = (truePositive['workExperience']) / (truePositive['workExperience'] + falseNegative['workExperience'])

    fmeasureEducation = (2 * precisionEducation * recallEducation) / (precisionEducation + recallEducation)
    fmeasureCertification = (2 * precisionCertification * recallCertification) / (precisionCertification + recallCertification)
    fmeasureSkill = (2 * precisionSkill * recallSkill) / (precisionSkill + recallSkill)
    fmeasureWorkExperience = (2 * precisionWorkExperience * recallWorkExperience) / (precisionWorkExperience + recallWorkExperience)

    return fmeasureCertification, fmeasureEducation, fmeasureSkill, fmeasureWorkExperience