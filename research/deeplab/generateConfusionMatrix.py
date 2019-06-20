import os
import argparse
parser = argparse.ArgumentParser()
import numpy as np
from PIL import Image

predArmAndIsArm = 0
predArmAndIsNotArm = 0
predNotArmIsArm = 0
predNotArmIsNotArm = 0
totalPixels = 0

parser.add_argument("--model_name", help="Model name used")
parser.add_argument("--base_name", help="Base file name used (train, val or test)")

args = parser.parse_args()

def ConfusionMatrix(baseName, inputFolderPath, modelName, backColorSegMap=0, frontColorLabel=1):
    textFile = np.loadtxt('/home/conteinerFiles/skin-images/list_folder/' + baseName + ".txt", dtype=str)
    count = 0
    predArmAndIsArm = 0
    predArmAndIsNotArm = 0
    predNotArmIsArm = 0
    predNotArmIsNotArm = 0
    totalPixels = 0
    totalArm = 0
    totalNotArm = 0
    for idx in textFile:
        seg_map = Image.open(inputFolderPath + str(count).zfill(6) + "_prediction.png")
        seg_map = np.array(seg_map)
        seg_map = seg_map[:,:,:1]
        seg_map = np.squeeze(seg_map)
        resultImg = seg_map
        label = Image.open("/home/conteinerFiles/skin-images/masks/" + idx + ".pbm")
        labelArray = np.array(label)
        labelArray = np.resize(labelArray, seg_map.shape)
        labelArray = labelArray * 1
        maskImg = labelArray
        count = count + 1
        height, width = labelArray.shape
        print "Arquivo: " + idx
        for y in range(0,height): 
            for x in range(0,width): 
                #print resultImg[y,x]
                #backgroundColor = 14
                #backgroundColor = 30
                if maskImg[y,x] == frontColorLabel:
                    totalArm = totalArm + 1
                else:
                    totalNotArm = totalNotArm + 1
                #print "color: "
                #print resultImg[y,x]
                # when resultImage value is arm and maskImage value is arm
                if resultImg[y,x] != backColorSegMap and maskImg[y,x] == frontColorLabel:
                    predArmAndIsArm = predArmAndIsArm + 1
                
                # when resultImage value is arm and maskImage value is not arm
                if resultImg[y,x] != backColorSegMap and maskImg[y,x] != frontColorLabel:
                    predArmAndIsNotArm = predArmAndIsNotArm + 1

                # when resultImage value is not arm and maskImage value is arm
                if resultImg[y,x] == backColorSegMap and maskImg[y,x] == frontColorLabel:
                    predNotArmIsArm = predNotArmIsArm + 1

                # when resultImage value is not arm and maskImage value is not arm
                if resultImg[y,x] == backColorSegMap and maskImg[y,x] != frontColorLabel:
                    predNotArmIsNotArm = predNotArmIsNotArm + 1

                totalPixels = totalPixels + 1                       

#        fileOut = "/home/hmelo/testando2/test_" + filename
#        cv2.imwrite(fileOut, img)

    logFile = open('/home/conteinerFiles/deeplab/models/research/deeplab/datasets/hsarah/exp/' + modelName + '_confMatrix.log', 'a+')
    print "predArmAndIsArm = " + str(predArmAndIsArm)
    logFile.write("predArmAndIsArm = " + str(predArmAndIsArm) + '\n')
    print "predArmAndIsNotArm = " + str(predArmAndIsNotArm)
    logFile.write("predArmAndIsNotArm = " + str(predArmAndIsNotArm) + '\n')
    print "predNotArmIsArm = " + str(predNotArmIsArm)
    logFile.write("predNotArmIsArm = " + str(predNotArmIsArm) + '\n')
    print "predNotArmIsNotArm = " + str(predNotArmIsNotArm)
    logFile.write("predNotArmIsNotArm = " + str(predNotArmIsNotArm) + '\n')
    print "totalPixels = " + str(totalPixels)
    logFile.write("totalPixels = " + str(totalPixels) + '\n')
    print "totalArm = " + str(totalArm)
    logFile.write("totalArm = " + str(totalArm) + '\n')
    print "totalNotArm = " + str(totalNotArm)
    logFile.write("totalNotArm = " + str(totalNotArm) + '\n')

        
if args.model_name:
    modelName = args.model_name
else:
    modelName = 'train_psp_100000_iterations_deeplabAug_correct'

if args.base_name:
    baseName = args.base_name
else:
    baseName = "test"

inputFolderPath = "/home/conteinerFiles/deeplab/models/research/deeplab/datasets/hsarah/exp/" + modelName + "/vis/segmentation_results/"
ConfusionMatrix(baseName, inputFolderPath, modelName)
#getConfusionMatrixByTextFile(maskPath, resultPath, generalPath, textFile, testFile)


