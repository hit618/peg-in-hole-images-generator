import argparse
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from DualArmAsm.imageTrainer.makeDataset.synth_utils import *
import cv2
import json

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


from zmqRemoteApi import RemoteAPIClient

parser = argparse.ArgumentParser()
# parser.add_argument("--frame-start", type=int, default=0)
# parser.add_argument("--frame-end", type=int, default=1000)
parser.add_argument("--image-num", type=list, default=[100,100,100])
parser.add_argument("--image-path", type=str, default='../dataset/vrepImage/hole12_2/image')
parser.add_argument("--annotate-path", type=str, default='../dataset/vrepImage/hole12_2/annotate')
parser.add_argument("--val", type=bool, default=True)
# parser.add_argument("--val-image-path", type=str, default='../dataset/classify/validation4/image')
# parser.add_argument("--val-annotate-path", type=str, default='../dataset/classify/validation4/annotate')
parser.add_argument("--val-image-path", type=str, default='../dataset/vrepImageVal/hole2L/image')
parser.add_argument("--val-annotate-path", type=str, default='../dataset/vrepImageVal/hole2L/annotate')
args = parser.parse_args()

client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True)
sim.startSimulation()
client.step()
deg = math.pi / 180

def restHole(holeHandle, holeHandle2=None, rgb = None):
    rgb_data = resetColor(holeHandle)
    if holeHandle2 is not None:
        rgb_data = resetColor(holeHandle2,rgb)
    return rgb_data
def resetPeg(objectHandle, xRange, yRange, zRange, rotRange, rgb = None):

    position = list(pegPosSampler2(xRange, yRange, zRange))
    # print(position)
    sim.setObjectPosition(objectHandle, sim.handle_world, position)
    quat = list(pegRotSampler(rotRange, express='quat'))
    # print(quat)
    sim.setObjectQuaternion(objectHandle, sim.handle_world, quat)
    # resetColor(objectHandle)
    rgb_data = resetColor(objectHandle, rgb)
    return position, quat, rgb_data

def resetCamera(pegTipHandle, objectHandle, dRange, eleRange, rotRange, positionRange):
    d = np.random.uniform(dRange[0], dRange[1])
    # elevation = np.random.uniform((180-55)*deg, (180-35)*deg)
    elevation = np.random.uniform((90 + eleRange[0]) * deg, (90 + eleRange[1]) * deg)
    pegTipPosition = sim.getObjectPosition(pegTipHandle, sim.handle_world)
    position = [pegTipPosition[0],pegTipPosition[1]+d*math.sin(math.pi-elevation),pegTipPosition[2]+d*math.cos(math.pi-elevation)] # hole is 0.2m
    # roll = np.random.uniform(-5*deg, 5*deg)
    rotvec = hyperSphereSurfaceSampler(3, np.random.uniform(0, rotRange * deg))
    rotvec[0] = rotvec[0]+ elevation
    cameraPose = Rotation.from_rotvec(rotvec).as_quat()
    # print(position)
    position_disturbance = hyperSphereSurfaceSampler(3, np.random.uniform(0, positionRange))
    position = list(np.array(position) + position_disturbance)
    sim.setObjectPosition(objectHandle, sim.handle_world, position)
    quat = list(cameraPose)
    # print(quat)
    sim.setObjectQuaternion(objectHandle, sim.handle_world, quat)
    return position, quat

def resetColor(objectHandle, rgb = None):
    # color_name = "random"
    color_component = sim.colorcomponent_ambient_diffuse
    # color_component = sim.colorcomponent_specular
    # color_component = sim.colorcomponent_emission
    if rgb is None:
        rgb_data = list(np.random.uniform(0, 1, 3))
    else:
        rgb_data = rgb
        # print(rgb_data)
    sim.setObjectColor(objectHandle, 0, color_component, rgb_data)
    return rgb_data

def resetLight(objectHandle,d_z,d_xy):
    # d_z = 2
    # d_xy = 1
    rot_z = np.random.uniform(0, 360 * deg)
    position = [d_xy * math.cos(rot_z), d_xy * math.sin(rot_z), d_z]
    # roll = np.random.uniform(-5*deg, 5*deg)
    vector_z = np.array([0,0,0.2])- np.array(position) #hole position - light position
    cameraPose_Z= vector_z/np.linalg.norm(vector_z)
    # vector_x = np.concatenate((position[0:2],[0])).dot(cameraPose_Z)
    vector_x = np.cross(np.concatenate((position[0:2],[0])),cameraPose_Z)
    cameraPose_X = vector_x/np.linalg.norm(vector_x)
    # cameraPose_Y = cameraPose_Z.dot(cameraPose_X)
    cameraPose_Y = np.cross(cameraPose_Z,cameraPose_X)
    cameraMat = np.concatenate((cameraPose_X,cameraPose_Y,cameraPose_Z)).reshape(3,3).T
    cameraPose = Rotation.from_matrix(cameraMat).as_quat()
    # print(position)
    sim.setObjectPosition(objectHandle, sim.handle_world, position)
    quat = list(cameraPose)
    # print(quat)
    sim.setObjectQuaternion(objectHandle, sim.handle_world, quat)
    return position, quat

def getAnnotate(cameraHandle,pegTipHandle,holeTipHandle,resolution):
    success = True
    cameraMat = sim.getObjectMatrix(cameraHandle,sim.handle_world)
    cameraPosition = np.array(cameraMat).reshape(3,4)[:,3]
    cameraRot = np.array(cameraMat).reshape(3, 4)[:, 0:3]
    pegPosition = sim.getObjectPosition(pegTipHandle, sim.handle_world)
    holePosition = sim.getObjectPosition(holeTipHandle, sim.handle_world)
    pegVector = cameraRot.T.dot(np.array(pegPosition)-cameraPosition)
    holeVector = cameraRot.T.dot(np.array(holePosition) - cameraPosition)

    f=1
    visualAngle = 30
    alfa_x = (resolution[0]/2) / (f * math.tan(visualAngle*deg))
    alfa_y = (resolution[1]/2) / (f * math.tan(visualAngle*deg))
    pegXI = alfa_x * f * pegVector[0] / pegVector[2] + resolution[0]/2
    pegYI = alfa_y * f * pegVector[1] / pegVector[2] + resolution[1]/2
    pegPositonI = [int(resolution[1]) - round(pegYI),int(resolution[0]) - round(pegXI)]
    # pegPositonI = [round(pegXI),round(pegYI)]

    holeXI = alfa_x * f * holeVector[0] / holeVector[2] + resolution[0] / 2
    holeYI = alfa_y * f * holeVector[1] / holeVector[2] + resolution[1] / 2
    holePositonI = [int(resolution[1]) - round(holeYI), int(resolution[0]) - round(holeXI)]
    # holePositonI = [round(holeXI), round(holeYI)]
    if min([pegXI, pegYI, holeXI, holeYI])<15 or max([pegXI, holeXI])>resolution[0]-15 or max([pegYI, holeYI])>resolution[1]-15:
        success = False

    return pegPositonI, holePositonI, success

def sceneReset(pegHandle):
    # resetHole(sim)
    resetPeg(pegHandle)
    # resetCamera()
    # resetLight()

def getImage(objectHandle):
    # sim.handleVisionSensor(objectHandle)
    image, resolution = sim.getVisionSensorImg(objectHandle,  0, 0.0, [0, 0], [0, 0])
    resolutionY = resolution[0]
    resolutionX = resolution[1]
    image_rgb_r = [image[i] for i in range(0, len(image), 3)]
    image_rgb_r = np.array(image_rgb_r)
    image_rgb_r = image_rgb_r.reshape(resolutionY, resolutionX)
    image_rgb_r = image_rgb_r.astype(np.uint8)

    image_rgb_g = [image[i] for i in range(1, len(image), 3)]
    image_rgb_g = np.array(image_rgb_g)
    image_rgb_g = image_rgb_g.reshape(resolutionY, resolutionX)
    image_rgb_g = image_rgb_g.astype(np.uint8)

    image_rgb_b = [image[i] for i in range(2, len(image), 3)]
    image_rgb_b = np.array(image_rgb_b)
    image_rgb_b = image_rgb_b.reshape(resolutionY, resolutionX)
    image_rgb_b = image_rgb_b.astype(np.uint8)

    result_rgb = cv2.merge([image_rgb_b, image_rgb_g, image_rgb_r])
    # 镜像翻转, opencv在这里返回的是一张翻转的图
    result_rgb = cv2.flip(result_rgb, 0)
    # cv2.imshow('rgb',result_rgb)
    # cv2.waitKey(1000)
    # plt.imshow(result_rgb)
    # plt.show()
    # matplotlib.image.imsave(args.image_path+'/test.png', result_rgb)
    return result_rgb, resolution

def poseSync(pegHandle2,pegHandle3,cameraHandle2,cameraHandle3,pegPosition,pegQuat,cameraPosition,cameraQuat):
    pegPosition2 = np.copy(pegPosition)
    pegPosition3 = np.copy(pegPosition)
    cameraPosition2 = np.copy(cameraPosition)
    cameraPosition3 = np.copy(cameraPosition)
    pegPosition2[0] = pegPosition2[0] + 5
    pegPosition3[0] = pegPosition3[0] + 7
    cameraPosition2[0] = cameraPosition2[0] + 5
    cameraPosition3[0] = cameraPosition3[0] + 7
    sim.setObjectPosition(pegHandle2, sim.handle_world, list(pegPosition2))
    sim.setObjectQuaternion(pegHandle2, sim.handle_world, pegQuat)
    sim.setObjectPosition(pegHandle3, sim.handle_world, list(pegPosition3))
    sim.setObjectQuaternion(pegHandle3, sim.handle_world, pegQuat)
    sim.setObjectPosition(cameraHandle2, sim.handle_world, list(cameraPosition2))
    sim.setObjectQuaternion(cameraHandle2, sim.handle_world, cameraQuat)
    sim.setObjectPosition(cameraHandle3, sim.handle_world, list(cameraPosition3))
    sim.setObjectQuaternion(cameraHandle3, sim.handle_world, cameraQuat)

def stateClassify(cameraHandle, cameraHandle2, cameraHandle3, pegTipHandle, holeTipHandle):
    rgb2, resolution = getImage(cameraHandle2)
    rgb3, resolution = getImage(cameraHandle3)
    rgb2 = np.transpose(rgb2, axes=(2, 0, 1))
    rgb3 = np.transpose(rgb3, axes=(2, 0, 1))
    mask2 = np.zeros(resolution)
    mask3 = np.zeros(resolution)
    mask3[np.where(rgb3[0] > 200)] = 1
    mask2[np.where(rgb2[0] > 200)] = 1
    # plt.figure()
    # plt.imshow(mask2)
    # plt.figure()
    # plt.imshow(mask3)
    # plt.figure()
    # plt.imshow(np.transpose(rgb2,[1,2,0]))
    # plt.figure()
    # plt.imshow(np.transpose(rgb3,[1,2,0]))
    # plt.show()

    mask2Area = np.sum(mask2)
    mask3Area = np.sum(mask3)

    pegPositonI, holePositonI, success = getAnnotate(cameraHandle, pegTipHandle, holeTipHandle, resolution) #矩阵坐标系
    if abs(mask2Area - mask3Area) < 20:
        stateIdx = 1  # 分离
    elif holePositonI[0] - pegPositonI[0] > 5:
        stateIdx = 2  # 重叠
    elif pegPositonI[0] - holePositonI[0] > 0:
        stateIdx = 0  # 遮挡
    else:
        stateIdx = 3 # 分类不明显
    # stateIdx = 0
    return stateIdx

def getMask(cameraHandle, cameraHandle2, cameraHandle3, pegTipHandle, holeTipHandle):
    rgb2, resolution = getImage(cameraHandle2)
    rgb3, resolution = getImage(cameraHandle3)
    rgb2 = np.transpose(rgb2, axes=(2, 0, 1))
    rgb3 = np.transpose(rgb3, axes=(2, 0, 1))
    maskHole = np.zeros(resolution)
    maskPeg = np.zeros(resolution)
    maskHole[np.where(rgb2[0] > 200)] = 1
    maskPeg[np.where(rgb2[2] > 200)] = 1
    # plt.figure()
    # plt.imshow(mask2)
    # plt.figure()
    # plt.imshow(mask3)
    # plt.figure()
    # plt.imshow(np.transpose(rgb2,[1,2,0]))
    # plt.figure()
    # plt.imshow(np.transpose(rgb3,[1,2,0]))
    # plt.show()

    # mask2Area = np.sum(mask2)
    # mask3Area = np.sum(mask3)

    # pegPositonI, holePositonI, success = getAnnotate(cameraHandle, pegTipHandle, holeTipHandle, resolution) #矩阵坐标系
    # if abs(mask2Area - mask3Area) < 20:
    #     stateIdx = 1  # 分离
    # elif holePositonI[0] - pegPositonI[0] > 5:
    #     stateIdx = 2  # 重叠
    # elif pegPositonI[0] - holePositonI[0] > 0:
    #     stateIdx = 0  # 遮挡
    # else:
    #     stateIdx = 3 # 分类不明显
    # stateIdx = 0
    return maskHole,maskPeg

def makeDataset():
    pegHandle = sim.getObject('/peg')
    holeHandle = sim.getObject('/Table')
    holeHandle2 = sim.getObject('/holeVrep')
    cameraHandle = sim.getObject('/Vision_sensor')
    lightHandle = sim.getObject('/Spotlight')
    pegTipHandle = sim.getObject('/pegTip')
    holeTipHandle = sim.getObject('/holeTip')

    pegHandle2 = sim.getObject('/peg2')
    cameraHandle2 = sim.getObject('/Vision_sensor2')
    pegHandle3 = sim.getObject('/peg3')
    cameraHandle3 = sim.getObject('/Vision_sensor3')
    imageNum = 0
    separatedNum = 0
    overlappedNum = 0
    shadedNum = 0
    # [0.22, 0.27] , [0.305, 0.355]
    while imageNum < np.sum(args.image_num):
        if separatedNum < args.image_num[1]:
            pegPosition, pegQuat, pegRgb = resetPeg(pegHandle, [-0.05,0.05], [-0.05,0.08], [0.22, 0.27], 10)# 先更新peg，再更新camera
            if np.random.rand() < 0.5:
                holeRgb = restHole(holeHandle,holeHandle2,pegRgb)
            client.step()
            # pegPosition, pegQuat = resetPeg(pegHandle, 0.05, [0.305, 0.355], 10)
            cameraPosition, cameraQuat = resetCamera(pegTipHandle, cameraHandle,[0.1, 0.15],[35, 65], 10, 0.03)
        elif shadedNum < args.image_num[0]:
            pegPosition, pegQuat, pegRgb = resetPeg(pegHandle, [-0.015,0.015], [0.01,0.08], [0.22, 0.27], 10)  # 先更新peg，再更新camera
            if np.random.rand() < 0.5:
                holeRgb = restHole(holeHandle, holeHandle2, pegRgb)
            client.step()
            cameraPosition, cameraQuat = resetCamera(pegTipHandle, cameraHandle, [0.1, 0.15], [35, 65], 10, 0.03)
        else:
            pegPosition, pegQuat, pegRgb = resetPeg(pegHandle, [-0.015,0.015], [-0.015,0.05], [0.22, 0.27], 10)# 先更新peg，再更新camera
            if np.random.rand() < 0.5:
                holeRgb = restHole(holeHandle, holeHandle2, pegRgb)
            client.step()
            cameraPosition, cameraQuat = resetCamera(pegTipHandle, cameraHandle, [0.1, 0.15], [35, 65], 10, 0.03)

        poseSync(pegHandle2, pegHandle3, cameraHandle2, cameraHandle3, pegPosition, pegQuat, cameraPosition, cameraQuat)
        # client.step()
        # if np.random.rand() < 0.7:
        lightPosition, lightQuat = resetLight(lightHandle,1.5,1.5)
        # else:
        #     lightPosition, lightQuat = resetLight(lightHandle, -10, 2)
        # poseSync(pegHandle, pegHandle2, pegHandle3)
        client.step()
        result_rgb, resolution = getImage(cameraHandle)
        pegPositonI, holePositonI, success = getAnnotate(cameraHandle, pegTipHandle, holeTipHandle, resolution)
        stateIdx = stateClassify(cameraHandle, cameraHandle2, cameraHandle3, pegTipHandle, holeTipHandle)
        if success:
            print(stateIdx)
            if stateIdx == 0:
                if shadedNum<args.image_num[stateIdx]:
                    iamgeName = str(stateIdx) + '-' + str(shadedNum)
                    saveImage = True
                    shadedNum = shadedNum + 1
                else:
                    saveImage = False

            elif stateIdx == 1:
                if separatedNum < args.image_num[stateIdx]:
                    iamgeName = str(stateIdx) + '-' + str(separatedNum)
                    saveImage = True
                    separatedNum = separatedNum + 1
                else:
                    saveImage = False

            elif stateIdx == 2:
                if overlappedNum < args.image_num[stateIdx]:
                    iamgeName = str(stateIdx) + '-' + str(overlappedNum)
                    saveImage = True
                    overlappedNum = overlappedNum + 1
                else:
                    saveImage = False

            elif stateIdx == 3:
                print('Category not obvious.')
                saveImage = False

            else:
                raise ValueError
            if saveImage:
                maskHole, maskPeg = getMask(cameraHandle, cameraHandle2, cameraHandle3, pegTipHandle, holeTipHandle)
                annotate={}
                annotate['peg_pos'] = pegPositonI
                annotate['hole_pos'] = holePositonI
                annotate['hole_size'] = 20 #mm
                annotate['cameraPosition'] = cameraPosition
                annotate['cameraQuat'] = cameraQuat
                annotate['pegPosition'] = pegPosition
                annotate['pegQuat'] = pegQuat
                annotate['lightPosition'] = lightPosition
                annotate['lightQuat'] = lightQuat
                annotate['resolution'] = resolution
                annotate['stateIdx'] = stateIdx
                annotate['maskHole'] = list(maskHole.flat)
                annotate['maskPeg'] = list(maskPeg.flat)

                if args.val:
                    matplotlib.image.imsave(args.val_image_path + '/' + iamgeName + '.png', result_rgb)
                    annotate_path = args.val_annotate_path + '/' + iamgeName + '.json'
                    with open(annotate_path, 'w') as json_file:
                        json.dump(annotate, json_file)
                else:
                    matplotlib.image.imsave(args.image_path + '/' + iamgeName + '.png', result_rgb)
                    annotate_path = args.annotate_path + '/' + iamgeName + '.json'
                    with open(annotate_path, 'w') as json_file:
                        json.dump(annotate, json_file)
                imageNum = imageNum + 1
        else:
            print('Peg out of range!')
    sim.stopSimulation()

def SynchronizeDataset(image_fp,annotate_fp,target_image_fp,target_annotate_fp):
    pegHandle = sim.getObject('/peg')
    holeHandle = sim.getObject('/Table')
    holeHandle2 = sim.getObject('/holeVrep')
    cameraHandle = sim.getObject('/Vision_sensor')
    lightHandle = sim.getObject('/Spotlight')
    pegTipHandle = sim.getObject('/pegTip')
    holeTipHandle = sim.getObject('/holeTip')

    pegHandle2 = sim.getObject('/peg2')
    cameraHandle2 = sim.getObject('/Vision_sensor2')
    pegHandle3 = sim.getObject('/peg3')
    cameraHandle3 = sim.getObject('/Vision_sensor3')

    for idx in range(300):
        annotate_fps = list(Path(annotate_fp).glob('*.json'))
        annot_fp = annotate_fps[idx]
        image_fps = list(Path(image_fp).glob('*.png'))
        img_fp = image_fps[idx]
        # target_annotate_fps = list(Path(target_annotate_fp).glob('*.json'))
        # target_annot_fp = target_annotate_fps[idx]
        # target_image_fps = list(Path(target_image_fp).glob('*.json'))
        # target_img_fp = target_image_fps[idx]
        with open(annot_fp, 'r') as json_file:
            annotation_dict = json.load(json_file)
            cameraPosition = annotation_dict['cameraPosition']
            cameraQuat = annotation_dict['cameraQuat']
            pegPosition = annotation_dict['pegPosition']
            pegQuat = annotation_dict['pegQuat']
            lightPosition = annotation_dict['lightPosition']
            lightQuat = annotation_dict['lightQuat']

            sim.setObjectPosition(pegHandle, sim.handle_world, list(pegPosition))
            sim.setObjectQuaternion(pegHandle, sim.handle_world, list(pegQuat))
            sim.setObjectPosition(cameraHandle, sim.handle_world, list(cameraPosition))
            sim.setObjectQuaternion(cameraHandle, sim.handle_world, list(cameraQuat))
            sim.setObjectPosition(lightHandle, sim.handle_world, list(lightPosition))
            sim.setObjectQuaternion(lightHandle, sim.handle_world, list(lightQuat))
            client.step()
            poseSync(pegHandle2, pegHandle3, cameraHandle2, cameraHandle3, pegPosition, pegQuat, cameraPosition,
                     cameraQuat)
            pegRgb = resetColor(pegHandle)
            if np.random.rand() < 0.5:
                holeRgb = restHole(holeHandle,holeHandle2, pegRgb)
            lightPosition, lightQuat = resetLight(lightHandle, 1.5, 1.5)
            client.step()
            result_rgb, resolution = getImage(cameraHandle)
            iamgeName = str(img_fp).split("\\")[-1]
            matplotlib.image.imsave(target_image_fp + '/' + iamgeName[0:-4] + '.png', result_rgb)

            maskHole,maskPeg = getMask(cameraHandle, cameraHandle2, cameraHandle3, pegTipHandle, holeTipHandle)
            annotation_dict['lightPosition'] = lightPosition
            annotation_dict['lightQuat'] = lightQuat
            annotation_dict['maskHole'] = list(maskHole.flat)
            annotation_dict['maskPeg'] = list(maskPeg.flat)


            # plt.imshow(maskHole)
            # plt.show()
            # plt.imshow(maskPeg)
            # plt.show()
            # iamgeName = str(annot_fp).split("\\")[-1]
            annotate_path = target_annotate_fp + '/' + iamgeName[0:-4] + '.json'
            with open(annotate_path, 'w') as json_file2:
                json.dump(annotation_dict, json_file2)
    sim.stopSimulation()

def addMask(annotate_fp):
    pegHandle = sim.getObject('/peg')
    holeHandle = sim.getObject('/Table')
    holeHandle2 = sim.getObject('/holeVrep')
    cameraHandle = sim.getObject('/Vision_sensor')
    lightHandle = sim.getObject('/Spotlight')
    pegTipHandle = sim.getObject('/pegTip')
    holeTipHandle = sim.getObject('/holeTip')

    pegHandle2 = sim.getObject('/peg2')
    cameraHandle2 = sim.getObject('/Vision_sensor2')
    pegHandle3 = sim.getObject('/peg3')
    cameraHandle3 = sim.getObject('/Vision_sensor3')

    for idx in range(300):
        annotate_fps = list(Path(annotate_fp).glob('*.json'))
        annot_fp = annotate_fps[idx]
        with open(annot_fp, 'r') as json_file:
            annotation_dict = json.load(json_file)
            cameraPosition = annotation_dict['cameraPosition']
            cameraQuat = annotation_dict['cameraQuat']
            pegPosition = annotation_dict['pegPosition']
            pegQuat = annotation_dict['pegQuat']
            lightPosition = annotation_dict['lightPosition']
            lightQuat = annotation_dict['lightQuat']

            sim.setObjectPosition(pegHandle, sim.handle_world, list(pegPosition))
            sim.setObjectQuaternion(pegHandle, sim.handle_world, list(pegQuat))
            sim.setObjectPosition(cameraHandle, sim.handle_world, list(cameraPosition))
            sim.setObjectQuaternion(cameraHandle, sim.handle_world, list(cameraQuat))
            sim.setObjectPosition(lightHandle, sim.handle_world, list(lightPosition))
            sim.setObjectQuaternion(lightHandle, sim.handle_world, list(lightQuat))
            client.step()
            poseSync(pegHandle2, pegHandle3, cameraHandle2, cameraHandle3, pegPosition, pegQuat, cameraPosition,
                     cameraQuat)
            client.step()
            maskHole, maskPeg = getMask(cameraHandle, cameraHandle2, cameraHandle3, pegTipHandle, holeTipHandle)
            annotation_dict['maskHole'] = list(maskHole.flat)
            annotation_dict['maskPeg'] = list(maskPeg.flat)
            # plt.imshow(maskHole)
            # plt.show()
            # plt.imshow(maskPeg)
            # plt.show()
            iamgeName = str(annot_fp).split("\\")[-1]
            annotate_path = annotate_fp + '/' + 'annotate_new/' + iamgeName + '.json'
            with open(annotate_path, 'w') as json_file2:
                json.dump(annotation_dict, json_file2)
    sim.stopSimulation()


    # result_rgb[pegPositonI[0]][pegPositonI[1]] = np.array([1, 1, 1])
    # result_rgb[holePositonI[0]][holePositonI[1]] = np.array([0, 0, 0])
    # plt.imshow(result_rgb)
    # plt.show()

if __name__ == '__main__':
    # pegHandle = sim.getObject('/peg')
    # cameraHandle = sim.getObject('/Vision_sensor')
    # lightHandle = sim.getObject('/Spotlight')
    # pegTipHandle = sim.getObject('/pegTip')
    # holeTipHandle = sim.getObject('/holeTip')
    # resetCamera(cameraHandle)
    # resetPeg(pegHandle)
    # resetLight(lightHandle)
    # result_rgb, resolution = getImage(cameraHandle)
    # getAnnotate(cameraHandle, pegTipHandle, holeTipHandle, resolution)
    # client.step()
    makeDataset()
    # annotate_fps=list(Path('../dataset/classify/train3/annotate').glob('*.json'))
    # annotate_fp = '../dataset/classify/train4/annotate'
    # annotate_fp = '../dataset/validationDataset4/annotate'
    # addMask(annotate_fp)
    # image_fp = '../dataset/vrepImageVal/hole2/image'
    # annotate_fp = '../dataset/vrepImageVal/hole2/annotate'
    # target_image_fp = '../dataset/vrepImageVal/hole11/image'
    # target_annotate_fp = '../dataset/vrepImageVal/hole11/annotate'
    # SynchronizeDataset(image_fp, annotate_fp, target_image_fp, target_annotate_fp)
