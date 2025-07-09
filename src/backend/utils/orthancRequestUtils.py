"""
https://orthanc.uclouvain.be/api/index.html#tag/Patients
 - This file downloads from Orthanc server using python and REST API
"""

import io
import pdb
import torch
import zipfile
import tempfile
import requests
import pydicom
import pprint
import traceback
import datetime
import numpy as np
from pathlib import Path, WindowsPath
import matplotlib.pyplot as plt

import pydicom_seg # to read .dcm with modality=SEG

import warnings
warnings.filterwarnings("ignore", message="The value length .* exceeds the maximum length of 64 allowed for VR LO.")

MODALITY_CT  = 'CT'
MODALITY_PT  = 'PT'
MODALITY_SEG = 'SEG'

URL_DICOM_SERVER = 'http://localhost:8042'
URL_ENDPOINT = 'dicom-web'

KEY_ID           = 'ID'
KEY_ORTHANC_ID   = 'OrthancId'
KEY_STUDIES    = 'Studies'
KEY_SERIES     = 'Series'
KEY_INSTANCES  = 'Instances'
KEY_STUDIES_ORTHANC_ID = 'StudiesOrthancId'
KEY_SERIES_ORTHANC_ID  = 'SeriesOrthancId'
KEY_INSTANCE_ORTHANC_ID = 'InstanceOrthancId'
KEY_STUDY_UID   = 'StudyUID'
KEY_SERIES_UID  = 'SeriesUID'
KEY_INSTANCE_UID = 'InstanceUID'
KEY_MODALITY    = 'Modality'

KEY_MODALITY_SEG = 'SEG'
KEY_SERIES_DESC  = 'SeriesDescription'

IDENTIFIER_SERIES_DESC_GT = 'Series-SEG-GT'
IDENTIFIER_SERIES_DESC_PRED = 'Series-SEG-Pred'

SERIES_DESC_GT  = '{}-' + IDENTIFIER_SERIES_DESC_GT
SERIES_DESC_PRED = '{}-' + IDENTIFIER_SERIES_DESC_PRED

PRIVATE_BLOCK_GROUP   = 0x1001
PRIVATE_BLOCK_CREATOR = 'Mody - AI-assisted Interactive Refinement v1.0'
TAG_OFFSET           = 0x1000
TAG_TIME_TO_BRUSH    = 0x01 # (1001,1001)	Unknown  Tag &  Data	1.8380001
TAG_TIME_TO_SCRIBBLE = 0x02
TAG_TIME_TO_DISTMAP  = 0x03
TAG_TIME_TO_INFER    = 0x04
TAG_TIME_EPOCH       = 0x05
VALUEREP_FLOAT32     = pydicom.valuerep.VR.FL # Floating Point Single 
VALUEREP_STRING      = pydicom.valuerep.VR.ST # Short Text

def getOrthancPatientIds(specificPatientId=None):
    """
    Returns
    -------
    res: dict, {'CHMR001': KEY_ORTHANC_ID: '8c9400a8-e7942cc9-a453c142-9e072032-b158df2e', 'KEY_STUDIES_ORTHANC_ID': ''}
    """

    res = {}

    try:

        # Step 1 - Get Orthanc Patient IDs
        query = URL_DICOM_SERVER + '/patients'
        response = requests.get(query, verify=False)
        if response.status_code == 200:
            patientOrthancIds = response.json()
            for patientOrthancId in patientOrthancIds:

                # Step 2 - Get Patient Data
                patientQuery = URL_DICOM_SERVER + '/patients/' + patientOrthancId
                patientResponse = requests.get(patientQuery, verify=False)
                if patientResponse.status_code == 200:
                    patientData        = patientResponse.json()
                    patientActualId    = patientData['MainDicomTags']['PatientID']
                    patientStudiesOrthancIds = patientData['Studies']
                    if specificPatientId is not None and patientActualId != specificPatientId:
                        continue
                    res[patientActualId] = {
                        KEY_ORTHANC_ID: patientOrthancId,
                        KEY_STUDIES: []
                    }
                    for patientStudiesOrthancId in patientStudiesOrthancIds:
                        res[patientActualId][KEY_STUDIES].append({KEY_STUDIES_ORTHANC_ID: patientStudiesOrthancId, KEY_STUDY_UID: None, KEY_SERIES: []})
                        
                        # Step 3 - Get Study Data
                        studyRequest = URL_DICOM_SERVER + '/studies/' + patientStudiesOrthancId
                        studyResponse = requests.get(studyRequest, verify=False)
                        if studyResponse.status_code == 200:
                            studyData = studyResponse.json()
                            studyUID  = studyData['MainDicomTags']['StudyInstanceUID']
                            res[patientActualId][KEY_STUDIES][-1][KEY_STUDY_UID] = studyUID
                            seriesOrthancIds = studyData['Series']
                            for seriesOrthancId in seriesOrthancIds:
                                res[patientActualId][KEY_STUDIES][-1][KEY_SERIES].append({KEY_SERIES_ORTHANC_ID: seriesOrthancId, KEY_SERIES_DESC: None, KEY_SERIES_UID: None, KEY_MODALITY: None, KEY_INSTANCE_UID: None})
                                
                                # Step 4 - Get Series Data
                                seriesRequest = URL_DICOM_SERVER + '/series/' + seriesOrthancId
                                seriesResponse = requests.get(seriesRequest, verify=False)
                                if seriesResponse.status_code == 200:
                                    seriesData = seriesResponse.json()
                                    seriesDesc = seriesData['MainDicomTags'].get('SeriesDescription', None)
                                    seriesUID  = seriesData['MainDicomTags']['SeriesInstanceUID']
                                    modality   = seriesData['MainDicomTags']['Modality']
                                    res[patientActualId][KEY_STUDIES][-1][KEY_SERIES][-1][KEY_SERIES_DESC] = seriesDesc
                                    res[patientActualId][KEY_STUDIES][-1][KEY_SERIES][-1][KEY_SERIES_UID] = seriesUID
                                    res[patientActualId][KEY_STUDIES][-1][KEY_SERIES][-1][KEY_MODALITY] = modality
                                    
                                    # Step 5 - Get Instance Data (for SEG only)
                                    if modality == KEY_MODALITY_SEG:
                                        # print (seriesData)
                                        instanceRequest = URL_DICOM_SERVER + '/instances/' + seriesData['Instances'][0]
                                        instanceResponse = requests.get(instanceRequest, verify=False)
                                        if instanceResponse.status_code == 200:
                                            instanceData = instanceResponse.json()
                                            instanceUID  = instanceData['MainDicomTags']['SOPInstanceUID']
                                            res[patientActualId][KEY_STUDIES][-1][KEY_SERIES][-1][KEY_INSTANCE_UID] = instanceUID
                                            res[patientActualId][KEY_STUDIES][-1][KEY_SERIES][-1][KEY_INSTANCE_ORTHANC_ID] = instanceData['ID']
                                        else:
                                            print (' - [ERROR][getOrthancPatientIds()] instanceResponse: ', instanceResponse.status_code, instanceResponse.reason)
                                else:
                                    print (' - [ERROR][getOrthancPatientIds()] seriesResponse: ', seriesResponse.status_code, seriesResponse.reason)
                        else:
                            print (' - [ERROR][getOrthancPatientIds()] studyResponse: ', studyResponse.status_code, studyResponse.reason)
                        
                else:
                    print (' - [ERROR][getOrthancPatientIds()] patientResponse: ', patientResponse.status_code, patientResponse.reason)
        else:
            print (' - [getOrthancPatientIds()] response: ', response.status_code, response.reason)
            
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return res

def getDownloadedFilePaths(tmpDirPath, zipContent):

    res = {}

    try:
        
        # Step 1 - Loop over zip files
        fileObj    = io.BytesIO(zipContent)
        zipObj     = zipfile.ZipFile(fileObj, 'r')
        for filename in zipObj.namelist():

            # Step 2 - Save file to disk
            zipObj.extract(filename, tmpDirPath)
            filePath = Path(tmpDirPath) / filename

            # Step 3 - Read DICOM file (for modality=[CT, PT, SEG])
            ds       = pydicom.dcmread(filePath, stop_before_pixels=True, specific_tags=[
                pydicom.tag.Tag((0x0008,0x0060)), pydicom.tag.Tag((0x0020, 0x0013))  
                ])
            
            if 'Modality' in ds:
                modality       = ds.Modality
            
            # Step 4 - Save file paths (according to modality)
            if modality in [MODALITY_CT, MODALITY_PT]:
                instanceNumber = None
                if 'InstanceNumber' in ds:
                    instanceNumber = int(ds.InstanceNumber)
                
                if modality not in res: 
                    if instanceNumber is not None:
                        res[modality] = {instanceNumber: filePath}
                else: 
                    res[modality][instanceNumber] = filePath

            elif modality in [MODALITY_SEG]:
                if modality not in res: 
                    res[modality] = [filePath]
                else: 
                    res[modality].append(filePath)

    except:
        traceback.print_exc()
        pdb.set_trace()
    

    # Step 99 - Sort MODALITY_CT and MODALITY_PT by key values and only keep the values
    if MODALITY_CT in res:
        res[MODALITY_CT] = [val for key, val in sorted(res[MODALITY_CT].items())]
    if MODALITY_PT in res:
        res[MODALITY_PT] = [val for key, val in sorted(res[MODALITY_PT].items())]
    
    return res

def convertDcmToTorchArray(dcmFilePaths):

    res = []

    try:
        
        for dcmFilePath in dcmFilePaths:
            ds = pydicom.dcmread(dcmFilePath)
            res.append(torch.Tensor(ds.pixel_array))

        if len(res):
            res = torch.stack(res,-1)

    except:
        traceback.print_exc()
        pdb.set_trace()

    return res

def plot(ctArray, ptArray, maskPredArray=None):
    """
    Params:
        ctArray: [H, W, D] # (coronal, saggital, axial)
        ptArray: [H, W, D]
        maskPredArray: [H, W, D]
    """
    
    try:
        
        randomSliceIds = np.random.choice(ctArray.shape[2], 3)
        f,axarr = plt.subplots(len(randomSliceIds),3,figsize=(15,15))

        for i, sliceId in enumerate(randomSliceIds):

            # Plot CT
            axarr[i,0].imshow(ctArray[:,:,sliceId], cmap='gray')
            axarr[i,0].set_ylabel('Slice: ' + str(sliceId))

            # Plot PT
            axarr[i,1].imshow(ptArray[:,:,sliceId], cmap='gray')

            # Plot Mask
            axarr[i,2].imshow(ctArray[:,:,sliceId], cmap='gray')
            axarr[i,2].imshow(ptArray[:,:,sliceId], cmap='gray', alpha=0.5)
            if maskPredArray is not None:
                axarr[i,2].contour(maskPredArray[:,:,sliceId])
        
        plt.show()


    except:
        traceback.print_exc()

def getPyDicomObjects(patientId, patientIdObj, modality, seriesDesc=None):

    try:
        
        # Step 0 - Init
        modalitySeriesURL = '/'.join([URL_DICOM_SERVER, KEY_SERIES.lower(), '{}', KEY_INSTANCES.lower()])
        instanceURL       = '/'.join([URL_DICOM_SERVER, KEY_INSTANCES.lower(), '{}', 'file'])
        modalitySeriesOrthancId = None
        listOfPydicomObjects = []

        # Step 1 - Get Orthanc Series ID
        for seriesObj in patientIdObj[patientId][KEY_STUDIES][0][KEY_SERIES]:
            if seriesObj[KEY_MODALITY] == modality:
                if seriesDesc is not None and seriesObj[KEY_SERIES_DESC] == seriesDesc:
                    modalitySeriesOrthancId = seriesObj[KEY_SERIES_ORTHANC_ID]
                else:
                    modalitySeriesOrthancId = seriesObj[KEY_SERIES_ORTHANC_ID]

        # Step 2 - Get URLs
        queryForSeries    = modalitySeriesURL.format(modalitySeriesOrthancId)
        responseForSeries = requests.get(queryForSeries)
        if responseForSeries.status_code == 200:
            instance_uids = responseForSeries.json()

            with tempfile.TemporaryDirectory() as tmpDirPath:
                for instance_uid_obj in instance_uids:
                    # print (' - instance_uid: ', instance_uid_obj[KEY_ID])
                    queryForInstance = instanceURL.format(instance_uid_obj[KEY_ID])
                    responseForInstance = requests.get(queryForInstance)
                    if responseForInstance.status_code == 200:
                        with open(Path(tmpDirPath) / 'instance.dcm', 'wb') as f:
                            f.write(responseForInstance.content)
                            ds = pydicom.dcmread(f.name)
                            listOfPydicomObjects.append(ds)
    
        # Step 3 - Sort listOfPydicomObjects
        try:
            listOfPydicomObjects.sort(key=lambda x: float(x.ImagePositionPatient[2]))
            # listOfPydicomObjects = sorted(listOfPydicomObjects, key=lambda x: x.InstanceNumber)
        except:
            pass

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return listOfPydicomObjects

def getSegsArray(patientId, patientIdObj):
    """
    Returns
    -------
    arrayGT, arrayPred: np.array, np.array (from ds.pixel_array)
    The arrayGT and arrayPred are in the shape of [axial, coronal, sagittal]
    """
    try:
        
        # Step 0 - Init
        segURL      = '/'.join([URL_DICOM_SERVER, KEY_INSTANCES.lower(), '{}', 'file'])
        orthancInstanceUIDGT, orthancInstanceUIDPred = None, None
        arrayGT, arrayPred = None, None
        # patientIdMeta = '-' + '-'.join(patientId.split('-')[1:])
        # if len(patientIdMeta): patientIdMeta = '-' + patientIdMeta

        # Step 1 - Get Orthanc Instance UIDs
        for seriesObj in patientIdObj[patientId][KEY_STUDIES][0][KEY_SERIES]:
            if seriesObj[KEY_MODALITY] == KEY_MODALITY_SEG:
                if seriesObj[KEY_SERIES_DESC] == SERIES_DESC_GT.format(patientId):
                    orthancInstanceUIDGT = seriesObj[KEY_INSTANCE_ORTHANC_ID]
                elif seriesObj[KEY_SERIES_DESC] == SERIES_DESC_PRED.format(patientId):
                    orthancInstanceUIDPred = seriesObj[KEY_INSTANCE_ORTHANC_ID]

        # Step 2 - Get URLs
        queryGT = segURL.format(orthancInstanceUIDGT)
        queryPred = segURL.format(orthancInstanceUIDPred)
        
        # Step 3 - Get Segs Array
        for query in [queryGT, queryPred]:
            with tempfile.TemporaryDirectory() as tmpDirPath:
                response = requests.get(query)
                if response.status_code == 200:
                    
                    # Step 2.1 - Save tmp file
                    with open(Path(tmpDirPath) / 'seg.dcm', 'wb') as f:
                        f.write(response.content)

                        # Step 2.2 - Read the dcm file
                        ds = pydicom.dcmread(f.name)
                        # if ds.SeriesDescription == SERIES_DESC_GT.format(patientId):
                        #     arrayGT = ds.pixel_array
                        # elif ds.SeriesDescription == SERIES_DESC_PRED.format(patientId):
                        #     arrayPred = ds.pixel_array
                        if IDENTIFIER_SERIES_DESC_GT in ds.SeriesDescription:
                            arrayGT = ds.pixel_array
                        elif IDENTIFIER_SERIES_DESC_PRED in ds.SeriesDescription:
                            arrayPred = ds.pixel_array

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return arrayGT, arrayPred

def downloadPatientZip(patientId, patientIdObj, modality=[], returns=False):

    ctArray, ptArray = None, None

    try:
        
        # Step 1 - Create a tmp folder
        with tempfile.TemporaryDirectory() as tmpDirPath:
            # print (' - tmpDirPath: ', tmpDirPath)

            # Step 2 - Get Orthanc Zip
            patientOrthancId = patientIdObj[patientId][KEY_ORTHANC_ID]
            query    = URL_DICOM_SERVER + '/patients/' + patientOrthancId + '/archive' # [TODO: I dont have to download all the data!]
            response = requests.post(query, data='{"Synchronous":true}', verify=False)
            if response.status_code == 200:
                
                # Step 3 - Extract Zip
                zipContent = response.content
                dcmFilePaths = getDownloadedFilePaths(tmpDirPath, zipContent)
                # print (' - dcmFilePaths: ', dcmFilePaths)
                # pdb.set_trace()

                # Step 4 - Convert dcms to torch arrays
                ctArray       = convertDcmToTorchArray(dcmFilePaths[MODALITY_CT])
                ptArray       = convertDcmToTorchArray(dcmFilePaths[MODALITY_PT])
                # maskPredArray = None
                # plot(ctArray, ptArray, maskPredArray)

                # dcmMaskPaths = dcmFilePaths[MODALITY_SEG]
                # dsGT, dsPred = None, None
                # pathMaskGT, pathMaskPred = None, None 
                # for dcmMaskPath in dcmMaskPaths:
                #     ds = pydicom.dcmread(dcmMaskPath, stop_before_pixels=True)
                #     if ds.SeriesDescription == SERIES_DESC_GT.format(patientId):
                #         pathMaskGT = dcmMaskPath
                #     elif ds.SeriesDescription == SERIES_DESC_PRED.format(patientId):
                #         pathMaskPred = dcmMaskPath

                # pdb.set_trace()
                

                # Step 5 - z-norm and concat data [CT, PET, Seg]

            else:
                print (' - [downloadPatientZip()] response: ', response.status_code, response.reason)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return ctArray, ptArray

def getSegArrayInShapeMismatchScene(listObjsCT, pathSeg, spacingZ=1, verbose=False):
    """
    Following RAS orientation (or inferior to superior)
     - the 0th slice is the most inferior slice (i.e. towards feet) and the last slice is the most superior slice (i.e. towards head).
    
    Here
     - we init segArrayFullSize with zeros and then fill it with the segArrayPartialSize (depending on the startIdx and endIdx)
    """

    segArrayFullSize = None
    seriesDesc       = None
    timeToBrushFromDataset = -1
    epochOfBrushStart = -1
    timeOfBrushStart = -1

    try:
        # Step 0 - Init
        startIdx, endIdx = -1, -1

        # Step 1 - Get PyDicomObjects
        dsFromC3D  = pydicom.dcmread(pathSeg)
        seriesDesc = dsFromC3D.SeriesDescription
        try:
            timeToBrushFromDataset = float(dsFromC3D[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_BRUSH)].value)
            epochOfBrushStart      = int(dsFromC3D[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_EPOCH)].value)
            timeOfBrushStart       = datetime.datetime.fromtimestamp(epochOfBrushStart/1000.0)
        except:
            pass

        # Step 2 - Find the startIdx and endIdx from dsFromC3D
        segImagePositionPatientStart = dsFromC3D.PerFrameFunctionalGroupsSequence[-1].PlanePositionSequence[0].ImagePositionPatient # inferior-most slice
        segImagePositionPatientEnd   = dsFromC3D.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient  # superior-most slice
        for idx, objCT in enumerate(listObjsCT):
            if objCT.ImagePositionPatient == segImagePositionPatientStart:
                startIdx = idx
            elif objCT.ImagePositionPatient == segImagePositionPatientEnd:
                endIdx = idx
        
        # Step 2.1 - There might be missing slices in the dsFromC3D(which are reflected in segArrayPartialSize)
        actualSliceIdxs   = np.array([float(obj.PlanePositionSequence[0].ImagePositionPatient[-1]) for obj in dsFromC3D.PerFrameFunctionalGroupsSequence])
        expectedSliceIdxs = np.arange(actualSliceIdxs.min(), actualSliceIdxs.max() + 1)
        missingSliceIdxs = np.setdiff1d(expectedSliceIdxs, actualSliceIdxs)

        # Step 3 - Create an empty segmentation array using the shape of the CT
        # print (' - [INFO] startIdx: ', startIdx, ' endIdx: ', endIdx)
        if startIdx != -1 and endIdx != -1:
            segArrayPartialSize = dsFromC3D.pixel_array # (coronal, sagittal, axial)
            segArrayFullSize    = np.zeros(listObjsCT[0].pixel_array.shape + (len(listObjsCT),))
            totalSlicesPartialSize = segArrayPartialSize.shape[0]
            
            rangeOfSliceIdxs = list(range(startIdx, endIdx+1))
            # rangeOfIdxs = list(range(startIdx, endIdx))
            iterActual = 0
            for iter, sliceIdx in enumerate(rangeOfSliceIdxs):
                try:
                    if sliceIdx in missingSliceIdxs:
                        if verbose: print ('    | - [INFO] Skipping sliceIdx: {}/(max={}) (iterActual={})'.format(sliceIdx, endIdx, iterActual))
                    else:
                        if iterActual <= len(rangeOfSliceIdxs) - len(missingSliceIdxs) - 1:
                            if len(missingSliceIdxs):
                                if verbose: print ('    | - [INFO] Processing sliceIdx: {}/(max={}) (iterActual={})'.format(sliceIdx, endIdx, iterActual))
                            # segArrayFullSize[..., endIdx-iter] = segArrayPartialSize[iter] # [Old] iter=0 seems to be superior here
                            # segArrayFullSize[..., sliceIdx] = segArrayPartialSize[totalSlicesPartialSize - 1 - iterActual]    # [New] iter=0 seems to be superior here
                            segArrayFullSize[..., sliceIdx] = segArrayPartialSize[-iterActual]    # [New] iter=0 seems to be superior here
                            iterActual += 1
                        else:
                            if verbose: print ('    | - [INFO] Skipping iter: ', iter)
                except:
                    print (f'    | - [ERROR][getSegArrayInShapeMismatchScene()] listObjSEGArray i.e .pixel_array): {segArrayPartialSize.shape}|| iter: {iter} | len(rangeOfIdxs): {len(rangeOfSliceIdxs)}')
                    traceback.print_exc()
                    pdb.set_trace()

            if verbose and len(missingSliceIdxs):
            # if 0:
                sliceCount = 5
                f,axarr = plt.subplots(3,sliceCount*2)
                axis3Idxs = np.argwhere(np.sum(segArrayFullSize, axis=(0,1))).flatten()
                axis2Idxs = np.argwhere(np.sum(segArrayFullSize, axis=(0,2))).flatten()
                axis1Idxs = np.argwhere(np.sum(segArrayFullSize, axis=(1,2))).flatten()
                fontSz=6
                for i in range(sliceCount):
                    axarr[0][i].imshow(segArrayFullSize[:,:,axis3Idxs[i]-1], cmap='gray'); axarr[0][i].set_title('SliceIdx: ' + str(axis3Idxs[i]-1), fontsize=fontSz)
                    axarr[1][i].imshow(segArrayFullSize[:,axis2Idxs[i]-1,:], cmap='gray'); axarr[1][i].set_title('SliceIdx: ' + str(axis2Idxs[i]-1), fontsize=fontSz)
                    axarr[2][i].imshow(segArrayFullSize[axis1Idxs[i]-1,:,:], cmap='gray'); axarr[2][i].set_title('SliceIdx: ' + str(axis1Idxs[i]-1), fontsize=fontSz)
                    axarr[0][i+sliceCount].imshow(segArrayFullSize[:,:,axis3Idxs[-sliceCount+i]+1], cmap='gray'); axarr[0][i+sliceCount].set_title('SliceIdx: ' + str(axis3Idxs[-sliceCount+i]+1), fontsize=fontSz)
                    axarr[1][i+sliceCount].imshow(segArrayFullSize[:,axis2Idxs[-sliceCount+i]+1,:], cmap='gray'); axarr[1][i+sliceCount].set_title('SliceIdx: ' + str(axis2Idxs[-sliceCount+i]+1), fontsize=fontSz)
                    axarr[2][i+sliceCount].imshow(segArrayFullSize[axis1Idxs[-sliceCount+i]+1,:,:], cmap='gray'); axarr[2][i+sliceCount].set_title('SliceIdx: ' + str(axis1Idxs[-sliceCount+i]+1), fontsize=fontSz)
                
                plt.suptitle('SeriesDesc: ' + seriesDesc + '\n In Axis3: startIdx: {} | endIdx: {} \n .pixel_array:{} || len(rangeOfIdxs): {}'.format(
                    startIdx, endIdx, segArrayPartialSize.shape, len(rangeOfSliceIdxs))
                    , y=1.0, fontsize=fontSz)
                plt.savefig('tmp-{}.png'.format(seriesDesc), dpi=300)
                pdb.set_trace()


            # Step 3.99 - Moveaxis
            segArrayFullSize = np.moveaxis(segArrayFullSize, [0,1,2], [2,1,0]) # (coronal, sagittal, axial) -> (axial, sagittal, coronal)
            segArrayFullSize = segArrayFullSize.astype(np.uint8)
            for axialSliceIdx in range(segArrayFullSize.shape[0]):
                # segArray[axialSliceIdx,::] = np.fliplr(np.rot90(segArray[axialSliceIdx,::], k=1)) # No
                segArrayFullSize[axialSliceIdx,::] = np.fliplr(np.rot90(segArrayFullSize[axialSliceIdx,::], k=3)) # Yesssss!!!!
        else:
            print ('   -> [ERROR][getSegArrayInShapeMismatchScene()] startIdx: {} | endIdx: {}'.format(startIdx, endIdx))
            pdb.set_trace()
    except:
        print ('\n   -> [ERROR][getSegArrayInShapeMismatchScene()] pathSeg: {}/{}\n'.format(*Path(pathSeg).parts[-2:]))
        traceback.print_exc()
        pdb.set_trace()
    
    return segArrayFullSize, seriesDesc, timeToBrushFromDataset, timeOfBrushStart, epochOfBrushStart

if __name__ == '__main__':

    try:
        
        # Step 1 - Define the patientId for download
        patientIdForDownload = None
        if 1:
            patientIdForDownload = 'CHMR028'
        
        # Step 2 - Download the patient zip (or for a specific modality)
        if patientIdForDownload is not None:
            patientIdObj = getOrthancPatientIds(patientIdForDownload)

            if 0:
                arrayGT, arrayPred = getSegsArray(patientIdForDownload, patientIdObj)
            elif 0:
                # listObjsCT = getPyDicomObjects(patientIdForDownload, patientIdObj, MODALITY_CT)
                listObjsSEGGT = getPyDicomObjects(patientIdForDownload, patientIdObj, MODALITY_SEG, seriesDesc=SERIES_DESC_GT.format(patientIdForDownload))
            elif 1:
                listObjsCT = getPyDicomObjects(patientIdForDownload, patientIdObj, MODALITY_CT)
                pathSeg = 'D:\\HCAI\\Project 5 - Interactive Contour Refinement\\code\\cornerstone3D-trials\\_experiments\\2024-10-15 12-06-39 -- gracious_torvalds__Prerak-Mody-NonExpert\\CHMR028-ManualRefine-1.dcm'
                _ = getSegArrayInShapeMismatchScene(listObjsCT, pathSeg)
            elif 0:
                downloadPatientZip(patientIdForDownload, patientIdObj, modality=[MODALITY_SEG], returns=True)
        

    except:
        traceback.print_exc()
        pdb.set_trace()



"""
// Make a curl request to localhost/patients
curl -X GET http://localhost:8042/patients

(ProstateX-0004)
curl -X GET http://localhost:8042/patients/b50780c0-21be0a34-3afd9d9e-e634aab6-61587b01
curl -X GET http://localhost:8042/studies/57577cc2-ce11a485-dc362647-57d39710-a5c8d4af

(HCAI-Interactive-XX)
curl -X GET http://localhost:8042/patients/42326a00-75df2637-b035c6ff-81927a0f-4af82587
curl -X GET http://localhost:8042/studies/937e2902-fd2d68cb-bd55de21-4da47523-65706f05
curl -X GET http://localhost:8042/series/d1682049-00cbdd91-16548797-5639007a-2eefe3b4

(CHMR-001)
curl -X GET http://localhost:8042/instances/8ba755b5-103b0834-405c40bd-b39e565c-6c97268d

(CHMR-028)
[Fail] curl -X GET http://localhost:8042/instances/1.2.826.0.1.3680043.8.498.78325365302799724640799265451668049689
[Fail] curl -X GET http://localhost:8042/instances/1.2.826.0.1.3680043.8.498.78325365302799724640799265451668049689/file3
[Works] curl -X GET http://localhost:8042/instances/a6843d8b-58e0adf0-4b05b279-d78017ee-75fb0bf7/file --output seg.dcm^

curl -X GET http://localhost:8042/series/164b5fa2-a98a839f-0c806186-c7beda38-930f5a76 --output seg.dcm
"""
