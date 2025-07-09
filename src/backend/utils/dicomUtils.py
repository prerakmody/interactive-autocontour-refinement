"""
This file converts .nii.gz files to .dcm files for CT, PT and SEG modalities.
"""
import sys
import pdb
import copy
import nrrd
import tqdm
import shutil
import random
import pydicom
import inspect
import traceback
import numpy as np
import nibabel as nib
from pathlib import Path
import SimpleITK as sitk

import torch

import plotext as pltTerm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# DICOM UIDs (Ref: http://dicom.nema.org/dicom/2013/output/chtml/part04/sect_I.4.html)
MODALITY_CT  = 'CT'
MODALITY_PT  = 'PT'
MODALITY_SEG = 'SEG'

SOP_CLASS_OBJ = {
    MODALITY_CT: '1.2.840.10008.5.1.4.1.1.2',
    MODALITY_PT: '1.2.840.10008.5.1.4.1.1.128', # [1.2.840.10008.5.1.4.1.1.128, 1.2.840.10008.5.1.4.1.1.130]
    MODALITY_SEG: '1.2.840.10008.5.1.4.1.1.66.4'
}
UID_TRANSFERSYNTAX  = '1.2.840.10008.1.2.1'

EXT_DCM  = '.dcm'
EXT_NRRD = '.nrrd'
EXT_NII  = '.nii.gz'
EXT_GZ   = '.gz'

def readNifti(pathNifti):
    data, header = None, None
    spacing, origin = None, None

    if Path(pathNifti).exists():
        nii_img      = nib.load(pathNifti)
        data, header = nii_img.get_fdata(), nii_img.header
        spacing      = header.get_zooms()
        origin       = header.get_qform()[:3,3]
    else:
        print (" - [readNifti()] Nifti file not found: ", pathNifti)
    
    return data, header, spacing, origin

def saveNifti(data, header, pathNifti):

    try:
        nii_img = nib.Nifti1Image(data, None, header)
        nib.save(nii_img, pathNifti)
    except:
        traceback.print_exc()
        pdb.set_trace()

def readNRRD(pathNRRD):

    data, header = None, None
    spacing, origin = None, None
    if Path(pathNRRD).exists():
        data, header = nrrd.read(pathNRRD)
        spacing      = tuple(np.diag(header["space directions"]))
        origin       = header.get("space origin", [0, 0, 0])
    else:
        print (" - [readNRRD()] NRRD file not found: ", pathNRRD)
    
    return data, header, spacing, origin

def readVolume(pathVolume):
    data, header = None, None

    if Path(pathVolume).suffix == EXT_NRRD:
        data, header, spacing, origin = readNRRD(pathVolume)
    elif Path(pathVolume).suffix == EXT_NII or Path(pathVolume).suffix == EXT_GZ:
        data, header, spacing, origin = readNifti(pathVolume)
    else:
        print (" - [readVolume()] Invalid file format: ", pathVolume)
    
    return data, header, spacing, origin

def getMaskWithLabel(maskArray, label):

    try:
        maskArray = maskArray.astype(np.uint8)
        maskArray[maskArray != label] = 0
        maskArray[maskArray == label] = 1
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return maskArray

def getDicomMeta(sopClassUID, sopInstanceUID):
    
    uid       = pydicom.uid.generate_uid()
    fileMeta = pydicom.dataset.FileMetaDataset()
    fileMeta.FileMetaInformationGroupLength = 254
    fileMeta.FileMetaInformationVersion     = b'\x00\x01'
    fileMeta.MediaStorageSOPClassUID        = sopClassUID
    fileMeta.MediaStorageSOPInstanceUID     = sopInstanceUID     
    fileMeta.TransferSyntaxUID              = UID_TRANSFERSYNTAX
    
    return fileMeta

def getBasicDicomDataset(patientName, studyUID, seriesUID, seriesNum, modality):

    # Step 1 - Create a basic dataset
    dataset = pydicom.dataset.Dataset()
    
    # Step 2 - Set the meta information
    dsMeta         = None
    sopInstanceUID = pydicom.uid.generate_uid()
    dsMeta         = getDicomMeta(SOP_CLASS_OBJ[modality], sopInstanceUID)
    dataset.file_meta = dsMeta

    # Step 3 - Patient Name    
    dataset.PatientName = patientName
    dataset.PatientID   = patientName

    # Step 4 - UIDs (useful when querying dicom servers)
    dataset.StudyInstanceUID  = studyUID
    dataset.SeriesInstanceUID = seriesUID
    dataset.SOPInstanceUID    = sopInstanceUID

    # Step 5 - Random tags for columns in the dicom file
    dataset.StudyDescription  = patientName + '-Study'
    dataset.SeriesDescription = patientName + '-Series-' + str(modality)
    dataset.SeriesNumber      = seriesNum
    dataset.ReferringPhysicianName = 'Dr. Mody :p'   

    # Step 6 - Other stuff
    dataset.is_little_endian = True
    dataset.SOPClassUID      = dsMeta.MediaStorageSOPClassUID
    dataset.Modality         = modality

    return dataset

def addCTPETDicomTags(ds, spacing, rows, cols):
    """
    From https://github.com/cornerstonejs/cornerstone3D/blob/v1.80.3/packages/core/src/utilities/generateVolumePropsFromImageIds.ts#L55
    const { BitsAllocated, PixelRepresentation, PhotometricInterpretation, ImageOrientationPatient, PixelSpacing, Columns, Rows, } = volumeMetadata;
    """

    try:
        # Step 1 - Position and Orientation
        ds.PatientPosition            = 'HFS'
        ds.ImageOrientationPatient    = [1, 0, 0, 0, 1, 0]
        ds.PositionReferenceIndicator = 'SN'
        ds.PhotometricInterpretation  = 'MONOCHROME2'

        # Step 2 - Pixel Data
        ds.Rows                       = rows
        ds.Columns                    = cols
        ds.PixelSpacing               = [float(spacing[0]), float(spacing[1])]
        ds.SliceThickness             = str(spacing[-1])

        # Step 3 - Pixel Datatype
        ds.BitsAllocated              = 16
        ds.BitsStored                 = 16
        ds.HighBit                    = 15
        ds.PixelRepresentation        = 1
        ds.SamplesPerPixel            = 1

        # Step 4 - Rescale
        ds.RescaleIntercept           = "0"
        ds.RescaleSlope               = "1"
        ds.RescaleType                = 'US' # US=Unspecified, HU=Hounsfield Units

        # Step 5 - Others
        ds.Manufacturer               = 'Hecktor2022-Cropped'

    except:
        traceback.print_exc()
        pdb.set_trace()

def makeCTPTDicomSlices(imageArray, origin, spacing, patientName, studyUID, seriesUID, seriesNum, pathFolder, modality, rotFunc):

    pixelValueList = []
    pathsList      = []

    try:
        # print ('')
        # print (modality)
        with tqdm.tqdm(total=imageArray.shape[0], leave=True, desc=' -- [makeCTPTDicomSlices({})]'.format(modality), disable=True) as pbarCT:
            for sliceIdx in range(imageArray.shape[-1]):
                
                # Step 1.0 - Create a basic dicom dataset
                dsCT = getBasicDicomDataset(patientName, studyUID, seriesUID, seriesNum, modality)
                addCTPETDicomTags(dsCT, spacing, imageArray.shape[0], imageArray.shape[1])
                
                # Step 1.1 - Set sliceIdx and origin
                dsCT.InstanceNumber       = str(sliceIdx+1)
                volOriginTmp              = list(copy.deepcopy(origin))
                volOriginTmp[-1]         += spacing[-1]*sliceIdx
                dsCT.ImagePositionPatient = volOriginTmp

                # Step 1.2 - Set pixel data
                pixelData        = rotFunc(imageArray[:,:,sliceIdx]) ## NOTE: helps to set right --> left orientation for .dcm files, refer: https://blog.redbrickai.com/blog-posts/introduction-to-dicom-coordinate
                if modality == MODALITY_CT:
                    pixelData = pixelData.astype(np.int16)
                if modality == MODALITY_PT:
                    pixelData = (pixelData * 1000).astype(np.int16)
                dsCT.PixelData = pixelData.tobytes()

                if sliceIdx == -1:
                    plt.imshow(pixelData, cmap='gray'); plt.title("{} Slice: {}".format(modality, sliceIdx))
                    plt.show(block=False)
                    pdb.set_trace()
                
                if modality == MODALITY_CT:
                    pixelValueList.extend(pixelData[(pixelData > -500) & (pixelData < 1200)].flatten().tolist())
                elif modality == MODALITY_PT:
                    pixelValueList.extend(pixelData[pixelData > 1.0].flatten().tolist())

                # Step 1.3 - Save the dicom file
                savePath = Path(pathFolder).joinpath(modality + '{:03d}'.format(sliceIdx) + '.' + str(dsCT.SOPInstanceUID) + EXT_DCM)
                dsCT.save_as(str(savePath), write_like_original=False)
                pathsList.append(savePath)

                pbarCT.update(1)
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return pixelValueList, pathsList

def set_segment_color(ds, segment_index, rgb_color):
    """
    Params
    ------
    ds: pydicom.dataset.FileDataset
        The DICOM file dataset
    segment_index: int
        The index of the segment to set the color for
    rgb_color: list
        The RGB color to set for the segment
    
    Notes
    -----
    >> ds.SegmentSequence[0]
    (0062, 0003)  Segmented Property Category Code Sequence  1 item(s) ----
    (0008, 0100) Code Value                          SH: '85756007'
    (0008, 0102) Coding Scheme Designator            SH: 'SCT'
    (0008, 0104) Code Meaning                        LO: 'Tissue'
    ---------
    (0062, 0004) Segment Number                      US: 1
    (0062, 0006) Segment Description                 ST: 'GTVp'
    (0062, 0008) Segment Algorithm Type              CS: 'AUTOMATIC'
    (0062, 0009) Segment Algorithm Name              LO: 'BaseAlgo'
    (0062, 000d) Recommended Display CIELab Value    US: [38563, 37582, 33812]
    (0062, 000f)  Segmented Property Type Code Sequence  1 item(s) ----
    (0008, 0100) Code Value                          SH: '85756007'
    (0008, 0102) Coding Scheme Designator            SH: 'SCT'
    ---------
    """
    try:
        def rgb_to_cielab(rgb):
            import skimage
            import skimage.color
            # Normalize RGB values to the range 0-1
            rgb_normalized = np.array(rgb) / 255.0
            # Convert RGB to CIELab
            cielab = skimage.color.rgb2lab(np.array([rgb_normalized]))
            return cielab.flatten()
        
        # Step 1 - Convert RGB to DICOM CIELab
        cielab = rgb_to_cielab(rgb_color)
        
        # Step 2 - DICOM CIELab values need to be scaled and converted to unsigned 16-bit integers
        L_star = int((cielab[0] / 100) * 65535)  # L* from 0 to 100
        a_star = int(((cielab[1] + 128) / 255) * 65535)  # a* from -128 to +127
        b_star = int(((cielab[2] + 128) / 255) * 65535)  # b* from -128 to +127
        # print ('   --> [set_segment_color()] rgb_color: {} --> cielab: {}'.format(rgb_color, [L_star, a_star, b_star]))
        
        # Step 3 - Set the color for the specified segment
        if 'SegmentSequence' in ds:
            ds.SegmentSequence[segment_index].RecommendedDisplayCIELabValue = [L_star, a_star, b_star]
        else:
            pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    # Save the modified DICOM file
    return ds

def makeSEGDicom(maskArray, maskSpacing, maskOrigin, metaInfoJsonPath, ctPathsList, patientName, maskType, studyUID, seriesNumber, contentCreatorName, pathFolderMask):

    try:

        # Step 1 - Convert to sitk image
        if not np.unique(maskArray).tolist() == [0, 1]:
            print (' - [makeSEGDicom] Unique values in maskArray: ', np.unique(maskArray).tolist())
            return
        maskImage = sitk.GetImageFromArray(np.moveaxis(maskArray, [0,1,2], [2,1,0]).astype(np.uint8)); print ("    -> [makeSEGDicom()] Doing np.moveaxis() for ", '-'.join([patientName, maskType])) # np([H,W,D]) -> np([D,W,H]) -> sitk([H,W,D]) 
        # maskImage = sitk.GetImageFromArray(np.moveaxis(maskArray, [0,1,2], [1,2,0]).astype(np.uint8)); print (" - Doing makeSEGDICOM's np.moveaxis() differently")
        maskImage.SetSpacing(maskSpacing)
        maskImage.SetOrigin(maskOrigin)
        # print (' - [maskImage] rows: {}, cols: {}, slices: {}'.format(maskImage.GetHeight(), maskImage.GetWidth(), maskImage.GetDepth())) ## SITK is (Width, Height, Depth)

        # Step 2 - Create a basic dicom dataset        ``
        import pydicom_seg
        template                    = pydicom_seg.template.from_dcmqi_metainfo(metaInfoJsonPath)
        template.SeriesDescription  = '-'.join([patientName, maskType])
        template.SeriesNumber       = seriesNumber
        template.ContentCreatorName = contentCreatorName
        # template.ContentLabel       = maskType
        writer                      = pydicom_seg.MultiClassWriter(template=template, inplane_cropping=False, skip_empty_slices=False, skip_missing_segment=False)
        ctDcmsList                  = [pydicom.dcmread(dcmPath, stop_before_pixels=True) for dcmPath in ctPathsList]
        dcm                         = writer.write(maskImage, ctDcmsList)
        # print (' - rows: {} | cols: {} | numberofframes:{}'.format(dcm.Rows, dcm.Columns, dcm.NumberOfFrames))
        
        # Step 3 - Save the dicom file
        if seriesNumber == 3:
            dcm = set_segment_color(dcm, 0, [0, 255, 0]) # GT
        elif seriesNumber == 4: 
            dcm = set_segment_color(dcm, 0, [255, 0, 0]) # Pred
        
        # print (dcm.SegmentSequence[0])
        dcm.StudyInstanceUID        = studyUID
        dcm.save_as(str(pathFolderMask / "mask.dcm"))

    except:
        traceback.print_exc()
        pdb.set_trace()

def terminalPlotHist(values, bins=100, titleStr="Histogram Plot"):

    try:
        
        pltTerm.hist(values, bins)
        pltTerm.title(titleStr + " (min: {} max: {})".format(np.min(values), np.max(values)))
        pltTerm.show()
        pltTerm.clf()

    except:
        traceback.print_exc()
        pdb.set_trace()

def studyDicomTags(ds): 
    """
    NOTES
    -----
    Modality=SEG
        - Per-frame Functional Groups Sequence: 
    """

    try:
        
        # Loop over dicom tags and print only the top-level ones
        for elem in ds:
            print (elem.name, elem.VR)
            if ds.Modality == 'SEG':
                if elem.name in ['Referenced Series Sequence', 'Segment Sequence', 'Shared Functional Groups Sequence']: #, 'Per-frame Functional Groups Sequence'] :
                    print (elem.name, elem.VR, elem.value)
            # if elem.VR != "SQ":
            #     print (elem.name, elem.VR)
            # else:
            #     print (elem.name, elem.VR, elem.value[0].name)
        
        pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()

def plot(ctArray, ptArray, maskArray, maskPredArray=None, sliceIds=[], patientName='', pathSavefigFolder=None):
    """
    Params:
        ctArray: [sagittal, coronal, axial]
        ptArray: [sagittal, coronal, axial]
        maskArray: [sagittal, coronal, axial]
        maskPredArray: [sagittal, coronal, axial]
    """
    
    # Keys - for colors
    COLORSTR_RED   = 'red'
    COLORSTR_GREEN = 'green'
    COLORSTR_PINK  = 'pink'
    COLORSTR_GRAY  = 'gray'

    # Define constants
    rotAxial    = lambda x: np.rot90(x, k=3)
    rotSagittal = lambda x: np.rot90(x, k=1)
    rotCoronal  = lambda x: np.rot90(x, k=1)

    totalSliceIdsToDisplay = 3

    try:
        
        if len(sliceIds) == 0:
            if maskArray is not None:
                gtPixelCountsInAxialSlices = np.sum(maskArray, axis=(0,1))
                randomSliceIds = np.random.choice(np.argwhere(gtPixelCountsInAxialSlices).flatten(), totalSliceIdsToDisplay)
            else:
                randomSliceIds = np.random.choice(ctArray.shape[2], totalSliceIdsToDisplay)
        else:
            randomSliceIds = sliceIds

        f,axarr = plt.subplots(3,len(randomSliceIds)*2, figsize=(15,15))

        axarr[0,0].set_ylabel('Axial')    # idx=2
        axarr[1,0].set_ylabel('Sagittal') # idx=0
        axarr[2,0].set_ylabel('Coronal')  # idx=1
        for i, sliceId in enumerate(randomSliceIds):
            
            # Plot Axial
            ctAxialSlice = rotAxial(ctArray[:,:,sliceId])
            ptAxialSlice = rotAxial(ptArray[:,:,sliceId])
            axarr[0,i*2].imshow(ctAxialSlice, cmap=COLORSTR_GRAY); axarr[0,i*2].set_title('SliceIdx: ' + str(sliceId))
            axarr[0,i*2].imshow(ptAxialSlice, cmap=COLORSTR_GRAY, alpha=0.3)
            axarr[0,i*2+1].imshow(ctAxialSlice, cmap=COLORSTR_GRAY); axarr[0,i*2+1].set_title('SliceIdx: ' + str(sliceId))
            axarr[0,i*2+1].imshow(ptAxialSlice, cmap=COLORSTR_GRAY, alpha=0.3)
            if maskArray is not None:
                maskAxialSlice = rotAxial(maskArray[:,:,sliceId])
                if np.max(maskAxialSlice) == 1:
                    axarr[0,i*2+1].contour(maskAxialSlice, colors=COLORSTR_GREEN)
            if maskPredArray is not None:
                maskAxialPredSlice = rotAxial(maskPredArray[:,:,sliceId])
                if np.max(maskAxialPredSlice) == 1:
                    axarr[0,i*2+1].contour(maskAxialPredSlice, colors=COLORSTR_RED)

            # Plot Sagittal
            ctSagittalSlice = rotSagittal(ctArray[sliceId,:,:])
            ptSagittalSlice = rotSagittal(ptArray[sliceId,:,:])
            axarr[1,i*2].imshow(ctSagittalSlice, cmap=COLORSTR_GRAY)
            axarr[1,i*2].imshow(ptSagittalSlice, cmap=COLORSTR_GRAY, alpha=0.3)
            axarr[1,i*2+1].imshow(ctSagittalSlice, cmap=COLORSTR_GRAY)
            axarr[1,i*2+1].imshow(ptSagittalSlice, cmap=COLORSTR_GRAY, alpha=0.3)
            if maskArray is not None:
                maskSagittalSlice = rotSagittal(maskArray[sliceId,:,:])
                if np.max(maskSagittalSlice) == 1:
                    axarr[1,i*2+1].contour(maskSagittalSlice, colors=COLORSTR_GREEN)
            if maskPredArray is not None:
                maskPredSagittallSlice = rotSagittal(maskPredArray[sliceId,:,:])
                if np.max(maskPredSagittallSlice) == 1:
                    axarr[1,i*2+1].contour(maskPredSagittallSlice, colors=COLORSTR_RED)

            # Plot Coronal
            ctCoronalSlice = rotCoronal(ctArray[:,sliceId,:])
            ptCoronalSlice = rotCoronal(ptArray[:,sliceId,:])
            axarr[2,i*2].imshow(ctCoronalSlice, cmap=COLORSTR_GRAY)
            axarr[2,i*2].imshow(ptCoronalSlice, cmap=COLORSTR_GRAY, alpha=0.3)
            axarr[2,i*2+1].imshow(ctCoronalSlice, cmap=COLORSTR_GRAY)
            axarr[2,i*2+1].imshow(ptCoronalSlice, cmap=COLORSTR_GRAY, alpha=0.3)
            if maskArray is not None:
                maskCoronalSlice = rotCoronal(maskArray[:,sliceId,:])
                if np.max(maskCoronalSlice) == 1:
                    axarr[2,i*2+1].contour(maskCoronalSlice, colors=COLORSTR_GREEN)
            if maskPredArray is not None:
                maskCoronalPredSlice = rotCoronal(maskPredArray[:,sliceId,:])
                if np.max(maskCoronalPredSlice) == 1:
                    axarr[2,i*2+1].contour(maskCoronalPredSlice, colors=COLORSTR_RED)
        
        pltTitleStr = 'Raw Array Plot (no .dcm stuff involved so far) \n' + patientName
        plt.suptitle(pltTitleStr)
        if pathSavefigFolder is None:
            plt.show()
        else:
            Path(pathSavefigFolder).mkdir(parents=True, exist_ok=True)
            plt.savefig(pathSavefigFolder / "{}.png".format(patientName))
            plt.close()

    except:
        traceback.print_exc()
        pdb.set_trace()

def doDeformMask(patientName, pathPatientMaskGT, pathPatientMaskPredDeformed, maskClassId, warpMagnitude):

    try:
        if Path(pathPatientMaskGT).exists():
            
            # Step 1 - Read
            maskArray, maskHeader, maskSpacing, maskOrigin = readVolume(pathPatientMaskGT)
            maskArray = getMaskWithLabel(maskArray, maskClassId)

            uniqueValsMask = np.unique(maskArray).tolist()
            if uniqueValsMask != [0,1]:
                print ('   - [doDeformMask()] Issue with patientName: ', patientName)
                print ('   - [doDeformMask()] Unique values in maskArray: ', uniqueValsMask)
                return
            
            # Step 2 - Deform
            maskArray = torch.tensor(maskArray).unsqueeze(0).unsqueeze(0) # [H,W,D] --> [1,1,H,W,D]
            maskArray = doRandomElasticDeform(maskArray, warpMagnitudeMin=warpMagnitude, warpMagnitudeMax=warpMagnitude, verbose=True)
            maskArray = maskArray.squeeze(0).squeeze(0).numpy()
            
            # Step 3 - Save
            # pathMaskPred = DIR_CLINIC / (patientName + '_maskpred.nii.gz')
            print ('    - [doDeformMask()] Saving to: ', pathPatientMaskPredDeformed)
            saveNifti(maskArray, maskHeader, pathPatientMaskPredDeformed)

        else:
            print (' - [doDeformMask()] Path does not exist: ', pathPatientMaskGT)

    except:
        traceback.print_exc()
        pdb.set_trace()

def doSmoothMask(patientName, pathPatientOGMask, pathPatientNewMask, maskClassId, sigma):

    try:

        # Step 0 - Init        
        import scipy.ndimage

        # Step 1 - Read 
        maskArrayOG, maskHeader, maskSpacing, maskOrigin = readVolume(pathPatientOGMask)
        maskArrayOG = getMaskWithLabel(maskArrayOG, maskClassId)
        
        # Step 2 - Smooth
        maskArrayOGSmooth = scipy.ndimage.gaussian_filter(maskArrayOG.astype(float), sigma=sigma)
        # maskArrayOGSmooth = (maskArrayOGSmooth > 0.5).astype(np.uint8)
        
        # Plot
        if 0:
            sliceId = np.random.choice(np.argwhere(np.sum(maskArrayOG, axis=(0,1))).flatten())
            # sliceId = 60 # CHMR005=[51, 57, 60 66]
            sliceId = 56 # CHMR001=[50, 57]
            if 0:
                f,axarr = plt.subplots(1,2)
                axarr[0].imshow(maskArrayOG[:,:,sliceId], cmap='gray')
                axarr[1].imshow(maskArrayOGSmooth[:,:,sliceId], cmap='gray')
                plt.title(patientName + ' | Slice: ' + str(sliceId))
            elif 1:
                sliceNeighbourCount = 3
                f,axarr = plt.subplots(2, sliceNeighbourCount*2 + 1, figsize=(15,5))
                for axarrId, sliceId_ in enumerate(range(sliceId - sliceNeighbourCount, sliceId + sliceNeighbourCount + 1)):
                    axarr[0][axarrId].imshow(maskArrayOG[:,:,sliceId_], cmap='gray')
                    axarr[0][axarrId].contour(maskArrayOG[:,:,sliceId_], levels=[0.5, 0.9], colors='green')
                    axarr[0][axarrId].contour(maskArrayOGSmooth[:,:,sliceId_], levels=[0.05, 0.1, 0.2], colors='pink')
                    axarr[0][axarrId].set_title('Slice: ' + str(sliceId_))
                    axarr[1][axarrId].imshow(maskArrayOGSmooth[:,:,sliceId_], cmap='gray')
                    axarr[1][axarrId].contour(maskArrayOGSmooth[:,:,sliceId_], levels=[0.05, 0.1, 0.2], colors='pink')
                plt.suptitle(patientName + ' | Slice: ' + str(sliceId) + '\n sigma: ' + str(sigma) + ' | contour levels=[0.05, 0.1, 0.2]')
            plt.show(block=False)
            pdb.set_trace()
        
        # Make a folder for filtered masks (and copy other data too)
        maskArrayOGSmooth[maskArrayOGSmooth < 0.2] = 0
        maskArrayOGSmooth[maskArrayOGSmooth > 0.2] = 1
        maskArrayOGSmooth = maskArrayOGSmooth.astype(np.uint8)
        saveNifti(maskArrayOGSmooth, maskHeader, pathPatientNewMask)
        print ('    -> [doSmoothMask()] Saved to: {} / {}'.format(*Path(pathPatientNewMask).parts[-2:]))

    except:
        traceback.print_exc()
        pdb.set_trace()

class DICOMConverterHecktor:

    def __init__(self, patientName, pathCT, pathPT, pathMask, pathMaskPred, maskClassId
                 , rotFunc, maskMetaInfoPath
                , maskGTSeriesDescSuffix, maskPredSeriesDescSuffix, maskGTCreatorName, maskPredCreatorName):
        
        # Step 1 - Basic info and paths
        self.patientName  = patientName
        self.pathCT       = pathCT
        self.pathPT       = pathPT
        self.pathMask     = pathMask
        self.pathMaskPred = pathMaskPred
        self.maskClassId  = int(maskClassId)

        # Step 2 - Additional info 
        self.rotFunc          = rotFunc
        self.maskMetaInfoPath = maskMetaInfoPath
        self.maskGTSeriesDescSuffix   = maskGTSeriesDescSuffix
        self.maskPredSeriesDescSuffix = maskPredSeriesDescSuffix
        self.maskGTCreatorName        = maskGTCreatorName
        self.maskPredCreatorName      = maskPredCreatorName

        
        self._readFiles()

    def _readFiles(self):
        
        try:

            # Step 1 - Read NRRD files
            self.ctArray, self.ctHeader, self.ctSpacing, self.ctOrigin = readVolume(self.pathCT)
            self.ptArray, self.ptHeader, self.ptSpacing, self.ptOrigin = readVolume(self.pathPT)
            self.maskArray, self.maskHeader, self.maskSpacing, self.maskOrigin                 = readVolume(self.pathMask)
            self.maskPredArray, self.maskPredHeader, self.maskPredSpacing, self.maskPredOrigin = readVolume(self.pathMaskPred)
            
            # Step 2 - Check for self.maskClassId
            self.maskArray     = getMaskWithLabel(self.maskArray, self.maskClassId)
            self.maskPredArray = getMaskWithLabel(self.maskPredArray, self.maskClassId)
            if np.unique(self.maskArray).tolist() != [0, 1]:
                print ('    -> [DICOMConverterHecktor._readFiles()] Unique values in maskArray: ', np.unique(self.maskArray).tolist())
                self.doConvertToDICOM = False
            else:
                self.doConvertToDICOM = True
            
            if self.doConvertToDICOM:
                # Step 1.1 - Some custom processing
                if 1:
                    if self.ctArray.shape[-1] == 145:
                        self.ctArray  = self.ctArray[:,:,:-1]
                        self.ptArray = self.ptArray[:,:,:-1]
                        self.maskArray = self.maskArray[:,:,:-1]
                    print ('    -> [DICOMConverterHecktor._readFiles()] CT: {}, PT: {}, Mask: {}, MaskPred: {}'.format(self.ctArray.shape, self.ptArray.shape, self.maskArray.shape, self.maskPredArray.shape))
                # assert self.ctArray.shape == self.ptArray.shape == self.maskArray.shape == self.maskPredArray.shape, " - [DICOMConverterHecktor] Shape mismatch: CT: {}, PET: {}, Mask: {}, MaskPred: {}".format(self.ctArray.shape, self.ptArray.shape, self.maskArray.shape, self.maskPredArray.shape)

                # Step 1.2 - Check Spacing
                floatify = lambda x: [float(i) for i in x] 
                self.ctSpacing = floatify(self.ctSpacing)
                self.ptSpacing = floatify(self.ptSpacing)
                self.maskSpacing = floatify(self.maskSpacing)
                self.maskPredSpacing = floatify(self.maskPredSpacing)
                assert self.ctSpacing == self.ptSpacing == self.maskSpacing == self.maskPredSpacing, " - [DICOMConverterHecktor] Spacing mismatch: CT: {}, PET: {}, Mask: {}, MaskPred: {}".format(self.ctSpacing, self.ptSpacing, self.maskSpacing, self.maskPredSpacing)

                # Step 3 - Get origins
                self.ctOrigin = [0, 0, 0]
                self.ptOrigin = [0, 0, 0]
                self.maskOrigin = [0, 0, 0]
                self.maskPredOrigin = [0, 0, 0]
                print ('    -> [DICOMConverterHecktor._readFiles()] Setting all origins to (0,0,0)')

                # Step 4 - Create folders and make UIDs
                self._createFolders()

                # Step 5 - Plot
                if 1:
                    plot(self.ctArray, self.ptArray, self.maskArray, self.maskPredArray, patientName=self.patientName, pathSavefigFolder=self.pathParentFolder / self.patientName)
                    # pdb.set_trace()
        except:
            traceback.print_exc()
            pdb.set_trace()

    def _createFolders(self):
        
        self.pathParentFolder   = Path(self.pathCT).parent.absolute()
        self.pathFolderCT       = self.pathParentFolder / self.patientName / "CT"
        self.pathFolderPT       = self.pathParentFolder / self.patientName / "PT"
        self.pathFolderMask     = self.pathParentFolder / self.patientName / "Mask"
        self.pathFolderMaskPred = self.pathParentFolder / self.patientName / "MaskPred"

        if Path(self.pathFolderCT).exists():
            shutil.rmtree(self.pathFolderCT)
        if Path(self.pathFolderPT).exists():
            shutil.rmtree(self.pathFolderPT)
        if Path(self.pathFolderMask).exists():
            shutil.rmtree(self.pathFolderMask)
        if Path(self.pathFolderMaskPred).exists():
            shutil.rmtree(self.pathFolderMaskPred)

        Path(self.pathFolderCT).mkdir(parents=True, exist_ok=True)
        Path(self.pathFolderPT).mkdir(parents=True, exist_ok=True)
        Path(self.pathFolderMask).mkdir(parents=True, exist_ok=True)
        Path(self.pathFolderMaskPred).mkdir(parents=True, exist_ok=True)

    def convertToDICOM(self):
        
        try:

            # Step 0 - Make commons UIDs
            if not self.doConvertToDICOM:
                print ('    - [ERROR][DICOMConverterHecktor.convertToDICOM()] Skipping volume to DICOM conversion ...')
                return
            self.studyUID = pydicom.uid.generate_uid()

            # Step 1 - Convert CT
            # print ('\n - [convertToDICOM()] rotFunc: ', inspect.getsource(self.rotFunc))
            if 1:
                ctSeriesUID = pydicom.uid.generate_uid()
                ctSeriesNum = 1
                ctPixelValueList, ctPathsList = makeCTPTDicomSlices(self.ctArray, self.ctOrigin, self.ctSpacing, self.patientName, self.studyUID, ctSeriesUID, ctSeriesNum, self.pathFolderCT, MODALITY_CT, self.rotFunc)
                # terminalPlotHist(ctPixelValueList, bins=100, titleStr="CT Histogram Plot")

            # Step 2 - Convert PT
            if 1:
                ptSeriesUID = pydicom.uid.generate_uid()
                ptSeriesNum = 2
                ptPixelValueList = makeCTPTDicomSlices(self.ptArray, self.ptOrigin, self.ptSpacing, self.patientName, self.studyUID, ptSeriesUID, ptSeriesNum, self.pathFolderPT, MODALITY_PT, self.rotFunc)
                # terminalPlotHist(ptPixelValueList, bins=100, titleStr="PT Histogram Plot")

            # Step 3 - Convert Mask
            if 1:

                # Step 3.1 - Mask (GT)
                # print ('    - [INFO][DICOMConverterHecktor.convertToDICOM()] Making SEG Dicom for GT Masks...')
                seriesNumber            = 3
                makeSEGDicom(self.maskArray, self.maskSpacing, self.maskOrigin, self.maskMetaInfoPath, ctPathsList
                             , self.patientName,  self.maskGTSeriesDescSuffix, self.studyUID, seriesNumber, self.maskGTCreatorName, self.pathFolderMask)
                
                # Step 3.2 - Mask (Pred)
                # print ('    - [INFO][DICOMConverterHecktor.convertToDICOM()] Making SEG Dicom for Predicted Masks...')
                seriesNumber            = 4
                makeSEGDicom(self.maskPredArray, self.maskPredSpacing, self.maskPredOrigin, self.maskMetaInfoPath, ctPathsList
                             , self.patientName,  self.maskPredSeriesDescSuffix, self.studyUID, seriesNumber, self.maskPredCreatorName, self.pathFolderMaskPred)
                

        except:
            traceback.print_exc()
            pdb.set_trace() 

def doRandomElasticDeform(mask: torch.tensor, warpMagnitudeMin:int=1, warpMagnitudeMax:int=4, verbose=False) -> torch.tensor:
    """
    Tip: Use this function to take a ground truth mask, and generate a (random) predicted mask.

    Parameters:
    -----------
        mask (torch.Tensor): Binary mask tensor containing 1s and 0s, [B,1,H,W,D]
    
    Returns:
    --------
        maskDeformed (torch.Tensor): Deformed mask, [B,1,H,W,D]
    """

    maskDeformed = []
    warpMagnitudeList = []

    try:
        
        # Step 0 - Ensure dimensionality
        assert len(mask.shape) == 5, f" - [randomElasticDeformImage()] Invalid mask shape {mask.shape}, make sure it is [B,1,H,W,D]"

        # Step 1 - Get the deformedMask
        for batchId in range(mask.shape[0]):
            thisMask = mask[batchId,0]

            # Step 2.1 - Get deformation
            warpMagnitude = random.randint(warpMagnitudeMin, warpMagnitudeMax)
            warpMagnitudeList.append(warpMagnitude)
            deformField3D = voxynthTransform.random_transform(shape=thisMask.shape, affine_probability=0, warp_probability=1, warp_integrations=0
                                        , warp_smoothing_range=(10,16), warp_magnitude_range=(warpMagnitude,warpMagnitude), isdisp=False)  # [H,W,D,3]
            
            # Step 2.2 - Apply deformation
            thisMaskWarped = voxynthTransform.spatial_transform(thisMask.float().unsqueeze(0), trf = deformField3D, isdisp=False) # [1,H,W,D]
            thisMaskWarped = thisMaskWarped.round().float()
            maskDeformed.append(thisMaskWarped)
        
        maskDeformed = torch.cat(maskDeformed, dim=0).unsqueeze(0) # [B,1,H,W,D]
        if verbose:
            print (f" - [randomElasticDeformImage()] Warp Magnitudes: {warpMagnitudeList}")

        # Step 99 - Plot (to debug)
        if 0:
            rows = 1 + mask.shape[0]
            f,axarr = plt.subplots(rows,2)
            for batchId in range(mask.shape[0]):
                sliceId = random.choice(torch.argwhere(thisMask.sum(axis=(0,1)) > 0))[0]
                axarr[batchId,0].imshow(mask[batchId,0,:,:,sliceId].cpu().numpy(), cmap='gray')
                axarr[batchId,1].imshow(maskDeformed[batchId,0,:,:,sliceId].cpu().numpy(), cmap='gray')
                axarr[batchId,0].set_ylabel(f"WarpMagnitude: {warpMagnitudeList[batchId]}")
                if batchId == 0:
                    axarr[batchId,0].set_title("GT Mask")
                    axarr[batchId,1].set_title("Deformed Mask")
            
            plt.show(block=False)
            pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return maskDeformed

############################################################
# Main
############################################################ 

if __name__ == "__main__":

    ########################## Step 0.1 - Init (dirs) 
    DIR_BACKUTILS = Path(__file__).parent.absolute() # <root>/src/backend/utils/
    DIR_SRC       = DIR_BACKUTILS.parent.parent      # <root>/src/
    DIR_MAIN      = DIR_SRC.parent                   # <root>/
    DIR_PREMAIN   = DIR_MAIN.parent                  # ../<root>
    DIR_ASSETS    = DIR_SRC / "assets"               # <root>/assets/    
    sys.path.append(str(DIR_BACKUTILS))

    if 0:
        DIR_DATA = DIR_MAIN.parent.absolute() / "_data"   # <pre-root>/_data/
    elif 1:
        DIR_DATA = DIR_MAIN.parent.absolute() / "_data"
    print ('\n - [__main__] DIR_DATA: ', DIR_DATA, DIR_DATA.exists(), '\n')

    ########################## Step 0.2 - Init (constants) 
    maskClassId = 1 # [1,2]
    filterSigma = 2
    rotFunc                  = lambda x: np.fliplr(np.rot90(x, k=3)) ## NOTE: helps to set right --> left orientation for .dcm files, refer: https://blog.redbrickai.com/blog-posts/introduction-to-dicom-coordinate
    # rotFunc                = lambda x:x
    maskMetaInfoPath         = DIR_ASSETS / 'metainfo-segmentation.json'
    maskGTSeriesDescSuffix   = 'Series-SEG-GT'
    maskPredSeriesDescSuffix = 'Series-SEG-Pred'
    maskGTCreatorName        = 'Hecktor2022'
    maskPredCreatorName      = 'Modys AI model'

    ########################## Step 0.3 - Init (vars)
    if 0:
        DIR_CLINIC           = DIR_DATA / "Hecktor2022" / "trial6-2022-CHUP"
        patientNames         = ['CHUP-000', 'CHUP-001', 'CHUP-002', 'CHUP-004', 'CHUP-005', 'CHUP-006', 'CHUP-007', 'CHUP-008', 'CHUP-009', 'CHUP-010', 'CHUP-011', 'CHUP-012', 'CHUP-013', 'CHUP-015', 'CHUP-016', 'CHUP-017', 'CHUP-018', 'CHUP-019', 'CHUP-020', 'CHUP-022', 'CHUP-023', 'CHUP-024', 'CHUP-025', 'CHUP-026', 'CHUP-027', 'CHUP-028', 'CHUP-029', 'CHUP-030', 'CHUP-032', 'CHUP-033', 'CHUP-034', 'CHUP-035', 'CHUP-036', 'CHUP-038', 'CHUP-039', 'CHUP-041', 'CHUP-042', 'CHUP-043', 'CHUP-044', 'CHUP-046', 'CHUP-047', 'CHUP-048', 'CHUP-049', 'CHUP-050', 'CHUP-051', 'CHUP-052', 'CHUP-053', 'CHUP-054', 'CHUP-055', 'CHUP-056', 'CHUP-057', 'CHUP-058', 'CHUP-059', 'CHUP-060', 'CHUP-061', 'CHUP-062', 'CHUP-063', 'CHUP-064', 'CHUP-065', 'CHUP-066', 'CHUP-067', 'CHUP-068', 'CHUP-069', 'CHUP-070', 'CHUP-071', 'CHUP-072', 'CHUP-073', 'CHUP-075']
        
        suffixCT              = "_img1.nii.gz"
        suffixPT              = "_img2.nii.gz"
        suffixMaskGT          = '_mask.nii.gz'
        suffixMaskGTFiltered  = '_mask-filtered.nii.gz'
        suffixMaskGTDeformed  = '_mask-deformed-warp{}.nii.gz'
        suffixPatientDeformed = '-gt-deformed-warp{}'
        suffixPatientFiltered = '-gt-filtered-gausssig{}'

        pathCT              = DIR_CLINIC / ('{}' + suffixCT)
        pathPT              = DIR_CLINIC / ('{}' + suffixPT)
        pathMaskGT          = DIR_CLINIC / ('{}' + suffixMaskGT)
        pathMaskGTFiltered  = DIR_CLINIC / ('{}' + suffixMaskGTFiltered)
        pathMaskGTDeformed  = DIR_CLINIC / ('{}' + suffixMaskGTDeformed)
        
        predPresent          = False
        doGTFilter           = True
        doDeform             = True
        warpMagnitude        = 4
    
    elif 1:

        if 0:
            DIR_CLINIC          = DIR_DATA / "Hecktor2021" / "trial2-CHMR"
            patientNames        = ['CHMR001', 'CHMR004', 'CHMR005', 'CHMR011', 'CHMR012', 'CHMR013', 'CHMR014', 'CHMR016', 'CHMR020', 'CHMR021', 'CHMR023', 'CHMR024', 'CHMR025', 'CHMR028', 'CHMR029', 'CHMR030', 'CHMR034', 'CHMR040']
        elif 1:
            DIR_CLINIC           = DIR_DATA / "Hecktor2022" / "trial6-2022-CHUP-withModelPredictions"
            patientNames         = ['CHUP-000', 'CHUP-001', 'CHUP-002', 'CHUP-004', 'CHUP-005', 'CHUP-006', 'CHUP-007', 'CHUP-008', 'CHUP-009', 'CHUP-010', 'CHUP-011', 'CHUP-012', 'CHUP-013', 'CHUP-015', 'CHUP-016', 'CHUP-017', 'CHUP-018', 'CHUP-019', 'CHUP-020', 'CHUP-022', 'CHUP-023', 'CHUP-024', 'CHUP-025', 'CHUP-026', 'CHUP-027', 'CHUP-028', 'CHUP-029', 'CHUP-030', 'CHUP-032', 'CHUP-033', 'CHUP-034', 'CHUP-035', 'CHUP-036', 'CHUP-038', 'CHUP-039', 'CHUP-041', 'CHUP-042', 'CHUP-043', 'CHUP-044', 'CHUP-046', 'CHUP-047', 'CHUP-048', 'CHUP-049', 'CHUP-050', 'CHUP-051', 'CHUP-052', 'CHUP-053', 'CHUP-054', 'CHUP-055', 'CHUP-056', 'CHUP-057', 'CHUP-058', 'CHUP-059', 'CHUP-060', 'CHUP-061', 'CHUP-062', 'CHUP-063', 'CHUP-064', 'CHUP-065', 'CHUP-066', 'CHUP-067', 'CHUP-068', 'CHUP-069', 'CHUP-070', 'CHUP-071', 'CHUP-072', 'CHUP-073', 'CHUP-075']
        
        suffixCT              = '_ct.nii.gz'
        suffixPT              = '_pt.nii.gz'
        suffixMaskGT          = '_gtvt.nii.gz'
        suffixMaskGTFiltered  = '_gtvt-filtered-gaussiansigma{}.nii.gz'
        suffixMaskPred        = "nrrd_{}_maskpred.nrrd"
        suffixPatientFiltered = '-gt-filtered-gausssig{}'

        pathCT               = DIR_CLINIC / ('{}' + suffixCT)
        pathPT               = DIR_CLINIC / ('{}' + suffixPT)
        pathMaskGT           = DIR_CLINIC / ('{}' + suffixMaskGT)
        pathMaskGTFiltered   = DIR_CLINIC / ('{}' + suffixMaskGTFiltered)
        pathMaskPred         = DIR_CLINIC / suffixMaskPred

        doDeform             = False
        doGTFilter           = True
        predPresent          = True

    if doDeform:
        DIR_VOXYNTH = DIR_BACKUTILS / "voxynth" # <root>/src/backend/utils/
        sys.path.append(str(DIR_VOXYNTH))
        import voxynth.transform as voxynthTransform

    ##########################
    # Patient loops
    ##########################
    if 1:
        print ('\n - [__main__] Starting DICOM conversion for patients ...')
        print (' - [__main__] DIR_CLINIC: ', DIR_CLINIC, '')
        for patientName in patientNames:
            print (f' \n ---------------------------------- : patientName: {patientName} \n')
            if patientName not in ['CHUP-044']:
                continue

            try:

                # Step 0 - Paths
                pathCTThis       = Path(str(pathCT).format(patientName))
                pathPTThis       = Path(str(pathPT).format(patientName))
                pathMaskGTThis   = Path(str(pathMaskGT).format(patientName))
                if predPresent:
                    pathMaskPredThis = Path(str(pathMaskPred).format(patientName))
                
                # Step 1 - Do GT filtering (if required)
                if doGTFilter:
                    print ('   - [Preprocess] Filtering GT for patient: ', patientName)
                    # Step 1.1 - Filter GT
                    pathMaskGTFilteredThis = Path(str(pathMaskGTFiltered).format(patientName, filterSigma))
                    doSmoothMask(patientName, pathMaskGTThis, pathMaskGTFilteredThis, maskClassId, sigma=filterSigma)
                
                    # Step 1.2 - Update variables (names + paths)
                    patientName = patientName + suffixPatientFiltered.format(filterSigma)
                    pathMaskGTThis = pathMaskGTFilteredThis
                    maskGTSeriesDescSuffix = maskGTSeriesDescSuffix # + suffixPatientFiltered.format(filterSigma)

                # Step 2 - Do deformations (if required)
                if doDeform:
                    print ('   - [Preprocess] Deforming GT for patient (for pred): ', patientName)
                    # Step 2.1 - Deform GT
                    pathMaskPredDeformed = Path(str(pathMaskGTDeformed).format(patientName, warpMagnitude))
                    doDeformMask(patientName, pathMaskGTThis, pathMaskPredDeformed, maskClassId, warpMagnitude=4)
                    
                    # Step 2.2 - Update variables (names + paths)
                    patientName = patientName + suffixPatientDeformed.format(warpMagnitude)
                    pathMaskPredThis = pathMaskPredDeformed
                    maskPredSeriesDescSuffix = maskPredSeriesDescSuffix #+ suffixPatientDeformed.format(warpMagnitude)
                

                # Step 99 - Convert to dicom
                print ('   - [Final] Converting to DICOM for patient: ', patientName)
                converterClass = DICOMConverterHecktor(patientName, pathCTThis, pathPTThis, pathMaskGTThis, pathMaskPredThis, maskClassId
                                                    , rotFunc, maskMetaInfoPath
                                                    , maskGTSeriesDescSuffix, maskPredSeriesDescSuffix, maskGTCreatorName, maskPredCreatorName)
                converterClass.convertToDICOM()
                print ('\n  - [Final] Done for patient: {}/{}'.format(*Path(converterClass.pathFolderCT.parent).parts[-2:]))
                pdb.set_trace()
        
            except:
                traceback.print_exc()
                pdb.set_trace()


"""
http://34.147.125.69:8042/ui/app/index.html#/
"""

"""
OHIF in Orthanc
 - ReferencedSeriesSequence is missing for the SEG 
 - for elem in ds: print (elem.name, elem.VR) if elem.VR == "SQ": pass

SEG modality in DICOM
 - np,moveaxis([0,1,2], [2,1,0]) shows it correctly in 3D Slicer and myC3D app
 - does not show correctly in matplotlib. Makes me question whether torch is getting the right arrays.
"""