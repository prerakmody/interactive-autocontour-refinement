"""
Plots outputs for AI-pencil sessions

Takes as input 
 - expID
 - viewType, sliceId, refineId
   - op => <refineId-viewType>.dcm[sliceId-1: sliceId+1]
   - op => <refineId-viewType>.png[sliceId-1: sliceId+1]
 - patientId
   - CT scan[sliceId-1: sliceId+1]
   - PET Scan[sliceId-1: sliceId+1]
   - GT[sliceId-1: sliceId+1]

"""

# import public libs
import pdb
import time
import pydicom
import traceback
import matplotlib
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Import private libs
import orthancRequestUtils

KEY_AXIAL    = 'Axial'
KEY_SAGITTAL = 'Sagittal'
KEY_CORONAL  = 'Coronal'

rotAxial    = lambda x: x
rotSagittal = lambda x: np.rot90(x, k=1)
rotCoronal  = lambda x: np.rot90(x, k=1)

KEY_REFINE      = 'Refine'
KEY_INTERACTION = 'interaction'
SERIESDESC_SUFFIX_REFINE     = 'Series-SEG-Refine'

EXT_DCM = '.dcm'
EXT_PNG = '.png'

fileNameForScribble    = lambda patientId, counter, viewType, sliceId: '-'.join([str(patientId), SERIESDESC_SUFFIX_REFINE, '{:03d}'.format(counter), viewType, 'slice{:03d}'.format(sliceId), KEY_INTERACTION]) + EXT_PNG
fileNameOldForScribble = lambda patientId, counter, viewType, sliceId: '-'.join([str(patientId), SERIESDESC_SUFFIX_REFINE, '{}'.format(counter), viewType, 'slice{:03d}'.format(sliceId), KEY_INTERACTION]) + EXT_PNG

fileNameForDCM    = lambda patientId, counter: '-'.join([str(patientId), KEY_REFINE, '{:03d}'.format(counter)]) + EXT_DCM
fileNameOldForDCM = lambda patientId, counter: '-'.join([str(patientId), KEY_REFINE, '{}'.format(counter)]) + EXT_DCM

DIR_FILE = Path(__file__).resolve().parent # <root>/src/backend/utils
DIR_ROOT = DIR_FILE.parent.parent.parent # <root>
DIR_EXPERIMENTS         = DIR_ROOT / '_experiments'
DIR_EXPERIMENTS_OUTPUTS = DIR_EXPERIMENTS / 'experiment-outputs'

CMAP_DEFAULT      = plt.cm.Oranges
RGBA_ARRAY_BLUE   = np.array([0   ,0 ,255,255])/255.
RGBA_ARRAY_YELLOW = np.array([218,165,32 ,255])/255.

KEY_SCRIBBLE_BGD = 'bgd'
KEY_SCRIBBLE_FGD = 'fgd'

HU_MIN, HU_MAX         = -250, 250
SUV_MIN_v1, SUV_MAX_v1 = 0   ,25000

COLOR_GRAY = 'gray'
COLOR_GREEN = 'green'
COLOR_RED = 'red'
COLOR_PINK = 'pink'

LINESTYLE_REFINE = 'dashed'

#################################################################
#                        PLOTTING UTILS
#################################################################

def getScribbleColorMap(cmap, opacityBoolForScribblePoints):
    """
    Creates a new colormap with modified opacity settings
    Params
    ------
    cmap: matplotlib.colors.Colormap object; takes as input values in the range: [0,255] and gives an RGBA output tuple with vals in the range of [0,1]
    opacityBoolForScribblePoints: bool
    """
    cmapNew, normNew = None, None
    
    try:
        
        # Step 1 - Get colors
        import matplotlib.colors
        colors = cmap(np.arange(cmap.N)) # cmap accepts values in the range: [0,256]

        # Step 2.1 - Set opacity
        colors[:,-1] = np.linspace(0, 1, cmap.N)

        # Step 2.2 - Set opacity to 0 for all colors, except the last one
        if opacityBoolForScribblePoints:
            colors[:,-1][:-1] = 0 # set opacity to 0 for all colors, except the last one
        
        # Step 3 - Create new colormap
        cmapNew = matplotlib.colors.ListedColormap(colors)

        # Step 4 - Normalize
        normNew = matplotlib.colors.BoundaryNorm(np.linspace(0, 1, cmap.N), cmap.N, clip=True)

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return cmapNew, normNew

def getSliceForScanArrayByViewTypeAndSliceId(array3D, viewType, sliceId, centerX=None, centerY=None):
    """
    Returns a slice from array3D based on viewType and sliceId
    array3D: (coronal, sagittal, axial)
    viewType: 'coronal', 'sagittal', 'axial'
    sliceId: int
    centerX, centerY: int
    BBOX_SIZE: (int, int)
    """

    # Step 0 - Init
    array2D = None 

    # Step 1 - Get slice
    if viewType == KEY_AXIAL:
        array2D = rotAxial(array3D[:,:,sliceId])
        
    elif viewType == KEY_SAGITTAL:
        array2D = rotSagittal(array3D[:, sliceId, :])
        
    elif viewType == KEY_CORONAL:
        array2D = rotCoronal(array3D[sliceId, :, :])
    
    # Step 2 - Crop as per BBOX_SIZE and center
    if array2D is not None and centerX is not None and centerY is not None:
        array2D = array2D[centerX-BBOX_SIZE[0]//2:centerX+BBOX_SIZE[0]//2, centerY-BBOX_SIZE[1]//2:centerY+BBOX_SIZE[1]//2]

    return array2D

def getSliceForMaskArrayByViewTypeAndSliceId(array3D, viewType, sliceId, centerX=None, centerY=None):
    """
    Returns a slice from array3D based on viewType and sliceId
    array3D: (axial, coronal, sagittal)
    """

    # Step 0 - Init
    array2D = None
    
    # Step 1 - Get slice
    if viewType == KEY_AXIAL:
        array2D = array3D[sliceId,:,:]
        
    elif viewType == KEY_SAGITTAL:
        array2D = np.flipud(array3D[:, :, sliceId]) # ??
        
    elif viewType == KEY_CORONAL:
        array2D = np.flipud(array3D[:, sliceId, :]) # ??
    
    # Step 2 - Crop as per BBOX_SIZE and center
    if array2D is not None and centerX is not None and centerY is not None:
        array2D = array2D[centerX-BBOX_SIZE[0]//2:centerX+BBOX_SIZE[0]//2, centerY-BBOX_SIZE[1]//2:centerY+BBOX_SIZE[1]//2]
    
    return array2D

#################################################################
#                            MAIN
#################################################################

def plot(prefix, expId, patientId, counter, viewType, sliceId, scribbleType):
    """
    ?
    """

    t0 = time.time()
    try:
        
        # Step 0 - Init
        if 1:
            pathIssueBool = False
            pathDCMPrev, pathDCMNow, pathScribble = None, None, None

            # Step 0.1 - Init (print)
            print ('\n\n ==================== PARAMS ==================== ')
            print ('\t- prefix   : ', prefix)
            print ('\t- expId    : ', expId)
            print ('\t- patientId: ', patientId)
            print ('\t- counter  : ', counter)
            print ('\t- viewType : ', viewType)
            print ('\t- sliceId  : ', sliceId)
            print ('\t- scribbleType: ', scribbleType)
            print (' ================================================ \n\n')

            # Step 0.2 - Init (get paths)
            pathExp = DIR_EXPERIMENTS / expId
            if not Path(pathExp).exists():
                print ('\t- [ERROR] Issue with pathExp: ', pathExp)
                pathExp = DIR_EXPERIMENTS / 'old1' / expId
                if not Path(pathExp).exists():
                    print ('\t- [ERROR] Issue with pathExp: ', pathExp)
                    pathIssueBool = True
            
            if counter > 1:
                pathDCMPrev = pathExp / fileNameForDCM(patientId, counter-1)
                if not Path(pathDCMPrev).exists():
                    print ('\t- [ERROR] Issue with pathDCMPrev: ', pathDCMPrev)
                    # pathIssueBool = True # use the base prediction

            pathDCMNow = pathExp / fileNameForDCM(patientId, counter)
            if not Path(pathDCMNow).exists():
                print ('\t- [ERROR] Issue with pathDCMMid: ', pathDCMNow)
                pathDCMNow = pathExp / fileNameOldForDCM(patientId, counter)
                if not Path(pathDCMNow).exists():
                    print ('\t- [ERROR] Issue with pathDCMNow: ', pathDCMNow)
                    pathIssueBool = True
            
            pathScribble = pathExp / fileNameForScribble(patientId, counter, viewType, sliceId)
            if not Path(pathScribble).exists():
                print ('\t- [ERROR] Issue with pathScribble: ', pathScribble)
                pathScribble = pathExp / fileNameOldForScribble(patientId, counter, viewType, sliceId)
                if not Path(pathScribble).exists():
                    print ('\t- [ERROR] Issue with pathScribble: ', pathScribble)
                    pathIssueBool = True
            if pathIssueBool:
                pdb.set_trace()
                return None
        
        # Step 1 - Load paths
        if 1:
            
            array3DCT, array3DPT, array3DGT, array3DPred, array3DPredPrev, array3DPredNow, array2DScribble = None, None, None, None, None, None, None
            
            try:
                
                # Step 1.1 - Load raw dicom data
                if 1:
                    patientIdObj       = orthancRequestUtils.getOrthancPatientIds(patientId)
                    if len(patientIdObj) == 0:
                        print (f"\t- [ERROR][plot()] patientIdObj is empty for patientId: {patientId}")
                        pdb.set_trace()
                    array3DCT, array3DPT   = orthancRequestUtils.downloadPatientZip(patientId, patientIdObj)
                    array3DGT, array3DPred = orthancRequestUtils.getSegsArray(patientId, patientIdObj) # [axial, coronal, sagittal]
                    print ('\t - [INFO] array3DCT.shape: ', array3DCT.shape)
                    print ('\t - [INFO] array3DPT.shape: ', array3DPT.shape)
                    print ('\t - [INFO] array3DGT.shape: ', array3DGT.shape)
                    print ('\t - [INFO] array3DPred.shape: ', array3DPred.shape)
                
                # Step 1.2 - Load prediction contours
                if 1:
                    listObjsCT         = orthancRequestUtils.getPyDicomObjects(patientId, patientIdObj, orthancRequestUtils.MODALITY_CT)
                    # array3DPredPrev, seriesDescPredPrev, _, _, _ = orthancRequestUtils.getSegArrayInShapeMismatchScene(listObjsCT, pathDCMPrev)
                    # array3DPredNow, seriesDescPredNow, _, _, _ = orthancRequestUtils.getSegArrayInShapeMismatchScene(listObjsCT, pathDCMNow)
                    if pathDCMPrev is not None:
                        array3DPredPrev = pydicom.dcmread(pathDCMPrev).pixel_array # data in orthanc is also read this way
                    else:
                        array3DPredPrev = np.array(array3DPred).copy()
                    array3DPredNow = pydicom.dcmread(pathDCMNow).pixel_array
                
                # Step 1.3 - Load scribble
                if 1:
                    array2DScribble = np.array(Image.open(pathScribble)) # [H,W,3] => R=pred, G=GT, B=scribble

                # 1.99 - Debug
                if 1:
                    print ('\t- [INFO] Data loaded successfully (Time taken: {:.2f} seconds)'.format(time.time()-t0))

            except:
                traceback.print_exc()
                pdb.set_trace()
        
        # Step 2 - Plot
        if 1:
            if array3DCT is not None and array3DPT is not None and array3DGT is not None and array3DPred is not None and array3DPredPrev is not None and array3DPredNow is not None and array2DScribble is not None:
                
                # Step 2.1 - Init
                fig,axarr = plt.subplots(1, 4, figsize=(10, 4))
                plt.subplots_adjust(wspace=0.02)

                # Step 2.1.2 - Init (scribble color)
                scribbleColorMapBaseFgd              = matplotlib.colors.ListedColormap([RGBA_ARRAY_YELLOW for _ in range(256)])
                scribbleColorMapFgd, scribbleNormFgd = getScribbleColorMap(scribbleColorMapBaseFgd, opacityBoolForScribblePoints=True)
                scribbleColorMapBaseBgd              = matplotlib.colors.ListedColormap([RGBA_ARRAY_BLUE for _ in range(256)])
                scribbleColorMapBgd, scribbleNormBgd = getScribbleColorMap(scribbleColorMapBaseBgd, opacityBoolForScribblePoints=True)
                
                # Step 2.1.3 - Init (get bbox coords)
                array2DPredPrev = getSliceForMaskArrayByViewTypeAndSliceId(array3DPredPrev, viewType, sliceId)
                if np.sum(array2DPredPrev) == 0:
                    print ('\t- [ERROR] No prediction found at sliceId: ', sliceId)
                    # pdb.set_trace()
                    print ('\t- [ERROR] Using array3DPredNow for bbox')
                    array2DPredPrev = getSliceForMaskArrayByViewTypeAndSliceId(array3DPredNow, viewType, sliceId)
                array2DPredPrevCenterXList, array2DPredPrevCenterYList = np.where(array2DPredPrev == 1)
                array2DPredPrevCenterX, array2DPredPrevCenterY = np.mean(array2DPredPrevCenterXList).astype(np.uint8), np.mean(array2DPredPrevCenterYList).astype(np.uint8)
                print ('\t- [INFO] array2DPredPrevCenterX: ', array2DPredPrevCenterX, 'array2DPredPrevCenterY: ', array2DPredPrevCenterY)
                if array2DPredPrevCenterX < BBOX_SIZE[0]//2:
                    array2DPredPrevCenterX = BBOX_SIZE[0]//2
                if array2DPredPrevCenterY < BBOX_SIZE[1]//2:
                    array2DPredPrevCenterY = BBOX_SIZE[1]//2
                if array2DPredPrevCenterX > array2DPredPrev.shape[0]-BBOX_SIZE[0]//2:
                    array2DPredPrevCenterX = array2DPredPrev.shape[0]-BBOX_SIZE[0]//2
                if array2DPredPrevCenterY > array2DPredPrev.shape[1]-BBOX_SIZE[1]//2:
                    array2DPredPrevCenterY = array2DPredPrev.shape[1]-BBOX_SIZE[1]//2
                print ('\t- [INFO] array2DPredPrevCenterX: ', array2DPredPrevCenterX, 'array2DPredPrevCenterY: ', array2DPredPrevCenterY)
                
                # Step 2.2 - Plot PT
                axarr[0].set_title('Slice s')
                # 
                if 'CHUP-059' in patientId:
                    sliceIdx = 17 # [12, 22, 25, 32, 39, 44]
                    print (f'\t- [INFO] Using sliceIdx {sliceIdx} for CHUP-059')
                    axarr[0].imshow(getSliceForScanArrayByViewTypeAndSliceId(array3DPT, viewType, sliceIdx, array2DPredPrevCenterX, array2DPredPrevCenterY), cmap=COLOR_GRAY) #, vmin=SUV_MIN_v1, vmax=SUV_MAX_v1)
                else:
                    axarr[0].imshow(getSliceForScanArrayByViewTypeAndSliceId(array3DPT, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), cmap=COLOR_GRAY) #, vmin=SUV_MIN_v1, vmax=SUV_MAX_v1)
                axarr[0].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DGT, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_GREEN)
                axarr[0].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredPrev, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_RED)
                axarr[0].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredNow, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_PINK, linestyles=LINESTYLE_REFINE)

                # Step 2.3 - Plot CT
                axarr[1].set_title('Slice s-1')
                axarr[1].imshow(getSliceForScanArrayByViewTypeAndSliceId(array3DCT, viewType, sliceId-1, array2DPredPrevCenterX, array2DPredPrevCenterY), cmap=COLOR_GRAY, vmin=HU_MIN, vmax=HU_MAX)
                axarr[1].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DGT, viewType, sliceId-1, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_GREEN)
                axarr[1].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredPrev, viewType, sliceId-1, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_RED)
                axarr[1].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredNow, viewType, sliceId-1, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_PINK, linestyles=LINESTYLE_REFINE)

                axarr[2].set_title('Slice s')
                axarr[2].imshow(getSliceForScanArrayByViewTypeAndSliceId(array3DCT, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), cmap=COLOR_GRAY, vmin=HU_MIN, vmax=HU_MAX)
                axarr[2].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DGT, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_GREEN)
                axarr[2].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredPrev, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_RED)
                axarr[2].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredNow, viewType, sliceId, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_PINK, linestyles=LINESTYLE_REFINE)

                axarr[3].set_title('Slice s+1')
                axarr[3].imshow(getSliceForScanArrayByViewTypeAndSliceId(array3DCT, viewType, sliceId+1, array2DPredPrevCenterX, array2DPredPrevCenterY), cmap=COLOR_GRAY, vmin=HU_MIN, vmax=HU_MAX)
                axarr[3].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DGT, viewType, sliceId+1, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_GREEN)
                axarr[3].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredPrev, viewType, sliceId+1, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_RED)
                axarr[3].contour(getSliceForMaskArrayByViewTypeAndSliceId(array3DPredNow, viewType, sliceId+1, array2DPredPrevCenterX, array2DPredPrevCenterY), levels=[0.5], colors=COLOR_PINK, linestyles=LINESTYLE_REFINE)

                # Step 2.4 - Plot scribble
                array2DScribbleThis = array2DScribble[:,:,2]
                if array2DPredPrevCenterX is not None and array2DPredPrevCenterY is not None:
                    array2DScribbleThis = array2DScribbleThis[array2DPredPrevCenterX-BBOX_SIZE[0]//2:array2DPredPrevCenterX+BBOX_SIZE[0]//2, array2DPredPrevCenterY-BBOX_SIZE[1]//2:array2DPredPrevCenterY+BBOX_SIZE[1]//2]
                if scribbleType == KEY_SCRIBBLE_FGD:
                    axarr[0].imshow(array2DScribbleThis, cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                    axarr[2].imshow(array2DScribbleThis, cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                elif scribbleType == KEY_SCRIBBLE_BGD:
                    axarr[0].imshow(array2DScribbleThis, cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                    axarr[2].imshow(array2DScribbleThis, cmap=scribbleColorMapBgd, norm=scribbleNormBgd)

                # Step 2.4 - Remove all ticks
                for ax in axarr:
                    ax.axis('off') 

                pathFile = DIR_EXPERIMENTS_OUTPUTS / f'{prefix}-{expId}-{patientId}-{counter:03d}-{viewType}-{sliceId:03d}-{scribbleType}.png'
                plt.savefig(pathFile, bbox_inches='tight', dpi=DPI)
                print (f' - Saving as {pathFile}')
                # plt.show(block=False)

            else:
                print ('\t- [ERROR] Issue with data loading')
        
        pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()

if __name__ == "__main__":

    # Step 0 - Plot params
    DPI = 200
    BBOX_SIZE = (72,72)

    try:
        
        # Fig 3a
        if 1:
            prefixParam    = 'Fig3a'
            expIdParam     = "2024-11-29 10-58-32 -- vibrant_villani__Yauheniya-Makarevich-NonExpert-AI-based"
            patientIdParam = 'CHUP-005-gt-filtered-gausssig2'
            counterParam   = 38
            viewTypeParam  = KEY_AXIAL
            sliceIdParam   = 54
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)

        # Fig 3b
        if 1:
            prefixParam    = 'Fig3b'
            expIdParam     = "2024-11-28 13-41-26 -- competent_haibt__Frank-Dankers-NonExpert-AI-based"
            patientIdParam = 'CHUP-005-gt-filtered-gausssig2'
            counterParam   = 3
            viewTypeParam  = KEY_AXIAL
            sliceIdParam   = 54
            scribbleTypeParam = KEY_SCRIBBLE_FGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)

        # Fig 3c
        if 1:
            prefixParam    = 'Fig3c'
            expIdParam     = "2024-11-27 09-48-36 -- trusting_mclaren__Prerak-Mody-NonExpert-AI-based"
            patientIdParam = 'CHUP-033-gt-filtered-gausssig2'
            counterParam   = 43
            viewTypeParam  = KEY_SAGITTAL
            sliceIdParam   = 84
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
        
        # Fig 3d
        if 1:
            prefixParam    = 'Fig3d'
            expIdParam     = "2024-12-04 12-06-33 -- jovial_tu__Faeze-Gholamiankhah-NonExpert-AI-based"
            patientIdParam = 'CHUP-033-gt-filtered-gausssig2'
            counterParam   = 25
            viewTypeParam  = KEY_SAGITTAL
            sliceIdParam   = 84
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)

        # Fig 3e
        if 1:
            prefixParam    = 'Fig3e'
            expIdParam     = "2024-08-08 18-01-36 -- quirky_williamson"
            patientIdParam = 'CHMR016'
            counterParam   = 1
            viewTypeParam  = KEY_CORONAL
            sliceIdParam   = 59
            scribbleTypeParam = KEY_SCRIBBLE_FGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
            # "D:\HCAI\Project 5 - Interactive Contour Refinement\code\cornerstone3D-trials\_experiments\old1\2024-08-08 18-01-36 -- quirky_williamson\CHMR016-Series-SEG-Refine-1-Coronal-slice059.png"
        
        # Fig 3f (wont work)
        if 1:
            prefixParam    = 'Fig3f'
            expIdParam     = "2024-12-05 12-37-15 -- blissful_hopper__Faeze-Gholamiankhah-NonExpert-AI-based"
            patientIdParam = 'CHUP-059-gt-filtered-gausssig2'
            counterParam   = 30
            viewTypeParam  = KEY_CORONAL
            sliceIdParam   = 42
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
        
        # Fig 3f
        if 1:
            prefixParam    = 'Fig3f'
            expIdParam     = "2024-12-09 09-39-20 -- vibrant_carver__Alex-Vieth-NonExpert-AI-based"
            patientIdParam = 'CHUP-044-gt-filtered-gausssig2'
            counterParam   = 40
            viewTypeParam  = KEY_CORONAL
            sliceIdParam   = 59
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
        
        # Fig 3g
        if 1:

            prefixParam    = 'Fig3g'
            expIdParam     = "2024-12-04 13-11-40 -- exciting_spence__Patrick-de Koning-NonExpert-AI-based"
            patientIdParam = 'CHUP-028-gt-filtered-gausssig2'
            counterParam   = 1 # 1, 3
            viewTypeParam  = KEY_AXIAL
            sliceIdParam   = 57 # 57, 64
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
        
        # Fig 3h
        if 1:

            prefixParam    = 'Fig3h'
            expIdParam     = "2024-12-09 13-16-41 -- vigorous_khayyam__Frank-Dankers-NonExpert-AI-based"
            patientIdParam = 'CHUP-064-gt-filtered-gausssig2'
            counterParam   = 1 
            viewTypeParam  = KEY_AXIAL
            sliceIdParam   = 65
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
        
        # Fig 3i (wont work due to issue in CHUP059-PET scan)
        if 0:

            prefixParam    = 'Fig3i'
            expIdParam     = "2024-12-13 08-11-36 -- quirky_ishizaka__Ruochen-Gao-NonExpert-AI-based"
            patientIdParam = 'CHUP-059-gt-filtered-gausssig2'
            counterParam   = 2
            viewTypeParam  = KEY_AXIAL
            sliceIdParam   = 76
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
        
        # Fig3i
        if 1:

            prefixParam    = 'Fig3i'
            expIdParam     = "2024-12-12 14-50-57 -- brave_diffie__Martin-De Jong-Expert-AI-based"
            patientIdParam = 'CHUP-033-gt-filtered-gausssig2'
            counterParam   = 1
            viewTypeParam  = KEY_SAGITTAL
            sliceIdParam   = 52
            scribbleTypeParam = KEY_SCRIBBLE_BGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)
        
        # Fig 3j
        if 1:

            prefixParam    = 'Fig3j'
            expIdParam     = "2024-12-09 13-16-41 -- vigorous_khayyam__Frank-Dankers-NonExpert-AI-based"
            patientIdParam = 'CHUP-064-gt-filtered-gausssig2'
            counterParam   = 8
            viewTypeParam  = KEY_AXIAL
            sliceIdParam   = 51
            scribbleTypeParam = KEY_SCRIBBLE_FGD
            plot(prefixParam, expIdParam, patientIdParam, counterParam, viewTypeParam, sliceIdParam, scribbleTypeParam)

    except:
        traceback.print_exc()
        pdb.set_trace()

"""
*On Windows
**start Docker machine too (else orthanc will not work)
cd "D:\HCAI\Project 5 - Interactive Contour Refinement\code\cornerstone3D-trials"
conda activate interactive-refinement
python src/backend/utils/plotScribbleUtils.py

"""