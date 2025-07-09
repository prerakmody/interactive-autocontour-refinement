"""
Process
1. Take manual and semi-automated folder input
2. Loop over the .dcm files in each folder and find out unique patientsIds
 - patientsManual = {'CHMR028' : [CHMR028-ManualRefine-1.dcm]}
 - patientsSemiAuto = {'CHMR028' : [CHMR028-Refine-1.dcm]}
3. Download the GT and prediction of this patient from Orthanc (localhost:8042)
"""

# Import public libs 
import os
import pdb
import tqdm
import copy
import json
import shutil
import time
import pyvista
import pydicom
import platform
import traceback
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.measure
import surface_distance
from pathlib import Path
import SimpleITK as sitk
from collections import Counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", message="The value length .* exceeds the maximum length of 64 allowed for VR LO.") # pydicom warning
warnings.filterwarnings("ignore", message="Points is not a float type.") # pyvista warning


# Import private libs
import orthancRequestUtils

# Constants
DIR_FILE        = Path(__file__).resolve().parent # <projectRoot>/src/backend/utils/
DIR_ROOT        = DIR_FILE.parent.parent.parent # <projectRoot>/src/backend/
DIR_EXPERIMENTS         = DIR_ROOT / '_experiments'
DIR_EXPERIMENTS_OUTPUTS = DIR_EXPERIMENTS / 'experiment-outputs'
DIR_TMP                 = DIR_ROOT / '_tmp'
DIR_EXPERIMENT_VIDEOS   = DIR_EXPERIMENTS_OUTPUTS / 'experiment-videos'

KEY_INTERACTIONCOUNT_MANUAL = 'Interaction (Count) - Manual'
KEY_INTERACTIONCOUNT_SEMIAUTO = 'Interaction (Count) - Semi-Auto'

KEY_ACTION_START_EPOCH_MANUAL = 'Action Start Epoch - Manual'
KEY_ACTION_START_EPOCH_SEMIAUTO = 'Action Start Epoch - Semi-Auto'

KEY_ACTION_SCROLL_DATA_MANUAL = 'Action Scroll Data - Manual'
KEY_ACTION_SCROLL_DATA_SEMIAUTO = 'Action Scroll Data - Semi-Auto'


KEY_TIME_ACTION_SECONDS_MANUAL             = 'Time(s) (Action) - Manual'
KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL   = 'Time(s) (From Last Action) - Manual'
KEY_TIME_ACTION_SECONDS_SEMIAUTO           = 'Time(s) (Action) - Semi-Auto'
KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO = 'Time(s) (From Last Action) - Semi-Auto'

KEY_TIME_ACTION_SECONDS_MANUAL_CUMM             = 'Time(s) (Action) - Manual (Cumm)'
KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM   = 'Time(s) (From Last Action) - Manual (Cumm)'
KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM           = 'Time(s) (Action) - Semi-Auto (Cumm)'
KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM = 'Time(s) (From Last Action) - Semi-Auto (Cumm)'

KEY_TIME_ACTION_SECONDS_MANUAL_CUMM_PERC             = 'Time(s) (Action) - Manual (Cumm) (as % of total)'
KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM_PERC   = 'Time(s) (From Last Action) - Manual (Cumm) (as % of total)'
KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM_PERC           = 'Time(s) (Action) - Semi-Auto (Cumm) (as % of total)'
KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM_PERC = 'Time(s) (From Last Action) - Semi-Auto (Cumm) (as % of total)'

KEY_MANUAL = 'Manual'
KEY_SEMIAUTO = 'Semi-Auto'
KEY_MANUAL_STEPS = 'Manual Steps'
KEY_SEMIAUTO_STEPS = 'Semi-Auto Steps'

KEY_MANUAL_DICE   = 'Dice - Manual'
KEY_SEMIAUTO_DICE = 'Dice - AI'
KEY_MANUAL_SURFACE_DICE = 'Surface Dice ({}mm) - Manual'
KEY_SEMIAUTO_SURFACE_DICE = 'Surface Dice ({}mm) - AI'
KEY_MANUAL_SURFACE_DICE_1MM = 'Surface Dice (1mm) - Manual'
KEY_MANUAL_SURFACE_DICE_2MM = 'Surface Dice (2mm) - Manual'
KEY_MANUAL_SURFACE_DICE_3MM = 'Surface Dice (3mm) - Manual'
KEY_SEMIAUTO_SURFACE_DICE_1MM = 'Surface Dice (1mm) - AI'
KEY_SEMIAUTO_SURFACE_DICE_2MM = 'Surface Dice (2mm) - AI'
KEY_SEMIAUTO_SURFACE_DICE_3MM = 'Surface Dice (3mm) - AI'
KEY_INTERP = ' (interp)'

KEY_FILEPATHS_VIZ_3D_AI = 'FilePathsViz3DAI'
KEY_FILEPATHS_VIZ_3D_MANUAL = 'FilePathsViz3DGT'

KEY_PATIENTID = 'patientId'
KEY_INTERACTIONTYPE = 'Interaction (Type)'
KEY_INTERACTIONCOUNT = 'Interaction (Count)'
KEY_METRIC = 'Metric'
KEY_VALUE  = 'Metric (Value)'
KEY_SCROLLS = 'Scrolls'
KEY_SCROLLS_NORM = 'Scrolls (Norm)'
KEY_SCROLLS_NORM_PLOT = 'Scrolls (Norm)(Plot)'
# KEY_TIME_ACTION           = 'Time(s) (Action)'
# KEY_TIME_FROM_LAST_ACTION = 'Time(s) (From Last Action)'
# KEY_TIME_ACTION_CUMM           = 'Time(s) (This Action) (Cumm)'
# KEY_TIME_FROM_LAST_ACTION_CUMM = 'Time(s) (From Last Action) (Cumm)'
KEY_TIME_ACTION           = 'Time for Interaction (in sec)'
KEY_TIME_FROM_LAST_ACTION = 'Time for Interaction + Navigation (in sec)'
# KEY_TIME_ACTION_CUMM           = 'Time for Interaction (in sec) (cummulative)'
# KEY_TIME_FROM_LAST_ACTION_CUMM = 'Time for Interaction + Navigation (in sec) (cummulative)'
KEY_TIME_ACTION_CUMM           = 'Total Editing Time (sec)'
KEY_TIME_FROM_LAST_ACTION_CUMM = 'Total Time (navigation + editing) (sec)'

KEY_INTERACTIONCOUNT_PERC           = 'Interaction (Count) (% of total)'
KEY_TIME_ACTION_CUMM_PERC           = 'Total Editing Time (% of total)'
KEY_TIME_FROM_LAST_ACTION_CUMM_PERC = 'Total Time (navigation + editing) (% of total)'

KEY_REFINE       = '-Refine'
KEY_MANUALREFINE = '-ManualRefine'

SUFFIX_SLICE_SCROLL_JSON = '__slice-scroll.json'
KEY_AXIAL   = 'Axial'
KEY_CORONAL = 'Coronal'
KEY_SAGITTAL= 'Sagittal'

VIDEO_EXT = '.mp4' # '.mp4', '.avi'
FILENAME_VIZ_3D_FORMAT           = '{}__{}__iter{:03d}__view{}.png'
FILENAME_VIZ_3D_FORMAT_AI        = '{}__AI__iter{:03d}__view{}.png'
FILENAME_VIZ_3D_FORMAT_MANUAL    = '{}__Manual__iter{:03d}__view{}.png'
FILENAME_VIZ_VIDEO_FORMAT_AI     = '{}__AI__iters{:03d}' + VIDEO_EXT
FILENAME_VIZ_VIDEO_FORMAT_MANUAL = '{}__Manual__iters{:03d}' + VIDEO_EXT
FILENAME_VIZ_VIDEO_FORMAT_NOGT   = '{}__NoGT__iters{:03d}' + VIDEO_EXT

XAXIS_FILEIDENTIFER = {
    KEY_INTERACTIONCOUNT                 : 'interactionCount'
    , KEY_INTERACTIONCOUNT_PERC          : 'interactionCountPerc'
    , KEY_TIME_ACTION_CUMM               : 'timeForAction'
    , KEY_TIME_FROM_LAST_ACTION_CUMM     : 'timeFromLastAction'
    , KEY_TIME_ACTION_CUMM_PERC          : 'timeForActionPerc'
    , KEY_TIME_FROM_LAST_ACTION_CUMM_PERC: 'timeFromLastActionPerc'
}

PATIENT_COUNTER_SPLITTER = '___'

SEABORN_PALETE = {
    KEY_MANUAL_DICE: 'blue'
    , KEY_SEMIAUTO_DICE: 'orange'
    , KEY_MANUAL_SURFACE_DICE_1MM: '#aec7e8'  # light blue
    , KEY_MANUAL_SURFACE_DICE_2MM: '#4a90e2'  # medium blue
    , KEY_MANUAL_SURFACE_DICE_3MM: '#1f3b4d'  # dark blue
    # , KEY_SEMIAUTO_SURFACE_DICE_1MM: '#ffbb78'  # light orange
    # , KEY_SEMIAUTO_SURFACE_DICE_2MM: '#ff9933'  # medium orange
    # , KEY_SEMIAUTO_SURFACE_DICE_3MM: '#d62728'  # dark orange
    , KEY_SEMIAUTO_SURFACE_DICE_1MM: '#ffe478'  # light yellow
    , KEY_SEMIAUTO_SURFACE_DICE_2MM: '#ffc533'  # medium yellow
    , KEY_SEMIAUTO_SURFACE_DICE_3MM: '#d62728'  # dark orange
}

PRIVATE_BLOCK_GROUP = 0x1001
PRIVATE_BLOCK_CREATOR = 'Mody - AI-assisted Interactive Refinement v1.0'
TAG_OFFSET           = 0x1000
TAG_TIME_TO_BRUSH    = 0x01 # (1001,1001)	Unknown  Tag &  Data	1.8380001
TAG_TIME_TO_SCRIBBLE = 0x02
TAG_TIME_TO_DISTMAP  = 0x03
TAG_TIME_TO_INFER    = 0x04
TAG_TIME_EPOCH       = 0x05
VALUEREP_FLOAT32     = pydicom.valuerep.VR.FL # Floating Point Single 
VALUEREP_STRING       = pydicom.valuerep.VR.ST # Short Text

######################################################
# Utils
######################################################

def compute_dice(mask_gt, mask_pred, meta=''):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        print (f" - [compute_dice()] Both masks are empty for {meta}")
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def getSurfaceDICE(maskGT, maskPred, spacing, tolerancesMM):
    """
    Parameters:
    -----------
    maskGT: torch.tensor, [H,W,sliceCount]
    maskPred: torch.tensor, [H,W,sliceCount]
    spacing: float, list[float], [3]
    tolerancesMM: list[float]

    Note:
    ----
    - getSurfaceDICE(finalGT, finalPred, spacing=[1.5,1.5,1.5], tolerancesMM=[1,2,3])
    """
    
    surfaceDICEs = {}

    try:
        
        # Step 0 - Init
        pass

        # Step 1 - Convert to numpy
        if not isinstance(maskGT, np.ndarray) and not isinstance(maskPred, np.ndarray):
            maskGT  = maskGT.cpu().numpy()
            maskPred= maskPred.cpu().numpy()
        
        # Step 2 - Get surface distance
        surfaceDistanceObj = surface_distance.compute_surface_distances(mask_gt=maskGT.astype(bool), mask_pred=maskPred.astype(bool), spacing_mm=spacing)

        # Step 3 - Compute surface dice at different tolerances
        for toleranceMM in tolerancesMM:
            surfaceDICE        = surface_distance.compute_surface_dice_at_tolerance(surfaceDistanceObj, toleranceMM)
            surfaceDICEs[toleranceMM] = surfaceDICE

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return surfaceDICEs

def plotMasks(patientId, array1, array2, array1Name, array2Name, sliceIdx, filenamePrefix='arrayGTAndPred'):
    """
    Parameters:
    -----------
    patientId: str
    array1  : np.ndarray, [axial, coronal, sagittal] --> data is expeted like this (TODO: should we also store it in .dcm files like this?)
    array2: np.ndarray, [axial, coronal, sagittal]
    sliceIdx : int
    filenamePrefix: str, 'arrayGTAndPred'

    Note
    ----
     - check src/backend/interaction-server.py:getCTArray() (for instance in ctInstances: ctArray[:, :, int(instance.InstanceNumber)-1] = instance.pixel_array)
       -- so there we are storing in the format [:,:, axial], while here we are storing in the format [axial, :,:]
     - RAS (Right, Anterior, Superior) coordinate system
       -- Coronal i.e. array1[coronalSliceIdx,:,:]
       -- Sagittal i.e. array1[:,sagittalSliceIdx,:]
       -- Axial i.e. array1[:,:,axialSliceIdx]
    """
    rotAxial    = lambda x: x
    # rotSagittal = lambda x: x #np.rot90(x, k=2)
    # rotCoronal  = lambda x: x #np.rot90(x, k=3)
    rotSagittal = lambda x: np.flipud(x) # Works
    rotCoronal  = lambda x: np.flipud(x) # Works
    
    # transform = lambda x: np.fliplr(np.rot90(x, k=3))
    # transform = lambda x: np.flipud(x)
    transform = lambda x: x

    try:
        
        f,axarr = plt.subplots(1,3, figsize=(6,6))
        if array1 is not None:

            array1Copy = array1.copy()
            for axialSliceIdx in range(array1Copy.shape[0]):
                array1Copy[axialSliceIdx,:,:] = transform(array1Copy[axialSliceIdx,:,:])
            
            axarr[0].imshow(array1Copy[sliceIdx,:,:], cmap='gray'); axarr[0].set_title('Axial')
            axarr[0].contour(array1Copy[sliceIdx,:,:], levels=[0.5], colors='green', linewidths=0.25)

            axarr[1].imshow(rotSagittal(array1Copy[:,:,sliceIdx]), cmap='gray'); axarr[1].set_title('Sagittal')
            axarr[1].contour(rotSagittal(array1Copy[:,:,sliceIdx]), levels=[0.5], colors='green', linewidths=0.25)

            axarr[2].imshow(rotCoronal(array1Copy[:,sliceIdx,:]), cmap='gray'); axarr[2].set_title('Coronal')
            axarr[2].contour(rotCoronal(array1Copy[:,sliceIdx,:]), levels=[0.5], colors='green', linewidths=0.25)
            
        if array2 is not None:   
            array2Copy = array2.copy()
            for axialSliceIdx in range(array2Copy.shape[0]):
                array2Copy[axialSliceIdx,:,:] = transform(array2Copy[axialSliceIdx,:,:])
            axarr[0].contour(array2Copy[sliceIdx,:,:], levels=[0.5], colors='red', linewidths=0.25)
            axarr[1].contour(rotSagittal(array2Copy[:,:,sliceIdx]), levels=[0.5], colors='red', linewidths=0.25)
            axarr[2].contour(rotCoronal(array2Copy[:,sliceIdx,:]), levels=[0.5], colors='red', linewidths=0.25)
        
        # Invert axes 

        plt.suptitle('{} (green) vs {} (red) \n patientId={} \n sliceIdx={}'.format(array1Name, array2Name, patientId, sliceIdx), y=0.75)
        plt.savefig('./_tmp/{}-{}-sliceIdx{}.png'.format(patientId, filenamePrefix, sliceIdx), dpi=300, bbox_inches='tight')

    except:
        traceback.print_exc()
        pdb.set_trace()

def getComponentCount(binaryArray, meta=''):
    """
    To check if the final refinements have more than 1 component

    parameters:
    -----------
    binaryArray: np.ndarray, [H,W], uint8
    """
    try:
        
        filter = sitk.ConnectedComponentImageFilter()
        _ = filter.Execute(sitk.GetImageFromArray(binaryArray))

        componentCount = filter.GetObjectCount()
        print (f"   --> [INFO][eval()]{meta} componentCount: {componentCount}")

    except:
        traceback.print_exc()
        pdb.set_trace()

######################################################
# Utils - PyVista
######################################################

def convertScribbleToArray(scribble2D, slicesId, maxSliceCount):

    try:
        pass

    except:
        traceback.print_exc()
        pdb.set_trace()

# test (for non-monitor machines)
def testPyvista():
    """
    pip install pyvista==0.44.{1,2}
    """
    try:

        import pyvista as pv

        # Create a simple mesh
        mesh = pv.Sphere()

        # Create a plotter object
        plotter = pv.Plotter(off_screen=True)
        _ = plotter.add_mesh(mesh)
        filePath = DIR_TMP / 'pyvista_test_screenshot.png'
        Path(filePath).parent.mkdir(parents=True, exist_ok=True)
        plotter.show(screenshot=filePath)
        return True
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def convertArrayToMesh(array):

    mesh = None

    try:
        
        # Step 1 - Copy
        arrayCopy = array.copy()

        # Step 2 - Get the vertices and faces
        verts, faces, _, _ = skimage.measure.marching_cubes(arrayCopy, level=0)

        # Step 3 - Convert the faces to the format needed by PyVista
        faces = np.hstack([[3] + list(face) for face in faces])

        # Step 4 - Create PyVista meshes
        mesh = pyvista.PolyData(verts, faces)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return mesh

def getPyvistaPlotter(doScreenshot, arrayRef=None):

    plotter = None
    try:
        
        # Step 1 - make a plotter
        plotter = pyvista.Plotter(off_screen=doScreenshot)

        # Step 2 - Make extras
        plotter.show_axes()
        plotter.add_point_labels(points=np.array([[0.0, 0.0, 0.0]]), labels=['Origin'], point_size=10, text_color='black') # weird "UserWarning: Points is not a float type." warning
        
        # Step 3 - Add a box (for reference)
        if arrayRef is not None:

            # Step 3.1 - Add points
            max_x, max_y, max_z = arrayRef.shape
            max_x, max_y, max_z = int(max_x), int(max_y), int(max_z)
            points = np.array([
                [0, 0, 0],
                [max_x, 0, 0],
                [0, max_y, 0],
                [0, 0, max_z],
                [max_x, max_y, 0],
                [max_x, 0, max_z],
                [0, max_y, max_z],
                [max_x, max_y, max_z]
            ])
            plotter.add_points(points, color='blue', point_size=10)
            
            # Step 3.2 - Add line segments connecting the points
            line_segments = [
                [0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 4], [2, 6], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]
            ]
            for segment in line_segments:
                line = pyvista.Line(points[segment[0]], points[segment[1]])
                plotter.add_mesh(line, color='black', line_width=2)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return plotter

def defineShortcutKeys(plotter, actorGT=None, actorPred=None, defaultCameraArray=None):

    try:
        
        # Step 1 - Toggling for arrays
        def toggle_visibility(actor):
            actor.SetVisibility(not actor.GetVisibility())
            plotter.render()
        
        if actorGT is not None:
            plotter.add_key_event('g', lambda: toggle_visibility(actorGT))
        if actorPred is not None:
            plotter.add_key_event('p', lambda: toggle_visibility(actorPred))
        if actorGT is not None and actorPred is not None:
            plotter.add_key_event('c', lambda: [toggle_visibility(plotter.actors[name]) for name in plotter.actors if 'Line' in name])

        # Step 2 - Set camera position
        def set_camera_position():
            if defaultCameraArray is not None:
                plotter.camera_position = defaultCameraArray
            else:
                plotter.camera_position = [(72.0, 72.0, -72*3 - 10), (72.0, 72.0, 72.0), (1.0, 0.0, 0.0)] # position, focal, viewup direction
            plotter.render()
        plotter.add_key_event('q', set_camera_position)

    except:
        traceback.print_exc()
        pdb.set_trace()

# Entry fun
def renderAndSaveSurface(arrayGT, arrayPred, userName, type, iterId, doScreenshot=True, arrayGTColor='green', arrayPredColor='red'):
    """
    Parameters:
    -----------
    arrayGT: np.ndarray  , [axial, coronal, sagittal], values=[0,1], uint8
    arrayPred: np.ndarray, [axial, coronal, sagittal], values=[0,1], uint8

    Notes
    -----
     - Here 72 is midpPoint of arrayGT/arrayPred
     - brew install imagemagick

    """

    filePathCombined = None

    try:
        
        # Step 0 - Init
        # doScreenshot = False; print (f"\n !!!!!!! [renderAndSaveSurface()] doScreenshot: {doScreenshot} \n")
        defaultCameraArray = [(72.0, 72.0, -72*3 - 10), (72.0, 72.0, 72.0), (1.0, 0.0, 0.0)] # position, focal=center of array{GT,Pred}, viewup direction (currently x-axis)

        # Step 1 - Get data
        if 1:
            meshGT = convertArrayToMesh(arrayGT)
            meshPred = convertArrayToMesh(arrayPred)
            if meshGT is None or meshPred is None:
                print (f" - [renderAndSaveSurface()] meshGT or meshPred is None")
                return

        # Step 2 - Create a plotter object
        plotter = getPyvistaPlotter(doScreenshot, arrayGT)
        if plotter is None:
            print (f" - [renderAndSaveSurface()] plotter is None")
            return
        
        # Step 3 - Add the meshes to the plotter
        actorGT = plotter.add_mesh(meshGT, color=arrayGTColor, opacity=0.5)
        actorPred = plotter.add_mesh(meshPred, color=arrayPredColor, opacity=0.5)
        defineShortcutKeys(plotter, actorGT, actorPred, defaultCameraArray)

        # Step 99 - Save the screenshot
        if doScreenshot:
            
            DIR_EXPERIMENT_VIDEOS_USER = DIR_EXPERIMENT_VIDEOS / userName

            # Step 99.1 - Save screenshots from all 4 views
            labelActor = plotter.add_point_labels(points=np.array([[144.0 - 10, 0.0, 0.0]]), labels=['FrameId:{:03d}'.format(iterId)], point_size=1, font_size=40, text_color='black') # weird "UserWarning: Points is not a float type." warning
            plotter.camera_position = [(72.0, 72.0, -72*4), (72.0, 72.0, 72.0), (1.0, 0.0, 0.0)]; plotter.render() # v1
            filePath1 = DIR_EXPERIMENT_VIDEOS_USER / FILENAME_VIZ_3D_FORMAT.format(userName, type, iterId, 1)
            Path(filePath1).parent.mkdir(parents=True, exist_ok=True)
            _ = plotter.screenshot(str(filePath1))
            plotter.remove_actor(labelActor)

            plotter.camera_position = [(72, 72*6, 72), (72.0, 72.0, 72.0), (1.0, 0.0, 0.0)]; plotter.render() # v2
            filePath2 = DIR_EXPERIMENT_VIDEOS_USER / FILENAME_VIZ_3D_FORMAT.format(userName, type, iterId, 2)
            _ = plotter.screenshot(str(filePath2))

            plotter.camera_position = [(72.0, 72.0, 72*6), (72.0, 72.0, 72.0), (1.0, 0.0, 0.0)]; plotter.render() # v3
            filePath3 = DIR_EXPERIMENT_VIDEOS_USER / FILENAME_VIZ_3D_FORMAT.format(userName, type, iterId, 3)
            _ = plotter.screenshot(str(filePath3))

            plotter.camera_position = [(72, -72*4, 72), (72.0, 72.0, 72.0), (1.0, 0.0, 0.0)]; plotter.render() # v4
            filePath4 = DIR_EXPERIMENT_VIDEOS_USER / FILENAME_VIZ_3D_FORMAT.format(userName, type, iterId, 4)
            _ = plotter.screenshot(str(filePath4))
            plotter.close() 

            # Step 99.2 - Combine all 4 screenshots into a single image (horizontally)
            filePathCombined = DIR_EXPERIMENT_VIDEOS_USER / FILENAME_VIZ_3D_FORMAT.format(userName, type, iterId, 'All')
            Path(filePathCombined).parent.mkdir(parents=True, exist_ok=True)

            filePath1Exists = Path(filePath1).exists()
            filePath2Exists = Path(filePath2).exists()
            filePath3Exists = Path(filePath3).exists()
            filePath4Exists = Path(filePath4).exists()
            if filePath1Exists and filePath2Exists and filePath3Exists and filePath4Exists:
                null_device = 'NUL' if platform.system() == 'Windows' else '/dev/null'
                # _ = os.system(f"magick convert +append '{filePath1}' '{filePath2}' '{filePath3}' '{filePath4}' '{filePathCombined}'  > /dev/null 2>&1")
                _ = os.system(f'magick convert +append "{filePath1}" "{filePath2}" "{filePath3}" "{filePath4}" "{filePathCombined}"  > {null_device} 2>&1')
                # _ = os.system(f'magick convert +append "{filePath1.name}" "{filePath2.name}" "{filePath3.name}" "{filePath4.name}" "{filePathCombined.name}"  > /dev/null 2>&1')

                # Step 99.3 - Clean up
                os.remove(filePath1)
                os.remove(filePath2)
                os.remove(filePath3)
                os.remove(filePath4)
            else:
                print (f" - [renderAndSaveSurface()] One of the files does not exist: file1: {filePath1Exists}, file2: {filePath2Exists}, file3: {filePath3Exists}, file4: {filePath4Exists}")
                

        else:
            print (f" - [renderAndSaveSurface()] Showing the plotter ...")
            plotter.show()
            camera_position = plotter.camera_position
            print(f"Camera Position: {camera_position}")
    
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return filePathCombined

def create_video_old(imagePaths, output_video_path, fps=1):

    try:
        import os
        import cv2
        
        if len(imagePaths) == 0:
            print (f" - [create_video()] No images found in imagePaths: {imagePaths}")
            return
        
        # Step 1 - Sort the image paths
        imagePaths.sort()  # Ensure the images are in the correct order

        # Step 2 - Read the first image to get the dimensions
        frame = cv2.imread(imagePaths[0])
        height, width, layers = frame.shape

        # Step 3.1 - Define the codec and create a VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID'
        fourcc = cv2.VideoWriter_fourcc(*'MJPG'), 
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Step 3.2 - Write the images to the video
        for imagePath in imagePaths:
            video.write(cv2.imread(imagePath))

        # Step 4 - Release the video object
        video.release()
        cv2.destroyAllWindows()
    
    except:
        traceback.print_exc()
        pdb.set_trace()

# Exit func
def create_video(imagePaths, output_video_path, fps, minFrames):

    try:
        import imageio.v2 as imageio 

        # Step 0 - Init
        print (' ===================================== ')
        ffmpeg_params = ['-probesize', '5000000']
        writer = imageio.get_writer(output_video_path, fps=fps, codec='libx264', ffmpeg_params=ffmpeg_params)
        print (' ===================================== ')

        # Step 1 - Write the images to the video
        lastImage = None
        framesDone = 0
        for image_idx, imagePath in enumerate(imagePaths):
            
            # Step 1.1 - Check if the image exists
            if imagePath is not None:
                lastImage = imageio.imread(imagePath)
            else:
                print (f" - [create_video()] imagePath: {image_idx} is None")
                if lastImage is not None:
                    lastImage = np.zeros_like(lastImage)
                else:
                    continue
            
            # Step 1.2 - Write the image
            writer.append_data(lastImage)
            framesDone += 1
        
        if framesDone < minFrames:
            print (f" - [create_video()] framesDone: {framesDone}, minFrames: {minFrames}")
            for _ in range(minFrames-framesDone):
                writer.append_data(lastImage)

        # Step 2 - Close the writer
        print (' ================= || ==================== ')
        writer.close()
        print (' ================= || ==================== ')

        # Step 99 -Clean up
        for imageId, imagePath in enumerate(imagePaths):
            if imagePath is not None and imageId != len(imagePaths)-1: # keep the last image
                os.remove(imagePath)

    except:
        traceback.print_exc()
        pdb.set_trace()


######################################################
# Evals
######################################################

getName = lambda path: '{} / {}'.format(Path(path).parts[-2], Path(path).parts[-1])

def getRefineDCMs(pathsManualExperiments, pathsSemiAutoExperiments):
    
    patientsManual, patientsSemiAuto = {}, {}
    patientIdCounterManual = {}
    patientIdCounterSemiAuto = {}
    try:
        
        # Step 0 - Init
        def extractCounter(dcmFile):
            return int(dcmFile.name.split('-')[-1].split('.')[0])

        # Step 1 - Loop over the manual experiments
        for pathManual in pathsManualExperiments:
            
            # Step 1.1 - Init
            if not Path(pathManual).exists():
                print (f" - [ERROR][getRefineDCMs()] pathManual: {pathManual} does not exist")
                continue
            
            # Step 1.2 - Get the .dcm files
            pathsDcms = list(pathManual.glob('**/*.dcm'))
            if len(pathsDcms) == 0:
                print (f" - [ERROR][getRefineDCMs()] No .dcm files found in pathManual: {pathManual}")
                pdb.set_trace()
                continue
            pathsDcms = sorted(pathsDcms, key=extractCounter)

            # Step 1.3 - Get the patientId (and handle >1 experiments on the same patientId)
            patientId = pathsDcms[0].name.split(KEY_MANUALREFINE)[0]
            if patientId in patientIdCounterManual: patientIdCounterManual[patientId] +=1
            else                                  : patientIdCounterManual[patientId] = 1
            patientIdNew = patientId + PATIENT_COUNTER_SPLITTER + str(patientIdCounterManual[patientId])
            
            # Step 1.4 - Store the .dcm files
            patientsManual[patientIdNew] = pathsDcms
            # print (f" - [INFO][getRefineDCMs(Manual)] patientId: {patientIdNew}, pathsDcms[0].name: {pathsDcms[0].name}")
        
        # Step 2 - Loop over the semi-auto experiments
        for pathSemiAuto in pathsSemiAutoExperiments:

            # Step 2.1 - Init
            if not Path(pathSemiAuto).exists():
                print (f" - [ERROR][getRefineDCMs()] pathSemiAuto: {pathSemiAuto} does not exist")
            
            # Step 2.2 - Get the .dcm files
            pathsDcmsSemiAuto = list(pathSemiAuto.glob('**/*.dcm'))
            if len(pathsDcmsSemiAuto) == 0:
                print (f" - [ERROR][getRefineDCMs()] No .dcm files found in pathSemiAuto: {pathSemiAuto}")
                pdb.set_trace()
                continue
            pathsDcmsSemiAuto = sorted(pathsDcmsSemiAuto, key=extractCounter)

            # Step 2.3 - Get the patientId (and handle >1 experiments on the same patientId)
            patientId = pathsDcmsSemiAuto[0].name.split(KEY_REFINE)[0]
            if patientId in patientIdCounterSemiAuto: patientIdCounterSemiAuto[patientId] +=1
            else                                     : patientIdCounterSemiAuto[patientId] = 1
            patientIdNew = patientId + PATIENT_COUNTER_SPLITTER + str(patientIdCounterSemiAuto[patientId])
            
            # Step 2.4 - Store the .dcm files
            patientsSemiAuto[patientIdNew] = pathsDcmsSemiAuto
            # print (f" - [INFO][getRefineDCMs(SemiAuto)] patientId: {patientIdNew}, pathsDcmsSemiAuto[0].name: {pathsDcmsSemiAuto[0].name}")
     
    except:
        traceback.print_exc()
        pdb.set_trace()
    
    # return sorted(patientsManual), sorted(patientsSemiAuto)
    return patientsManual, patientsSemiAuto

def eval(pathsManualExperiments, pathsSemiAutoExperiments, userName, cropLastSlices, useGT):
     
    try:

        # Step 0 - Init
        resDICE = {} # filled in Step 2 and used in Step 5
        resArrays = {}
        spacing = None
        # surfaceDICETolerances = [1,2,3]
        surfaceDICETolerances = [2]

        # Step 0.1 - Print the settings
        print (' \n ===================================== \n')
        print (' ------ Data related')
        print (' - doAllIters    : ', doAllIters)
        print (' - cropLastSlices: ', cropLastSlices)
        print (' - useGT         : ', useGT)
        print (' - userName      : ', userName)
        print (' - surfaceDICETolerances: ', surfaceDICETolerances)
        print (' - iovStuff      : ', iovStuff)
        print ('\n ------ Plot related')
        print (' - plotFigs     : ', plotFigs)
        print (' - useScrollData: ', useScrollData)
        print (' - showTitle    : ', showTitle, ' (set this to False for paper)')
        print (' - makeRender   : ', makeRender)
        print (' - makeVideo    : ', makeVideo, ' (this slows down the script!)')
        print (' - FIG_DPI      : ', FIG_DPI)
        print (' \n ===================================== \n')

        doPyvistaBool = testPyvista() and makeRender

        # Step 1 - Get the manual and semi-auto experiments
        print (' - [src/backend/utils/evalUtils.py][eval()] Step 1.0 - Get .dcm paths ...')
        patientsManual, patientsSemiAuto = getRefineDCMs(pathsManualExperiments, pathsSemiAutoExperiments)
        allPatientIds = sorted(list(set(list(patientsManual.keys()) + list(patientsSemiAuto.keys()))))
          
        # Step 2 - Get GT/Pred
        tStart = time.time()
        for patientIdx, patientId in enumerate(allPatientIds):
            startPatientTime = time.time()

            ## -------------------------------------- Step 2.0 - Init
            if 1:
                resDICE[patientId] = {
                                    KEY_INTERACTIONCOUNT_MANUAL:[], KEY_INTERACTIONCOUNT_SEMIAUTO:[]
                                    , KEY_MANUAL_DICE:[], KEY_SEMIAUTO_DICE:[]
                                    , KEY_MANUAL_SURFACE_DICE_1MM: [], KEY_MANUAL_SURFACE_DICE_2MM: [], KEY_MANUAL_SURFACE_DICE_3MM: []
                                    , KEY_SEMIAUTO_SURFACE_DICE_1MM: [], KEY_SEMIAUTO_SURFACE_DICE_2MM: [], KEY_SEMIAUTO_SURFACE_DICE_3MM: []
                                    , KEY_ACTION_START_EPOCH_MANUAL: [], KEY_ACTION_START_EPOCH_SEMIAUTO: []
                                    , KEY_TIME_ACTION_SECONDS_MANUAL: [], KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL: [], KEY_TIME_ACTION_SECONDS_SEMIAUTO: [], KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO: []
                                    , KEY_TIME_ACTION_SECONDS_MANUAL_CUMM: [], KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM: [], KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM: [], KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM: []
                                    , KEY_FILEPATHS_VIZ_3D_AI: [], KEY_FILEPATHS_VIZ_3D_MANUAL: []
                                    }
                arrayGTForManual, arrayGTForSemiAuto, baseDICE, baseSurfaceDICE = None, None, None, None
                resArrays[patientId] = {KEY_MANUAL: None, KEY_SEMIAUTO: None, KEY_MANUAL_STEPS: -1, KEY_SEMIAUTO_STEPS: -1}

                # Step 2.1 - Some .dcm and Orthanc stuff
                print (f" - [src/backend/utils/evalUtils.py][eval()] Step 2.1 - Get .dcm and Orthanc stuff for patientId: {patientId} ...")
                patientIDOrthanc   = patientId.split(PATIENT_COUNTER_SPLITTER)[0]
                patientIdObj       = orthancRequestUtils.getOrthancPatientIds(patientIDOrthanc)
                if len(patientIdObj) == 0:
                    print (f"   --> [ERROR][eval()] patientIdObj is empty for patientId: {patientId}")
                    pdb.set_trace()
                    continue
                arrayGT, arrayPred = orthancRequestUtils.getSegsArray(patientIDOrthanc, patientIdObj) # [axial, coronal, sagittal]
                # if useGT:
                #     renderAndSaveSurface(arrayGT, arrayPred, filePath= DIR_EXPERIMENT_VIDEOS / f'surface__{patientId}__view-{{}}.png')
                # plotMasks(patientId, arrayGT, arrayPred, 'arrayGT', 'arrayPred', 62, 'arrayGT-arrayPred-fromOrthanc')
                # plotMasks(patientId, arrayGT, arrayPred, 'arrayGT', 'arrayPred', 82, 'arrayGT-arrayPred-fromOrthanc')
                # plotMasks(patientId, arrayGT, arrayPred, 'arrayGT', 'arrayPred', 95, 'arrayGT-arrayPred-fromOrthanc')
                    
                if np.sum(arrayGT) == 0 or np.sum(arrayPred) == 0 or arrayGT is None or arrayPred is None:
                    print (f"   --> [ERROR][eval()] arrayGT or arrayPred is empty for patientId: {patientId}")
                    pdb.set_trace()
                    continue
                if cropLastSlices:
                    arrayGTSliceIdxs = np.argwhere(np.sum(arrayGT, axis=(1,2))).flatten() # sum over coronal and sagittal
                    arrayGTStartSliceIdx, arrayGTStartendIdx = arrayGTSliceIdxs[1], arrayGTSliceIdxs[-2]
                    print (f"   --> [INFO][eval()] arrayGTStartSliceIdx: {arrayGTStartSliceIdx} arrayGTStartendIdx: {arrayGTStartendIdx}")
                else:
                    arrayGTStartSliceIdx, arrayGTStartendIdx = 0, arrayGT.shape[2]-1
                
                # render_and_save_surface(arrayGT, arrayPred, filename=f'./_tmp/surface_render_{patientId}.png')
                # pdb.set_trace()

                listObjsCT         = orthancRequestUtils.getPyDicomObjects(patientIDOrthanc, patientIdObj, orthancRequestUtils.MODALITY_CT)
                spacing            = list([float(each) for each in listObjsCT[0].PixelSpacing]) + [float(listObjsCT[0].SliceThickness)]

                # Step 2.2 - Get the GT and Pred masks (and compute DICE for step=0)
                if useGT:
                    baseDICE = compute_dice(arrayGT[:,:,arrayGTStartSliceIdx:arrayGTStartendIdx+1], arrayPred[:,:,arrayGTStartSliceIdx:arrayGTStartendIdx+1], meta=f"{patientId} - GT/Pred")
                    baseSurfaceDICE = getSurfaceDICE(arrayGT[:,:,arrayGTStartSliceIdx:arrayGTStartendIdx+1], arrayPred[:,:,arrayGTStartSliceIdx:arrayGTStartendIdx+1], spacing, tolerancesMM=surfaceDICETolerances)
                    resDICE[patientId][KEY_MANUAL_DICE].append(baseDICE)
                    resDICE[patientId][KEY_SEMIAUTO_DICE].append(baseDICE)
                    for surfaceDICETolerance in surfaceDICETolerances:
                        resDICE[patientId][KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance)].append(baseSurfaceDICE[surfaceDICETolerance])
                        resDICE[patientId][KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance)].append(baseSurfaceDICE[surfaceDICETolerance])
                    print (f"   --> [INFO][eval()] baseDICE: {baseDICE:.2f}, baseSurfaceDICE: {['{:.2f}'.format(baseSurfaceDICE[surfaceDICETolerance]) for surfaceDICETolerance in surfaceDICETolerances]}")
                    # pdb.set_trace()
                    resDICE[patientId][KEY_INTERACTIONCOUNT_MANUAL].append(0)
                    resDICE[patientId][KEY_INTERACTIONCOUNT_SEMIAUTO].append(0)
                    resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL].append(0)
                    resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL].append(0)
                    resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM].append(0)
                    resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM].append(0)
                    resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO].append(0)
                    resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO].append(0)
                    resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM].append(0)
                    resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM].append(0)
                else:
                    pass # in case we don't have GT, we refer to the last edited mask as GT (different in case of manual and semi-auto)
                
                # Step 2.3 - Get {caseName}__slice-scroll.json for manual and semi-auto
                if useScrollData:
                    scrollJsonManual, scrollJsonSemiAuto = {}, {}
                    pathScrollJsonManual   = pathsManualExperiments[patientIdx] / f'{patientId.split('__')[0]}{SUFFIX_SLICE_SCROLL_JSON}'
                    pathScrollJsonSemiAuto = pathsSemiAutoExperiments[patientIdx] / f'{patientId.split('__')[0]}{SUFFIX_SLICE_SCROLL_JSON}'
                    if Path(pathScrollJsonManual).exists():
                        scrollJsonManual = json.load(open(pathScrollJsonManual))
                        scrollJsonManual = dict(list(scrollJsonManual[KEY_AXIAL].items()) + list(scrollJsonManual[KEY_CORONAL].items()) + list(scrollJsonManual[KEY_SAGITTAL].items()))
                    if Path(pathScrollJsonSemiAuto).exists():
                        scrollJsonSemiAuto = json.load(open(pathScrollJsonSemiAuto))
                        scrollJsonSemiAuto = dict(list(scrollJsonSemiAuto[KEY_AXIAL].items()) + list(scrollJsonSemiAuto[KEY_CORONAL].items()) + list(scrollJsonSemiAuto[KEY_SAGITTAL].items()))

            ## -------------------------------------- Step 3.1 - Read patientsManual[patientId] and compute DICE
            arrayGTForManual = None
            arrayListRefineForManual = []
            if 1 and patientId in patientsManual:
                try:
                    print (" - [src/backend/utils/evalUtils.py][eval()] Step 3.1 - Compute DICE for manual refinement ({} steps)...".format(len(patientsManual[patientId])))

                    # Step 3.1.0 - Get GT for manual brushes
                    if 1:
                        if useGT:
                            arrayGTForManual = arrayGT.copy() # [axial, coronal, sagittal]
                        else:
                            if useManualAsGT or useThisAsGT:
                                arrayGTForManual, seriesDescForManual, _, _, _ = orthancRequestUtils.getSegArrayInShapeMismatchScene(listObjsCT, patientsManual[patientId][-1]) # [axial, coronal, sagittal]
                                assert arrayGTForManual.sum() > 0, f"   --> [ERROR][eval()] arrayGTForManual.sum() is 0 for patientId: {patientId}"
                                # plotMasks(patientId, arrayGT, arrayGTForManual, 'arrayGT', 'arrayGTForManual', 62, filenamePrefix='arrayGT-vs-arrayGTForManual')
                                # plotMasks(patientId, arrayGT, arrayGTForManual, 'arrayGT', 'arrayGTForManual', 82, filenamePrefix='arrayGT-vs-arrayGTForManual')
                                # plotMasks(patientId, arrayGT, arrayGTForManual, 'arrayGT', 'arrayGTForManual', 95, filenamePrefix='arrayGT-vs-arrayGTForManual')
                                # pdb.set_trace()
                            elif useAIAsGT:
                                dsForSemiAuto    = pydicom.dcmread(patientsSemiAuto[patientId][-1])
                                arrayGTForManual = dsForSemiAuto.pixel_array.astype(np.uint8) # [axial, coronal, sagittal]
                                seriesDecForSemiAuto = dsForSemiAuto.SeriesDescription
                                assert arrayGTForManual.sum() > 0, f" - [ERROR][eval()] arrayGTForManual.sum() is 0 for patientId: {patientId}"
                            
                            resArrays[patientId][KEY_MANUAL] = arrayGTForManual # only in the useGT=False scenario
                            resArrays[patientId][KEY_MANUAL_STEPS] = len(patientsManual[patientId])
                        
                        if doPyvistaBool:
                            filePath = renderAndSaveSurface(arrayGTForManual, arrayPred, userName, 'Manual', 0) # here arrayGTForManual is the final corrected mask (manually)
                            resDICE[patientId][KEY_FILEPATHS_VIZ_3D_MANUAL].append(filePath)
                        
                    # Step 3.1.1 - Loop over the manual-brush .dcm files
                    if doAllIters:
                        prevDiceManual, newDiceManual = 0, 0
                        prevTimeOfBrushStart, timeOfBrushStart = None, None
                        with tqdm.tqdm(total=len(patientsManual[patientId]), leave=True, desc='  |- [Manual: {}]'.format(patientId)) as pbarManual:
                            for fileId, pathDcmFile in enumerate(patientsManual[patientId]):
                                try:

                                    # Step 3.1.1 - Read manual-brush .dcm file (and also get timing information)
                                    if 1:
                                        prevTimeOfBrushStart = copy.deepcopy(timeOfBrushStart)
                                        maskArrayManualRefine, seriesDescManualRefine, timeToBrushFromDataset, timeOfBrushStart, epochOfBrushStart = orthancRequestUtils.getSegArrayInShapeMismatchScene(listObjsCT, pathDcmFile)
                                        arrayListRefineForManual.append(maskArrayManualRefine)
                                        # print (f"   --> [INFO][eval()] timeToBrushFromDataset: {timeToBrushFromDataset:.2f} timeOfBrushStart: {timeOfBrushStart}")
                                        if fileId == len(patientsManual[patientId])-1:
                                            getComponentCount(maskArrayManualRefine, meta='[Manual]')
                                    
                                    # Step 3.2.3 - Append timing information to global variables
                                    if 1:
                                        resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL].append(timeToBrushFromDataset)
                                        if timeToBrushFromDataset != -1:
                                            resDICE[patientId][KEY_ACTION_START_EPOCH_MANUAL].append(epochOfBrushStart)
                                            if fileId > 0:
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL].append((timeOfBrushStart - prevTimeOfBrushStart).seconds)
                                                resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM].append(resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM][-1] + resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL][-1])
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM].append(resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM][-1] + resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-1])
                                            else:
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL].append(0.1)
                                                resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM].append(timeToBrushFromDataset)
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM].append(0.1)
                                        else:
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL].append(-1)
                                            resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM].append(-1)
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM].append(-1)
                                        
                                        # Step 3.2.4 - Check for outliers in timings information
                                        if resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-1] > TIME_TOOMUCH:
                                            initVal     = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-1]
                                            initValCumm = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM][-1]
                                            print (f"   --> [ERROR][eval()] Too much time gap between brushes ({resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-1]}) vs {resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-5:]}")
                                            # resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1]      = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-2]
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-1]      = np.mean(resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-5:-1])
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM][-1] = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM][-2] + resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-1]
                                            newVal     = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][-1]
                                            newValCumm = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM][-1]
                                            print (f"     --> [ERROR][eval()] Corrected time gap between brushes ({initVal} -> {newVal}) || cumm ({initValCumm} -> {newValCumm})")

                                    # Step 3.1.5 - Compute metrics
                                    if 1:
                                        prevDiceManual    = copy.deepcopy(newDiceManual)
                                        newDiceManual     = compute_dice(maskArrayManualRefine[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:,], arrayGTForManual[arrayGTStartSliceIdx:arrayGTStartendIdx+1:,:], meta=f"{patientId} - {pathDcmFile}")
                                        surfaceDiceManual = getSurfaceDICE(maskArrayManualRefine[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], arrayGTForManual[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], spacing, tolerancesMM=[1,2,3])
                                        resDICE[patientId][KEY_MANUAL_DICE].append(newDiceManual)
                                        resDICE[patientId][KEY_INTERACTIONCOUNT_MANUAL].append(fileId+1)
                                        for surfaceDICETolerance in surfaceDICETolerances:
                                            resDICE[patientId][KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance)].append(surfaceDiceManual[surfaceDICETolerance])
                                        # print (f" - [INFO][eval()][{KEY_MANUAL_DICE}][id={fileId}] patientId: {patientId} id: {fileId} dice: {dice}")
                                        
                                        if fileId > 0:
                                            deltaDICEManual = newDiceManual - prevDiceManual
                                            # if deltaDICEManual < -0.01:
                                            #     print (f"   --> [ERROR][eval()][{KEY_MANUAL_DICE}][patientId={patientId}, idx={fileId}, dcmFile: {getName(pathDcmFile)}] deltaDICE: {deltaDICEManual:.2f}")

                                    # Step 3.1.6 - Do visualization
                                    if 1:
                                        if 0:
                                            plotMasks(patientId, arrayGTForManual, maskArrayManualRefine, 'arrayGTForManual', 'maskArrayManualRefine', 62, 'arrayGTForManual-vs-maskArrayManualRefine')
                                            plotMasks(patientId, arrayGTForManual, maskArrayManualRefine, 'arrayGTForManual', 'maskArrayManualRefine', 82, 'arrayGTForManual-vs-maskArrayManualRefine')
                                            plotMasks(patientId, arrayGTForManual, maskArrayManualRefine, 'arrayGTForManual', 'maskArrayManualRefine', 95, 'arrayGTForManual-vs-maskArrayManualRefine')
                                            pdb.set_trace()
                                            # sliceIdx = 63; plt.imshow(maskArrayManualRefine[:,:,sliceIdx], cmap='gray'); plt.imshow(arrayGTForManual[:,:,sliceIdx], alpha=0.5, cmap='gray');plt.show(block=False);pdb.set_trace()
                                        
                                        if doPyvistaBool:
                                            filePath = renderAndSaveSurface(arrayGTForManual, maskArrayManualRefine, userName, 'Manual', fileId+1)
                                            resDICE[patientId][KEY_FILEPATHS_VIZ_3D_MANUAL].append(filePath)

                                except:
                                    print (f"   --> [ERROR][eval()][{KEY_MANUAL_DICE}][id={fileId}] Error in patientId: {patientId} dcmFile: {getName(pathDcmFile)}")
                                    resDICE[patientId][KEY_MANUAL_DICE].append(-1)
                                    for surfaceDICETolerance in surfaceDICETolerances:
                                        resDICE[patientId][KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance)].append(-1)
                                    traceback.print_exc()
                                    pdb.set_trace()
                                
                                pbarManual.update(1)

                        if makeVideo:
                            pathVideoManual = DIR_EXPERIMENT_VIDEOS / userName / FILENAME_VIZ_VIDEO_FORMAT_MANUAL.format(userName, len(patientsManual[patientId])-1)
                            create_video(resDICE[patientId][KEY_FILEPATHS_VIZ_3D_MANUAL], pathVideoManual, fps=2, minFrames=len(resDICE[patientId][KEY_FILEPATHS_VIZ_3D_MANUAL]))
                
                except:
                    print (f"   --> [ERROR][eval()][Manual] Error in patientId: {patientId}")
                    traceback.print_exc()
                    pdb.set_trace()

            ## -------------------------------------- Step 3.2 - Read patientsSemiAuto[patientId] and compute DICE
            arrayGTForSemiAuto = None
            arrayListRefineForSemiAuto = []
            if 1 and patientId in patientsSemiAuto:
                try:
                    print (" - [src/backend/utils/evalUtils.py][eval()] Step 3.2 - Compute DICE for semi-auto refinement ({} steps)...".format(len(patientsSemiAuto[patientId])))
                    
                    # Step 3.2.0 - Get GT for AI-scribbles
                    if 1:
                        if useGT:
                            arrayGTForSemiAuto = arrayGT.copy() # [axial, coronal, sagittal]
                        else:
                            if useAIAsGT or useThisAsGT:
                                dsForSemiAuto = pydicom.dcmread(patientsSemiAuto[patientId][-1])
                                arrayGTForSemiAuto = dsForSemiAuto.pixel_array.astype(np.uint8) # [axial, coronal, sagittal]
                                seriesDecForSemiAuto = dsForSemiAuto.SeriesDescription
                                assert arrayGTForSemiAuto.sum() > 0, f" - [ERROR][eval()] arrayGTForSemiAuto.sum() is 0 for patientId: {patientId}"
                                # plotMasks(patientId, arrayGT, arrayGTForSemiAuto, 'arrayGT', 'arrayGTForSemiAuto', sliceIdx=62, filenamePrefix='arrayGT-vs-arrayGTForSemiAuto')
                                # plotMasks(patientId, arrayGT, arrayGTForSemiAuto, 'arrayGT', 'arrayGTForSemiAuto', sliceIdx=82, filenamePrefix='arrayGT-vs-arrayGTForSemiAuto')
                                # plotMasks(patientId, arrayGT, arrayGTForSemiAuto, 'arrayGT', 'arrayGTForSemiAuto', sliceIdx=95, filenamePrefix='arrayGT-vs-arrayGTForSemiAuto')
                                # pdb.set_trace()
                            elif useManualAsGT:
                                arrayGTForSemiAuto, seriesDescForManual, _, _, _ = orthancRequestUtils.getSegArrayInShapeMismatchScene(listObjsCT, patientsManual[patientId][-1]) # [axial, coronal, sagittal]
                                assert arrayGTForSemiAuto.sum() > 0, f"   --> [ERROR][eval()] arrayGTForSemiAuto.sum() is 0 for patientId: {patientId}"
                            
                            resArrays[patientId][KEY_SEMIAUTO] = arrayGTForSemiAuto # only in the useGT=False scenario
                            resArrays[patientId][KEY_SEMIAUTO_STEPS] = len(patientsSemiAuto[patientId])
                            
                        if doPyvistaBool:
                            filePath = renderAndSaveSurface(arrayGTForSemiAuto, arrayPred, userName, 'AI', 0)
                            resDICE[patientId][KEY_FILEPATHS_VIZ_3D_AI].append(filePath)

                    if doAllIters:
                        prevDiceSemiAuto, newDiceSemiAuto = 0, 0
                        prevTimeOfScribbleStart, timeOfScribbleStart = None, None
                        with tqdm.tqdm(total=len(patientsSemiAuto[patientId]), leave=True, desc='  |- [SemiAuto: {}]'.format(patientId)) as pbarSemiAuto:
                            for fileId, pathDcmFile in enumerate(patientsSemiAuto[patientId]):
                                try:

                                    # Step 3.2.1 - Read AI-scribble .dcm file
                                    if 1:
                                        ds                          = pydicom.dcmread(pathDcmFile)
                                        maskArrayForSemiAutoRefine  = ds.pixel_array
                                        arrayListRefineForSemiAuto.append(maskArrayForSemiAutoRefine)
                                        if fileId == len(patientsSemiAuto[patientId])-1:
                                            getComponentCount(maskArrayForSemiAutoRefine, meta='[Semi-Auto]')
                                        seriesDescForSemiAutoRefine = ds.SeriesDescription

                                    # Step 3.2.2 - Get timing information
                                    if 1:
                                        prevTimeOfScribbleStart     = copy.deepcopy(timeOfScribbleStart)
                                        try:
                                            timeToScribble      = ds[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_SCRIBBLE)].value
                                            tDistMap            = ds[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_DISTMAP)].value
                                            tInfer              = ds[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_INFER)].value
                                            tMakeSegDCM         = 0.7 # [NOTE: Assuming for now]
                                            totalAITimeOnServer = tDistMap + tInfer + tMakeSegDCM
                                            totalAITimeOnClient = 2.0 # [NOTE: Assuming for now]
                                            scribbleStartEpoch  = int(ds[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_EPOCH)].value)
                                            timeOfScribbleStart = datetime.datetime.fromtimestamp(scribbleStartEpoch/1000.0)
                                            # print (f"   --> [INFO][eval()] timeToScribble: {timeToScribble:.2f} | tDistMap: {tDistMap:.2f} + tInfer: {tInfer:.2f} + tMakeSegDCM: {tMakeSegDCM:.2f} = {totalAITimeOnServer:.2f} |  timeOfScribbleStart: {timeOfScribbleStart}")
                                        except:
                                            timeToScribble, tDistMap, tInfer, scribbleStartEpoch = -1, -1, -1, -1
                                        resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO].append(timeToScribble)

                                        # Step 3.2.3 - Append timing information to global variables
                                        if timeToScribble != -1:
                                            resDICE[patientId][KEY_ACTION_START_EPOCH_SEMIAUTO].append(scribbleStartEpoch)
                                            if fileId > 0:
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO].append(float((timeOfScribbleStart - prevTimeOfScribbleStart).seconds) - totalAITimeOnClient)
                                                resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM].append(resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM][-1] + resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO][-1])
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM].append(resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM][-1] + resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1])
                                            else:
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO].append(0.1)
                                                resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM].append(timeToScribble)
                                                resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM].append(0.1)
                                        else:
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO].append(-1)
                                            resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM].append(-1)
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM].append(-1)
                                        
                                        # Step 3.2.4 - Check for outliers in timings information
                                        if resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1] > TIME_TOOMUCH:
                                            initVal     = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1]
                                            initValCumm = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM][-1]
                                            print (f"   --> [ERROR][eval()] Too much time gap between scribbles ({resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1]}) vs {resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-5:]}")
                                            # resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1]      = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-2]
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1]      = np.mean(resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-5:-1])
                                            resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM][-1] = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM][-2] + resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1]
                                            newVal     = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][-1]
                                            newValCumm = resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM][-1]
                                            print (f"     --> [ERROR][eval()] Corrected time gap between scribbles ({initVal} -> {newVal}) || cumm ({initValCumm} -> {newValCumm})")
                                        
                                    # Step 3.2.5 - Compute metrics
                                    if 1:
                                        prevDiceSemiAuto = copy.deepcopy(newDiceSemiAuto)
                                        newDiceSemiAuto        = compute_dice(maskArrayForSemiAutoRefine[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], arrayGTForSemiAuto[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], meta=f"{patientId} - {pathDcmFile}")
                                        newSurfaceDiceSemiAuto = getSurfaceDICE(maskArrayForSemiAutoRefine[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], arrayGTForSemiAuto[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], spacing, tolerancesMM=[1,2,3])
                                        resDICE[patientId][KEY_SEMIAUTO_DICE].append(newDiceSemiAuto)
                                        resDICE[patientId][KEY_INTERACTIONCOUNT_SEMIAUTO].append(fileId+1)
                                        for surfaceDICETolerance in surfaceDICETolerances:
                                            resDICE[patientId][KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance)].append(newSurfaceDiceSemiAuto[surfaceDICETolerance])
                                        if fileId > 0:
                                            deltaDICESemiAuto = newDiceSemiAuto - prevDiceSemiAuto
                                            # if deltaDICESemiAuto < -0.01:
                                            #     print (f"   --> [ERROR][eval()][{KEY_SEMIAUTO_DICE}][patientId={patientId}, idx={fileId}, dcmFile: {getName(pathDcmFile)}] deltaDICE: {deltaDICESemiAuto:.2f}")

                                    # Step 3.2.6 - Do vizualization
                                    if 1:
                                        if 0:
                                            plotMasks(patientId, arrayGTForSemiAuto, maskArrayForSemiAutoRefine, 'arrayGTForSemiAuto', 'maskArrayForSemiAutoRefine', 62, filenamePrefix='arrayGTForSemiAuto-vs-maskArrayForSemiAutoRefine')
                                            plotMasks(patientId, arrayGTForSemiAuto, maskArrayForSemiAutoRefine, 'arrayGTForSemiAuto', 'maskArrayForSemiAutoRefine', 82, filenamePrefix='arrayGTForSemiAuto-vs-maskArrayForSemiAutoRefine')
                                            plotMasks(patientId, arrayGTForSemiAuto, maskArrayForSemiAutoRefine, 'arrayGTForSemiAuto', 'maskArrayForSemiAutoRefine', 95, filenamePrefix='arrayGTForSemiAuto-vs-maskArrayForSemiAutoRefine')
                                            pdb.set_trace()

                                        if doPyvistaBool:
                                            filePath = renderAndSaveSurface(arrayGTForSemiAuto, maskArrayForSemiAutoRefine, userName, 'AI', fileId+1)
                                            resDICE[patientId][KEY_FILEPATHS_VIZ_3D_AI].append(filePath)

                                except:
                                    print (f"   --> [ERROR][eval()][{KEY_SEMIAUTO_DICE}][patientId={patientId}, idx={fileId}, dcmFile: {getName(pathDcmFile)}]")
                                    resDICE[patientId][KEY_SEMIAUTO_DICE].append(-1)
                                    for surfaceDICETolerance in surfaceDICETolerances:
                                        resDICE[patientId][KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance)].append(-1)
                                    traceback.print_exc()
                                    pdb.set_trace()
                            
                                pbarSemiAuto.update(1)
                    
                        if makeVideo:
                            pathVideoAI = DIR_EXPERIMENT_VIDEOS / userName / FILENAME_VIZ_VIDEO_FORMAT_AI.format(userName, len(resDICE[patientId][KEY_FILEPATHS_VIZ_3D_AI])-1)
                            create_video(resDICE[patientId][KEY_FILEPATHS_VIZ_3D_AI], pathVideoAI, fps=2, minFrames=len(resDICE[patientId][KEY_FILEPATHS_VIZ_3D_MANUAL])) # at least as many as manual frames

                except:
                    print (f" - [ERROR][eval()][Semi-Auto] Error in patientId: {patientId}")
                    traceback.print_exc()
                    pdb.set_trace()
                
            ## -------------------------------------- Step 3.3 - Store the scroll data
            if 1:
                resDICE[patientId][KEY_ACTION_SCROLL_DATA_MANUAL] = {}
                resDICE[patientId][KEY_ACTION_SCROLL_DATA_SEMIAUTO] = {}
                if useScrollData:
                    if len(scrollJsonManual) > 0:
                        scrollCountPerActionManual = Counter(np.digitize(sorted([int(each) for each in scrollJsonManual.keys()]), resDICE[patientId][KEY_ACTION_START_EPOCH_MANUAL]))
                        scrollCountPerActionManual = dict(sorted(scrollCountPerActionManual.items()))
                        resDICE[patientId][KEY_ACTION_SCROLL_DATA_MANUAL] = scrollCountPerActionManual # values from [0, max(resDICE[patientId][KEY_INTERACTIONCOUNT_MANUAL])-1],
                        print (f"   --> [INFO][eval()] max(scrollCountPerActionManual.values()): {max(scrollCountPerActionManual.values())}")
                    if len(scrollJsonSemiAuto) > 0:
                        scrollCountPerActionSemiAuto = Counter(np.digitize(sorted([int(each) for each in scrollJsonSemiAuto.keys()]), resDICE[patientId][KEY_ACTION_START_EPOCH_SEMIAUTO]))
                        scrollCountPerActionSemiAuto = dict(sorted(scrollCountPerActionSemiAuto.items()))
                        resDICE[patientId][KEY_ACTION_SCROLL_DATA_SEMIAUTO] = scrollCountPerActionSemiAuto
                        print (f"   --> [INFO][eval()] max(scrollCountPerActionSemiAuto.values()): {max(scrollCountPerActionSemiAuto.values())}")

            ## -------------------------------------- Step 4 - Some other stuff
            if 1 and patientId in patientsManual and patientId in patientsSemiAuto:
                if baseDICE is None and arrayGTForManual is not None and arrayGTForSemiAuto is not None:
                    
                    # Step 4.1 - Compare manual vs AI-assisted
                    if not useGT:
                        try:
                            
                            # Compare arrayGTForManual (coronal, sagittal, axial) with arrayGTForSemiAuto (axial, coronal, sagittal)
                            if useThisAsGT:
                                baseDICE        = compute_dice(arrayGTForManual[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], arrayGTForSemiAuto[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], meta=f"{patientId} - GT/Pred")
                                baseSurfaceDICE = getSurfaceDICE(arrayGTForManual[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], arrayGTForSemiAuto[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], spacing, tolerancesMM=[1,2,3])
                                print (f" - [INFO][eval(useGT={useGT})][patientId: {patientId}] baseDICE: {baseDICE}")
                                print (f" - [INFO][eval(useGT={useGT})][patientId: {patientId}] baseSurfaceDICE: {baseSurfaceDICE}")
                                getComponentCount(arrayGTForManual[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], meta='[Manually Refined Array]')
                                getComponentCount(arrayGTForSemiAuto[arrayGTStartSliceIdx:arrayGTStartendIdx+1,:,:], meta='[AI-Assisted Array]')
                            # plotMasks(patientId, arrayGTForManual, arrayGTForSemiAuto, 'arrayGTForManual', 'arrayGTForSemiAuto', sliceIdx=62, filenamePrefix='arrayGTForManual-vs-arrayGTForSemiAuto')
                            # plotMasks(patientId, arrayGTForManual, arrayGTForSemiAuto, 'arrayGTForManual', 'arrayGTForSemiAuto', sliceIdx=82, filenamePrefix='arrayGTForManual-vs-arrayGTForSemiAuto')
                            # plotMasks(patientId, arrayGTForManual, arrayGTForSemiAuto, 'arrayGTForManual', 'arrayGTForSemiAuto', sliceIdx=95, filenamePrefix='arrayGTForManual-vs-arrayGTForSemiAuto')
                        except:
                            traceback.print_exc()

                    # Step 4.2 - 3D visualize manual vs AI-assisted together
                    if doPyvistaBool:

                        # Step 4.2.1 - Visualize manual and AI-refined masks (so no GT)
                        try:
                            noGTFilePaths = []
                            totalFrames = max(len(arrayListRefineForManual), len(arrayListRefineForSemiAuto))
                            print (f" - [INFO][eval()] Doing noGT viz for totalFrames: {totalFrames-1}")
                            with tqdm.tqdm(total=totalFrames, leave=False) as pbarNoGT:
                                for frameId in range(max(len(arrayListRefineForManual), len(arrayListRefineForSemiAuto))):
                                    try:
                                        maskArrayManualRefine      = arrayListRefineForManual[frameId]   if frameId < len(arrayListRefineForManual)   else arrayListRefineForManual[-1]
                                        maskArrayForSemiAutoRefine = arrayListRefineForSemiAuto[frameId] if frameId < len(arrayListRefineForSemiAuto) else arrayListRefineForSemiAuto[-1]
                                        # filePath = renderAndSaveSurface(maskArrayManualRefine, maskArrayForSemiAutoRefine, userName, 'NoGT', frameId, arrayGTColor='blue', arrayPredColor='orange')
                                        filePath = renderAndSaveSurface(maskArrayManualRefine, maskArrayForSemiAutoRefine, userName, 'C{}'.format(patientIdx+1) + '-NoGT', frameId, arrayGTColor='blue', arrayPredColor=[250, 97, 2])
                                        noGTFilePaths.append(filePath)
                                    except:
                                        traceback.print_exc()
                                        pdb.set_trace()
                                    
                                    pbarNoGT.update(1)
                            
                            # Step 4.2.2 - Create video (without GT)
                            if makeVideo:
                                pathVideoNoGT = DIR_EXPERIMENT_VIDEOS / userName / FILENAME_VIZ_VIDEO_FORMAT_NOGT.format(userName, len(noGTFilePaths)-1)
                                create_video(noGTFilePaths, pathVideoNoGT, fps=2, minFrames=len(noGTFilePaths))

                        except:
                            traceback.print_exc()
                            pdb.set_trace()

            print (' - \n Time for patient: {:.2f}s (Total time={:.2f}s) \n'.format(time.time()-startPatientTime, time.time()-tStart))

        ## -------------------------------------- Step 5 - Plot
        if plotFigs and arrayGTForManual is not None and arrayGTForSemiAuto is not None and doAllIters:
            print ('\n\n ----------------------------------------------- [src/backend/utils/evalUtils.py][eval()] Step 4 - Plot ...')
            print ('  -> [INFO][eval()] allPatientIds: ', allPatientIds)

            if 1:
                # Step 5.0.1 - Init (define matplolib variables)
                FIGSIZE = (15, 8)
                FONTSIZE_XTICKS = 18 # [10, 13, 15, 18]
                FONTSIZE_YTICKS = 18 # [10, 13, 15, 18]
                FONTSIZE_XLABEL = 22 # [13, 18, 22]
                FONTSIZE_YLABEL = 22 # [13, 18, 22]
                FONTSIZE_LEGEND = 13 # []
                
                if useGT:
                    PLT_YLIM = (0.6, 1.025)
                    FONTSIZE_XTICKS = 22 # [10, 13, 15, 18, 22]
                    FONTSIZE_YTICKS = 22 # [10, 13, 15, 18, 22]
                    FONTSIZE_XLABEL = 25 # [13, 18, 22]
                    FONTSIZE_YLABEL = 25 # [13, 18, 22]
                    FONTSIZE_LEGEND = 25 # [13,25]
                else:
                    FONTSIZE_XTICKS = 22 # [10, 13, 15, 18, 22]
                    FONTSIZE_YTICKS = 22 # [10, 13, 15, 18, 22]
                    FONTSIZE_XLABEL = 25 # [13, 18, 22]
                    FONTSIZE_YLABEL = 25 # [13, 18, 22]
                    FONTSIZE_LEGEND = 25 # [13,25]
                    if 'CHMR005' in userName:
                        PLT_YLIM = (0.9, 1.025) # CHMR005 (Session 2)
                    elif 'CHMR023' in userName:
                        PLT_YLIM = (0.80, 1.025) # CHMR023 (Session 3)
                    elif 'CHMR016' in userName:
                        PLT_YLIM = (0.90, 1.025) # CHMR016 (Session 4)
                    elif 'CHUP-033' in userName: # P1
                        PLT_YLIM = (0.70, 1.025)
                    elif 'CHUP-059' in userName: # P2
                        PLT_YLIM = (0.85, 1.025)
                    elif 'CHUP-005' in userName: # P3
                        PLT_YLIM = (0.7, 1.025)
                    elif 'CHUP-064' in userName: # P4
                        PLT_YLIM = (0.60, 1.025)
                    elif 'CHUP-028' in userName: # P4
                        PLT_YLIM = (0.60, 1.025)
                    elif 'CHUP-044' in userName: # P6
                        PLT_YLIM = (0.70, 1.025)
                    else:
                        print (f"   --> [INFO][eval()] userName: Custom PLT_YLIM not found for {userName} not found")
                        PLT_YLIM = (0.80, 1.025)

                # PLT_YLIM = (0.7, 1.05)
                PLT_XLIM_INSEC_ACTION      = (0,700)
                PLT_XLIM_INSEC_LAST_ACTION = (0,900)
                SECOND_INTERVAL = 2
                PERC_INTERVAL = 0.5 # i.e., total of 200 points (from 0% to 100%)

                # Step 5.0.2 - Make a list of all metrics
                interactionCountMetricsManual = [KEY_INTERACTIONCOUNT_MANUAL]
                interactionCountMetricsSemiAuto = [KEY_INTERACTIONCOUNT_SEMIAUTO]
                timeMetricsManual   = [KEY_TIME_ACTION_SECONDS_MANUAL_CUMM, KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM]
                timeMetricsSemiAuto = [KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM, KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM]
                diceMetricsManual   = [KEY_MANUAL_DICE]   + [KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance) for surfaceDICETolerance in surfaceDICETolerances]
                diceMetricsSemiAuto = [KEY_SEMIAUTO_DICE] + [KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance) for surfaceDICETolerance in surfaceDICETolerances]
                diceMetricsAll      = diceMetricsManual + diceMetricsSemiAuto

                # Step 5.1.0 - Convert 1 x resDICE into 3 x dataframes (for 3 graphs)
                def convertObjToDatframe(obj, diceMetrics, timeMetric=None, timeMetricPlotName=None):
                    """
                    Parameters:
                        obj: dict, {patientId: {metricKey: [val1, val2, ...]}}
                        diceMetrics: list
                        timeMetric: str
                        timeMetricPlotName: str
                    """

                    df = None
                    dfData = []
                    try:
                        
                        for patientId in obj:
                            
                            # Step 1 - Get scroll data (on the basis of manual vs semi-auto)
                            if not percInterp and useScrollData:
                                if KEY_MANUAL_DICE in diceMetrics:
                                    patientScrollObj = obj[patientId][KEY_ACTION_SCROLL_DATA_MANUAL]
                                elif KEY_SEMIAUTO_DICE:
                                    patientScrollObj = obj[patientId][KEY_ACTION_SCROLL_DATA_SEMIAUTO]
                            else:
                                patientScrollObj = {}

                            # Step 2 - Loop over metrics in obj i.e. 
                            for metricKey in obj[patientId]:
                                try:
                                    if metricKey not in diceMetrics:
                                        continue
                                    for idx, val in enumerate(obj[patientId][metricKey]):
                                        # dfDataThis = [patientId, metricKey, idx, val, patientScrollObj.get(idx, -1)]
                                        # dfDataThis = [patientId, metricKey, idx, val, patientScrollObj.get(idx, 0) + 1] # patientScrollObj = {actionId: scrollCount, ...} e.g. Counter({0: 302, 1: 74, 17: 61, 12: 60, 8: 49, 18: 43, 16: 32, 15: 24, 4: 6, 3: 3, 14: 1})
                                        scrollCountThisIdx = patientScrollObj.get(idx, 0) + 1
                                        scrollCountThisIdxNorm = min(scrollCountThisIdx / MAX_SCROLL_COUNT, 1)
                                        dfDataThis = [patientId, metricKey, idx, val, scrollCountThisIdx, scrollCountThisIdxNorm]

                                        if timeMetric is not None:
                                            dfDataThis.append(obj[patientId][timeMetric][idx])  
                                        dfData.append(dfDataThis)
                                except:
                                    traceback.print_exc()
                                    pdb.set_trace()
                        
                        dfColumnNames = [KEY_PATIENTID, KEY_METRIC, KEY_INTERACTIONCOUNT, KEY_VALUE, KEY_SCROLLS, KEY_SCROLLS_NORM]
                        if timeMetric is not None and timeMetricPlotName is not None:
                            dfColumnNames.append(timeMetricPlotName)
                        df = pd.DataFrame(dfData, columns=dfColumnNames)

                    except:
                        traceback.print_exc()
                        pdb.set_trace()

                    return df
                
                # Step 5.1.1 - Convert resDICE into dataframes x 3 (for 3 graphs)
                resDICENew = {}
                resDICENewTimeForAction = {}
                resDICENewTimeFromLastAction = {}
                for patientId in resDICE:
                    resDICENew[patientId]                   = {KEY_MANUAL_DICE: [], KEY_SEMIAUTO_DICE: [], KEY_ACTION_SCROLL_DATA_MANUAL: resDICE[patientId][KEY_ACTION_SCROLL_DATA_MANUAL], KEY_ACTION_SCROLL_DATA_SEMIAUTO: resDICE[patientId][KEY_ACTION_SCROLL_DATA_SEMIAUTO]}
                    resDICENewTimeForAction[patientId]      = {KEY_MANUAL_DICE: [], KEY_SEMIAUTO_DICE: [], KEY_ACTION_SCROLL_DATA_MANUAL: resDICE[patientId][KEY_ACTION_SCROLL_DATA_MANUAL], KEY_ACTION_SCROLL_DATA_SEMIAUTO: resDICE[patientId][KEY_ACTION_SCROLL_DATA_SEMIAUTO]}
                    resDICENewTimeFromLastAction[patientId] = {KEY_MANUAL_DICE: [], KEY_SEMIAUTO_DICE: [], KEY_ACTION_SCROLL_DATA_MANUAL: resDICE[patientId][KEY_ACTION_SCROLL_DATA_MANUAL], KEY_ACTION_SCROLL_DATA_SEMIAUTO: resDICE[patientId][KEY_ACTION_SCROLL_DATA_SEMIAUTO]}
                    for surfaceDICETolerance in surfaceDICETolerances:
                        resDICENew[patientId][KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance)] = []
                        resDICENewTimeForAction[patientId][KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance)] = []
                        resDICENewTimeForAction[patientId][KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance)] = []
                        resDICENewTimeFromLastAction[patientId][KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance)] = []
                        resDICENewTimeFromLastAction[patientId][KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance)] = []
                    
                    maxStepsForManual  = len(resDICE[patientId][KEY_MANUAL_DICE]) - 1
                    maxStepsForSemiAuto = len(resDICE[patientId][KEY_SEMIAUTO_DICE]) - 1
                    maxStepsForPatient  = max(maxStepsForManual, maxStepsForSemiAuto)
                    print (f"   --> [INFO][eval()][{patientId}] maxStepsForManual             : {maxStepsForManual} | maxStepsForSemiAuto: {maxStepsForSemiAuto} | maxStepsForPatient: {maxStepsForPatient}")
                    
                    maxTimeActionForManual   = max(resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM])
                    maxTimeActionForSemiAuto = max(resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM])
                    maxTimeActionForPatient  = max(maxTimeActionForManual, maxTimeActionForSemiAuto)
                    print (f"   --> [INFO][eval()][{patientId}] maxTimeActionForManual        : {maxTimeActionForManual:.2f} | maxTimeActionForSemiAuto: {maxTimeActionForSemiAuto:.2f} | maxTimeActionForPatient: {maxTimeActionForPatient:.2f}")
                    
                    maxTimeFromLastActionForManual   = max(resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM])
                    maxTimeFromLastActionForSemiAuto = max(resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM])
                    maxTimeFromLastActionForPatient  = max(maxTimeFromLastActionForManual, maxTimeFromLastActionForSemiAuto)
                    print (f"   --> [INFO][eval()][{patientId}] maxTimeFromLastActionForManual: {maxTimeFromLastActionForManual:.2f} | maxTimeFromLastActionForSemiAuto: {maxTimeFromLastActionForSemiAuto:.2f} | maxTimeFromLastActionForPatient: {maxTimeFromLastActionForPatient:.2f}")
                    print ('')

                    # Step 1.5.2 - For manual (time metrics)
                    for xAxis in timeMetricsManual + interactionCountMetricsManual:
                        
                        # Step 1.5.2.1 - Interpolate for xAxis
                        xValsNew    = resDICE[patientId][xAxis] # xValsNew = [t1, t2, ..., tn] for timeMetricsManual and [step1, step2, ..., stepn] for interactionCountMetricsManual
                        if percInterp: 
                            if xAxis in interactionCountMetricsManual:
                                xValsPerc = list(np.array(xValsNew)/maxStepsForPatient*100) # now all values between [0,100] though not linearly spaced
                            elif xAxis == KEY_TIME_ACTION_SECONDS_MANUAL_CUMM: 
                                xValsPerc = list(np.array(xValsNew)/maxTimeActionForPatient*100) # now all values between [0,100] though not linearly spaced
                            elif xAxis == KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM: 
                                xValsPerc = list(np.array(xValsNew)/maxTimeFromLastActionForPatient*100)
                            # xValsNew = np.arange(xValsPerc[0], xValsPerc[-1], PERC_INTERVAL) # every 0.5% (200 points)
                            xValsNew = np.arange(0, xValsPerc[-1], PERC_INTERVAL) # every 0.5% (200 points)
                            # print (' - patientId: ', patientId, ' || xAxis: ', xAxis, ' || xValsNew: ', xValsNew)
                        if xAxis in interactionCountMetricsManual                   : resDICENew[patientId][xAxis]                     = xValsNew
                        elif xAxis == KEY_TIME_ACTION_SECONDS_MANUAL_CUMM           : resDICENewTimeForAction[patientId][xAxis]        = xValsNew
                        elif xAxis == KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM : resDICENewTimeFromLastAction[patientId][xAxis]   = xValsNew

                        # Step 1.5.2.2 - Interpolate for yAxis
                        for metric in diceMetricsManual:
                            yVals    = resDICE[patientId][metric]
                            if percInterp:  yValsNew = np.interp(xValsNew, xValsPerc, yVals)
                            else         :  yValsNew = yVals
                            if xAxis in interactionCountMetricsManual                  : resDICENew[patientId][metric]                   = yValsNew
                            elif xAxis == KEY_TIME_ACTION_SECONDS_MANUAL_CUMM          : resDICENewTimeForAction[patientId][metric]      = yValsNew
                            elif xAxis == KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM: resDICENewTimeFromLastAction[patientId][metric] = yValsNew
                            
                    # Step 1.5.3 - For semi-auto (time metrics)
                    for xAxis in timeMetricsSemiAuto + interactionCountMetricsSemiAuto:
                        
                        # Step 1.5.3.1 - Interpolate for xAxis
                        xValsNew    = resDICE[patientId][xAxis] # xValsNew = [t1, t2, ..., tn]
                        if percInterp: 
                            if xAxis in interactionCountMetricsSemiAuto:
                                xValsPerc = list(np.array(xValsNew)/maxStepsForPatient*100)
                            elif xAxis == KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM: 
                                xValsPerc = list(np.array(xValsNew)/maxTimeActionForPatient*100) # now all values between [0,100] though not linearly spaced
                            elif xAxis == KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM: 
                                xValsPerc = list(np.array(xValsNew)/maxTimeFromLastActionForPatient*100)
                            # xValsNew = np.arange(xValsPerc[0], xValsPerc[-1], PERC_INTERVAL) # every 0.5% (200 points)
                            xValsNew = np.arange(0, xValsPerc[-1], PERC_INTERVAL) # every 0.5% (200 points)
                            
                        if xAxis in interactionCountMetricsSemiAuto                  : resDICENew[patientId][xAxis]                   = xValsNew
                        elif xAxis == KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM          : resDICENewTimeForAction[patientId][xAxis]      = xValsNew
                        elif xAxis == KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM: resDICENewTimeFromLastAction[patientId][xAxis] = xValsNew

                        # Step 1.5.3.2 - Interpolate for yAxis
                        for metric in diceMetricsSemiAuto:
                            yVals    = resDICE[patientId][metric] # [metric@t1, metric@t2, ..., metric@tn]
                            if percInterp: yValsNew = np.interp(xValsNew, xValsPerc, yVals)
                            else         : yValsNew = yVals
                            if xAxis in interactionCountMetricsSemiAuto                  : resDICENew[patientId][metric]                   = yValsNew
                            elif xAxis == KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM          : resDICENewTimeForAction[patientId][metric]      = yValsNew
                            elif xAxis == KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM: resDICENewTimeFromLastAction[patientId][metric] = yValsNew
        
                # Step 5.1.2 - Convert resDICE into dataframes x 3 (outputs --> dfInteractionCount, dfTimeForAction, dfTimeFromLastAction)
                if percInterp:
                    # len(resDICENew[patientId][KEY_MANUAL_DICE]) == len(resDICENew[patientId][KEY_INTERACTIONCOUNT_MANUAL])
                    # len(resDICENew[patientId][KEY_SEMIAUTO_DICE]) == len(resDICENew[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM])
                    dfInteractionCountManual   = convertObjToDatframe(resDICENew, diceMetricsManual  , KEY_INTERACTIONCOUNT_MANUAL, KEY_INTERACTIONCOUNT_PERC)
                    dfInteractionCountSemiAuto = convertObjToDatframe(resDICENew, diceMetricsSemiAuto, KEY_INTERACTIONCOUNT_SEMIAUTO, KEY_INTERACTIONCOUNT_PERC)
                    dfInteractionCount         = pd.concat([dfInteractionCountManual, dfInteractionCountSemiAuto])
                else:
                    # dfInteractionCount = convertObjToDatframe(resDICE, diceMetricsAll, None, None) # NOTE: Incorrect
                    dfInteractionCountManual   = convertObjToDatframe(resDICENew, diceMetricsManual  , None, None)
                    dfInteractionCountSemiAuto = convertObjToDatframe(resDICENew, diceMetricsSemiAuto, None, None)
                    dfInteractionCount         = pd.concat([dfInteractionCountManual, dfInteractionCountSemiAuto])

                
                if doTimeStuff:
                    if percInterp:
                        dfTimeForActionManual   = convertObjToDatframe(resDICENewTimeForAction, diceMetricsManual, KEY_TIME_ACTION_SECONDS_MANUAL_CUMM, KEY_TIME_ACTION_CUMM_PERC)
                        dfTimeForActionSemiAuto = convertObjToDatframe(resDICENewTimeForAction, diceMetricsSemiAuto, KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM, KEY_TIME_ACTION_CUMM_PERC)
                    else:
                        dfTimeForActionManual   = convertObjToDatframe(resDICENewTimeForAction, diceMetricsManual, KEY_TIME_ACTION_SECONDS_MANUAL_CUMM, KEY_TIME_ACTION_CUMM)
                        dfTimeForActionSemiAuto = convertObjToDatframe(resDICENewTimeForAction, diceMetricsSemiAuto, KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM, KEY_TIME_ACTION_CUMM)
                    dfTimeForAction         = pd.concat([dfTimeForActionManual, dfTimeForActionSemiAuto])
                    
                    if percInterp:
                        dfTimeFromLastActionManual = convertObjToDatframe(resDICENewTimeFromLastAction, diceMetricsManual, KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM_PERC)    
                        dfTimeFromLastActionSemiAuto = convertObjToDatframe(resDICENewTimeFromLastAction, diceMetricsSemiAuto, KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM_PERC)
                    else:
                        dfTimeFromLastActionManual   = convertObjToDatframe(resDICENewTimeFromLastAction, diceMetricsManual, KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM)
                        dfTimeFromLastActionSemiAuto = convertObjToDatframe(resDICENewTimeFromLastAction, diceMetricsSemiAuto, KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM)
                    dfTimeFromLastAction         = pd.concat([dfTimeFromLastActionManual, dfTimeFromLastActionSemiAuto])
                
                # Step 5.2.0 - Plot (Init)
                if percInterp:
                    xAxisAndDfObj = {KEY_INTERACTIONCOUNT_PERC: dfInteractionCount}
                else:
                    xAxisAndDfObj = {KEY_INTERACTIONCOUNT: dfInteractionCount}
                if doTimeStuff:
                    if percInterp:
                        xAxisAndDfObj[KEY_TIME_ACTION_CUMM_PERC]           = dfTimeForAction
                        xAxisAndDfObj[KEY_TIME_FROM_LAST_ACTION_CUMM_PERC] = dfTimeFromLastAction
                    else:
                        xAxisAndDfObj[KEY_TIME_ACTION_CUMM]           = dfTimeForAction
                        xAxisAndDfObj[KEY_TIME_FROM_LAST_ACTION_CUMM] = dfTimeFromLastAction
                # xAxisAndDfObj = {
                #     KEY_INTERACTIONCOUNT: dfInteractionCount, KEY_TIME_ACTION_CUMM: dfTimeForAction, KEY_TIME_FROM_LAST_ACTION_CUMM: dfTimeFromLastAction
                # }
                maxDICEManual  = dfInteractionCount[dfInteractionCount[KEY_METRIC] == KEY_MANUAL_DICE][KEY_VALUE].max()
                maxDICESemiAuto = dfInteractionCount[dfInteractionCount[KEY_METRIC] == KEY_SEMIAUTO_DICE][KEY_VALUE].max()

            ## -------------------------------------- Step 5.2.0 - Plot
            if percInterp:
                xAxisNames = [KEY_INTERACTIONCOUNT_PERC]
            else:
                xAxisNames = [KEY_INTERACTIONCOUNT]
            if doTimeStuff:
                if percInterp:
                    xAxisNames += [KEY_TIME_ACTION_CUMM_PERC, KEY_TIME_FROM_LAST_ACTION_CUMM_PERC]
                else:
                    xAxisNames += [KEY_TIME_ACTION_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM]
            
            for xAxis in xAxisNames:
                print (f"\n   --> [INFO][eval()] xAxis: {xAxis}")
                try:
                    # Step 5.2.1 - Basic data
                    df            = xAxisAndDfObj[xAxis]
                    dfDICE        = df[df[KEY_METRIC].isin([KEY_MANUAL_DICE, KEY_SEMIAUTO_DICE])]
                    dfSurfaceDICE = df[df[KEY_METRIC].isin([KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance) for surfaceDICETolerance in surfaceDICETolerances] + [KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance) for surfaceDICETolerance in surfaceDICETolerances])]
                    
                    # Step 5.2.2 - Plot lines for DICE and surfaceDICE
                    fig, ax = plt.subplots(figsize=FIGSIZE)
                    if 0 and useScrollData:
                        sns.lineplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfDICE       , palette=SEABORN_PALETE, linestyle='solid' , legend=showLegend) # , marker='o'
                        sns.lineplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfSurfaceDICE, palette=SEABORN_PALETE, linestyle='dashed', legend=showLegend) # , marker='o'
                    else:
                        sns.lineplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfDICE       , palette=SEABORN_PALETE, marker='o', linestyle='solid' , legend=showLegend, markersize=5) # 
                        sns.lineplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfSurfaceDICE, palette=SEABORN_PALETE, marker='o', linestyle='dashed', legend=showLegend, markersize=5) # 
                    plt.axhline(y=maxDICEManual, color=SEABORN_PALETE[KEY_MANUAL_DICE], linestyle='dotted')
                    plt.axhline(y=maxDICESemiAuto, color=SEABORN_PALETE[KEY_SEMIAUTO_DICE], linestyle='dotted')

                    # Step 5.2.3 - Plot scatter points for DICE(only)
                    if useScrollData:
                        keyScrollNormsV2 = np.array(dfDICE[KEY_SCROLLS_NORM].tolist())
                        keyScrollNormsV2[keyScrollNormsV2 < 0.5] = 0.001 # all scrolls less than MAX_SCROLL_COUNT/2 are considered as MAX_SCROLL_COUNT*0.001
                        # sizesScrollScatterPoints = [int(MULTIPLIER_SCATTER_SIZE*each) for each in dfDICE[KEY_SCROLLS_NORM]] # dfDICE[KEY_SCROLLS_NORM] = [0.1, 0.2, 0.3, ...1.0]
                        sizesScrollScatterPoints = [int(MULTIPLIER_SCATTER_SIZE*each) for each in keyScrollNormsV2] # dfDICE[KEY_SCROLLS_NORM] = [0.1, 0.2, 0.3, ...1.0]
                        dfDICE[KEY_SCROLLS_NORM_PLOT] = sizesScrollScatterPoints
                        print (f"   --> [INFO][eval()] sizesScrollScatterPoints: max={max(sizesScrollScatterPoints)} : {sizesScrollScatterPoints}")
                        # pdb.set_trace()
                        # sns.scatterplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfDICE       , palette=SEABORN_PALETE, legend=False, size=KEY_SCROLLS, sizes=(10, 400))
                        # sns.scatterplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfDICE       , palette=SEABORN_PALETE, legend=False, sizes=sizesScrollScatterPoints)
                        # sns.scatterplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfDICE, palette=SEABORN_PALETE, legend=False, size=KEY_SCROLLS_NORM, sizes=(MIN_SCATTER_SIZE, MIN_SCATTER_SIZE+MULTIPLIER_SCATTER_SIZE)) # works
                        sns.scatterplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfDICE, palette=SEABORN_PALETE, legend=False, size=KEY_SCROLLS_NORM_PLOT, sizes=(1,MULTIPLIER_SCATTER_SIZE), alpha=0.5) # works

                    # Step 5.2.4 - Axes stuff (fixed)
                    plt.grid()
                    plt.xticks(fontsize=FONTSIZE_XTICKS)
                    plt.yticks(fontsize=FONTSIZE_YTICKS)
                    plt.xlabel(xAxis, fontsize=FONTSIZE_XLABEL)
                    plt.ylabel(KEY_VALUE, fontsize=FONTSIZE_YLABEL)
                    
                    # Step 5.2.5 - Axes stuff (dynamic)
                    if 1:
                        plt.ylim(*PLT_YLIM)
                        if not singleEval:
                            if xAxis == KEY_TIME_FROM_LAST_ACTION_CUMM:
                                plt.xlim(*PLT_XLIM_INSEC_LAST_ACTION)
                            elif xAxis == KEY_TIME_ACTION_CUMM:
                                plt.xlim(*PLT_XLIM_INSEC_ACTION)
                        if xAxis == KEY_INTERACTIONCOUNT:
                            plt.xticks(ticks=range(int(df[xAxis].min()), int(df[xAxis].max()) + 1, 10))
                        elif xAxis in [KEY_TIME_ACTION_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM]:
                            if singleEval:
                                plt.xticks(ticks=range(0, int(df[xAxis].max()) + 1, 60)) # i.e. 60sec = 1 minute
                            else:
                                if xAxis == KEY_TIME_ACTION_CUMM:
                                    plt.xticks(ticks=range(0, PLT_XLIM_INSEC_ACTION[-1] + 1, 60))
                                elif xAxis == KEY_TIME_FROM_LAST_ACTION_CUMM:
                                    plt.xticks(ticks=range(0, PLT_XLIM_INSEC_LAST_ACTION[-1] + 1, 60))

                    # Step 5.2.4 - Title
                    if 1:
                        titleStr = f"Manual vs Semi-automated Contour Refinement \n cropLastSlices: {cropLastSlices} || GT: {useGT} || TIME_TOOMUCH: {TIME_TOOMUCH} || percInterp: {percInterp}"
                        if not useGT and not useThisAsGT:
                            if useAIAsGT:
                                titleStr += ' (AI as GT)'
                            elif useManualAsGT:
                                titleStr += ' (Manual as GT)'
                        titleStr += f"\n PatientIds: {allPatientIds}"
                        if not privacyMode:
                            titleStr += '\n userName: {}'.format(userName)
                        
                        if showTitle:
                            plt.title(titleStr)

                    # Step 5.2.3 - Save
                    if 1:
                        plt.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)
                        useGTStr = ''
                        if useGT:
                            useGTStr = str(True)
                        else:
                            useGTStr = str(False)
                            if useAIAsGT:
                                useGTStr += '-AI-GT'
                            elif useManualAsGT:
                                useGTStr += '-Manual-GT'

                        interpStr = 'interp{}'
                        if percInterp:
                            interpStr = interpStr.format('Perc')
                        else:
                            interpStr = interpStr.format('None')
                        pathFig = str(Path(DIR_EXPERIMENTS_OUTPUTS) / ('{}__{}__title{}__cropped{}__timeMax{}s__useGT{}__xAxis-{}__manual_vs_semiauto.png'.format(userName, interpStr, showTitle, cropLastSlices, TIME_TOOMUCH, useGTStr, XAXIS_FILEIDENTIFER[xAxis])))
                        print (f"\n\n  -> [INFO][eval()] Saving the plot to: '{pathFig}'\n")
                        Path(pathFig).parent.mkdir(parents=True, exist_ok=True)
                        plt.savefig(pathFig, dpi=FIG_DPI, bbox_inches='tight')
                        # plt.show(block=False)
                
                except:
                    traceback.print_exc()
                    pdb.set_trace()
            
            if 0:
                resPlot = []
                for patientId in allPatientIds:
                    for fileId in range(len(patientsManual[patientId])):
                        # timingInfoForFileIdManual = [resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL][fileId], resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL][fileId], resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM][fileId], resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM][fileId]]
                        timingInfoForFileIdManual = [resDICE[patientId][KEY_TIME_ACTION_SECONDS_MANUAL_CUMM][fileId], resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_MANUAL_CUMM][fileId]]
                        resPlot.append([patientId, KEY_MANUAL_DICE, fileId, resDICE[patientId][KEY_MANUAL_DICE][fileId]] + timingInfoForFileIdManual)
                        for surfaceDICETolerance in surfaceDICETolerances:
                            resPlot.append([patientId, KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance), fileId, resDICE[patientId][KEY_MANUAL_SURFACE_DICE.format(surfaceDICETolerance)][fileId]] + timingInfoForFileIdManual)
                    for fileId in range(len(patientsSemiAuto[patientId])):
                        # timingInfoForFileIdSemiAuto = [resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO][fileId], resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO][fileId], resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM][fileId], resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM][fileId]]
                        timingInfoForFileIdSemiAuto = [resDICE[patientId][KEY_TIME_ACTION_SECONDS_SEMIAUTO_CUMM][fileId], resDICE[patientId][KEY_TIME_FROM_LAST_ACTION_SECONDS_SEMIAUTO_CUMM][fileId]]
                        resPlot.append([patientId, KEY_SEMIAUTO_DICE, fileId, resDICE[patientId][KEY_SEMIAUTO_DICE][fileId]] + timingInfoForFileIdSemiAuto)
                        for surfaceDICETolerance in surfaceDICETolerances:
                            resPlot.append([patientId, KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance), fileId, resDICE[patientId][KEY_SEMIAUTO_SURFACE_DICE.format(surfaceDICETolerance)][fileId]] + timingInfoForFileIdSemiAuto)

                # Step 4.1 - Get data
                # df     = pd.DataFrame(resPlot, columns=[KEY_PATIENTID, KEY_METRIC, KEY_INTERACTIONCOUNT, KEY_VALUE, KEY_TIME_ACTION, KEY_TIME_FROM_LAST_ACTION, KEY_TIME_ACTION_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM])
                df     = pd.DataFrame(resPlot, columns=[KEY_PATIENTID, KEY_METRIC, KEY_INTERACTIONCOUNT, KEY_VALUE, KEY_TIME_ACTION_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM])
                dfMain = df[df[KEY_METRIC].isin([KEY_MANUAL_DICE, KEY_SEMIAUTO_DICE])]
                dfSec  = df[~df[KEY_METRIC].isin([KEY_MANUAL_DICE, KEY_SEMIAUTO_DICE])]
                dfAgg = df.groupby(KEY_METRIC)[KEY_VALUE].agg(['max', 'min'])
                maxDICEManual   = dfAgg.loc[KEY_MANUAL_DICE, 'max']
                maxDICESemiAuto = dfAgg.loc[KEY_SEMIAUTO_DICE, 'max']
                minDICEManual   = dfAgg.loc[KEY_MANUAL_DICE, 'min']
                minDICESemiAuto = dfAgg.loc[KEY_SEMIAUTO_DICE, 'min']

                try:
                    
                    for xAxis in [KEY_INTERACTIONCOUNT, KEY_TIME_ACTION_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM]:
                        fig, ax = plt.subplots(figsize=FIGSIZE)
                        sns.lineplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfMain, palette=SEABORN_PALETE, linestyle='solid')
                        sns.lineplot(x=xAxis, y=KEY_VALUE, hue=KEY_METRIC, data=dfSec, palette=SEABORN_PALETE, linestyle='dashed')
                        # sns.boxplot(x=KEY_INTERACTIONCOUNT, y=KEY_VALUE, hue=KEY_METRIC, data=df, palette=SEABORN_PALETE)
                        plt.axhline(y=maxDICEManual, color=SEABORN_PALETE[KEY_MANUAL_DICE], linestyle='dashed', alpha=0.5)
                        plt.axhline(y=maxDICESemiAuto, color=SEABORN_PALETE[KEY_SEMIAUTO_DICE], linestyle='dashed', alpha=0.5)
                        plt.axhline(y=minDICEManual, color=SEABORN_PALETE[KEY_MANUAL_DICE], linestyle='dotted', alpha=0.5)
                        plt.axhline(y=minDICESemiAuto, color=SEABORN_PALETE[KEY_SEMIAUTO_DICE], linestyle='dotted', alpha=0.5)
                        
                        plt.ylim(*PLT_YLIM)
                        if not singleEval:
                            if xAxis == KEY_TIME_FROM_LAST_ACTION_CUMM:
                                plt.xlim(*PLT_XLIM_INSEC_LAST_ACTION)
                            elif xAxis == KEY_TIME_ACTION_CUMM:
                                plt.xlim(*PLT_XLIM_INSEC_ACTION)
                        titleStr = f"Manual vs Semi-automated Contour Refinement \n cropLastSlices: {cropLastSlices} || GT: {useGT}"
                        titleStr += f"\n PatientIds: {allPatientIds}"
                        if not privacyMode:
                            titleStr += '\n userName: {}'.format(userName)
                        plt.title(titleStr)
                        
                        
                        if xAxis == KEY_INTERACTIONCOUNT:
                            plt.xticks(ticks=range(int(df[xAxis].min()), int(df[xAxis].max()) + 1, 10))
                        elif xAxis in [KEY_TIME_ACTION_CUMM, KEY_TIME_FROM_LAST_ACTION_CUMM]:
                            if singleEval:
                                plt.xticks(ticks=range(0, int(df[xAxis].max()) + 1, 60)) # i.e. 60sec = 1 minute
                            else:
                                if xAxis == KEY_TIME_ACTION_CUMM:
                                    plt.xticks(ticks=range(0, PLT_XLIM_INSEC_ACTION[-1] + 1, 60))
                                elif xAxis == KEY_TIME_FROM_LAST_ACTION_CUMM:
                                    plt.xticks(ticks=range(0, PLT_XLIM_INSEC_LAST_ACTION[-1] + 1, 60))
                        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_YTICKS)  # Adjust the labelsize as needed
                        plt.legend(loc='lower right', fontsize=FONTSIZE_LEGEND)

                        print (f"  -> [INFO][eval()] Saving the plot to: '{pathFig}'")
                        plt.savefig(pathFig, dpi=FIG_DPI, bbox_inches='tight')
                        # plt.show(block=False)

                except:
                    traceback.print_exc()
                    pdb.set_trace()
                # pdb.set_trace()

        # Step 6 - Compare inter-annotator agreement (for manual vs semi-auto)
        if iovStuff:
            print ("\n  --> [INFO][eval()] Inter-observer variability stuff for {}".format(userName))
            try:
                # Step 6.1 - Get average manual and semi-auto arrays (across clinicians)
                if 1:
                    avgArrayManual, avgArraySemiAuto = np.zeros_like(arrayGTForManual), np.zeros_like(arrayGTForSemiAuto)
                    for refinementType in [KEY_MANUAL, KEY_SEMIAUTO]:
                        for patientId in resArrays: # here patientId will have an id with it to indicate the clinician ID
                            if refinementType == KEY_MANUAL:
                                avgArrayManual += resArrays[patientId][KEY_MANUAL]
                            elif refinementType == KEY_SEMIAUTO:
                                avgArraySemiAuto += resArrays[patientId][KEY_SEMIAUTO]

                    avgArrayManual   = np.round(avgArrayManual/len(resArrays)).astype(np.uint8)
                    avgArraySemiAuto = np.round(avgArraySemiAuto/len(resArrays)).astype(np.uint8)

                # Step 6.2 - Compare average arrays to each clinicians arrays
                if 1:
                    avgDICEManual, avgDICESemiAuto = [], []
                    avgSDiceManual, avgSDiceSemiAuto = {surfaceDICETolerance: [] for surfaceDICETolerance in surfaceDICETolerances}, {surfaceDICETolerance: [] for surfaceDICETolerance in surfaceDICETolerances}
                    interObserverDices, interObserverSurfaceDices = {}, {}
                    
                    for refinementType in [KEY_MANUAL, KEY_SEMIAUTO]:
                        interObserverDices[refinementType], interObserverSurfaceDices[refinementType] = {}, {}
                        for patientId in resArrays:
                            interObserverDices[refinementType][patientId] = []
                            interObserverSurfaceDices[refinementType][patientId] = {surfaceDICETolerance: [] for surfaceDICETolerances in surfaceDICETolerances}

                            # Compare resArrays[patientId][refinementType] with with all other resArrays[patientIdOther][refinementType]
                            thisArray = resArrays[patientId][refinementType]
                            for patientIdOther in resArrays:
                                if patientId != patientIdOther:
                                    otherArray = resArrays[patientIdOther][refinementType]
                                    interObserverDices[refinementType][patientId].append(compute_dice(thisArray, otherArray))
                                    surfaceDICEs = getSurfaceDICE(thisArray, otherArray, spacing, surfaceDICETolerances)
                                    for surfaceDICETolerance in surfaceDICETolerances: interObserverSurfaceDices[refinementType][patientId][surfaceDICETolerance].append(surfaceDICEs[surfaceDICETolerance])
                            
                            if refinementType == KEY_MANUAL:
                                avgDICEManual.append(compute_dice(resArrays[patientId][KEY_MANUAL], avgArrayManual))
                                surfaceDICEs = getSurfaceDICE(resArrays[patientId][KEY_MANUAL], avgArrayManual, spacing, surfaceDICETolerances)
                                for surfaceDICETolerance in surfaceDICETolerances: avgSDiceManual[surfaceDICETolerance].append(surfaceDICEs[surfaceDICETolerance])
                            elif refinementType == KEY_SEMIAUTO:
                                avgDICESemiAuto.append(compute_dice(resArrays[patientId][KEY_SEMIAUTO], avgArraySemiAuto))
                                surfaceDICEs = getSurfaceDICE(resArrays[patientId][KEY_SEMIAUTO], avgArraySemiAuto, spacing, surfaceDICETolerances)
                                for surfaceDICETolerance in surfaceDICETolerances: avgSDiceSemiAuto[surfaceDICETolerance].append(surfaceDICEs[surfaceDICETolerance])
                
                    interObserverDicesManual = np.median([each for patientId in interObserverDices[KEY_MANUAL] for each in interObserverDices[KEY_MANUAL][patientId]])
                    interObserverDicesSemiAuto = np.median([each for patientId in interObserverDices[KEY_SEMIAUTO] for each in interObserverDices[KEY_SEMIAUTO][patientId]])
                    interObserverSurfaceDicesManual = {surfaceDICETolerance: np.median([each for patientId in interObserverSurfaceDices[KEY_MANUAL] for each in interObserverSurfaceDices[KEY_MANUAL][patientId][surfaceDICETolerance]]) for surfaceDICETolerance in surfaceDICETolerances}
                    interObserverSurfaceDicesSemiAuto = {surfaceDICETolerance: np.median([each for patientId in interObserverSurfaceDices[KEY_SEMIAUTO] for each in interObserverSurfaceDices[KEY_SEMIAUTO][patientId][surfaceDICETolerance]]) for surfaceDICETolerance in surfaceDICETolerances}
                    print (f"  -> [INFO][eval()] interObserverDicesManual: {interObserverDicesManual:.2f}")
                    print (f"  -> [INFO][eval()] interObserverDicesSemiAuto: {interObserverDicesSemiAuto:.2f}")
                    roundedInterObserverSurfaceDicesManual = {k: round(v, 2) for k, v in interObserverSurfaceDicesManual.items()}
                    print (f"  -> [INFO][eval()] interObserverSurfaceDicesManual: {roundedInterObserverSurfaceDicesManual}")
                    roundedInterObserverSurfaceDicesSemiAuto = {k: round(v, 2) for k, v in interObserverSurfaceDicesSemiAuto.items()}
                    print (f"  -> [INFO][eval()] interObserverSurfaceDicesSemiAuto: {roundedInterObserverSurfaceDicesSemiAuto}")
                
                # Step 6.3 - Print
                if 1:
                    print (f"  -> [INFO][eval()] patientIds: {allPatientIds}")
                    print (f"  -> [INFO][eval()] avgDICEManual: {[round(each,2) for each in avgDICEManual]} (mean={np.mean(avgDICEManual):.2f}, std={np.std(avgDICEManual):.2f})")
                    print (f"  -> [INFO][eval()] avgDICESemiAuto: {[round(each,2) for each in avgDICESemiAuto]} (mean={np.mean(avgDICESemiAuto):.2f}, std={np.std(avgDICESemiAuto):.2f})")
                    for surfaceDICETolerance in surfaceDICETolerances:
                        print (f"  -> [INFO][eval()] avgSDiceManual[{surfaceDICETolerance}mm]: {[round(each,2) for each in avgSDiceManual[surfaceDICETolerance]]} (mean={np.mean(avgSDiceManual[surfaceDICETolerance]):.2f}, std={np.std(avgSDiceManual[surfaceDICETolerance]):.2f})")
                        print (f"  -> [INFO][eval()] avgSDiceSemiAuto[{surfaceDICETolerance}mm]: {[round(each,2) for each in avgSDiceSemiAuto[surfaceDICETolerance]]} (mean={np.mean(avgSDiceSemiAuto[surfaceDICETolerance]):.2f}, std={np.std(avgSDiceSemiAuto[surfaceDICETolerance]):.2f})")

                # Step 6.4 - make 3D
                if 1:
                    for refinementType in [KEY_MANUAL, KEY_SEMIAUTO]:
                        for patientId_, patientId in enumerate(resArrays): # patientId_ here actually means clinician ID
                            if refinementType == KEY_MANUAL:
                                gtArray = avgArrayManual
                                steps   = resArrays[patientId][KEY_MANUAL_STEPS]
                            elif refinementType == KEY_SEMIAUTO:
                                gtArray = avgArraySemiAuto
                                steps   = resArrays[patientId][KEY_SEMIAUTO_STEPS]

                            predColor = None
                            gtColor = 'black' # 'green', 'black'
                            if refinementType == KEY_MANUAL    : predColor = 'blue'
                            elif refinementType == KEY_SEMIAUTO: predColor = 'orange'
                            _ = renderAndSaveSurface(gtArray, resArrays[patientId][refinementType], userName, 'C{}'.format(patientId_+1) + '-Avg-{}'.format(refinementType), steps, arrayGTColor=gtColor, arrayPredColor=predColor) # arrayGTColor is for the avgArray

                # pdb.set_trace()

            except:
                traceback.print_exc()
                pdb.set_trace()
        

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    pdb.set_trace()

if __name__ == "__main__":

    ## -------------------- Step 1 - Define the manual and semi-auto experiments --------------------
    
    # Data-related
    useGT         = True
    percInterp    = False # set to True only when a cohort of users are compared since we need a common x-axis. Otherwise, set to False when evaluating a single user
    doTimeStuff   = True # almost always set to True
    useScrollData = True
    doAllIters    = True # this should generally be True
    iovStuff      = True 

    # Meh
    useThisAsGT   = True # this should generally be True
    useManualAsGT = False
    useAIAsGT     = False
    
    # Plot-related
    singleEval  = True
    privacyMode = False
    showLegend  = True
    makeVideo   = False
    makeRender  = False
    showTitle   = True
    plotFigs    = False # this should generally be True
    FIG_DPI     = 200 # [300, 200]

    # Plot (scatter) related
    MAX_SCROLL_COUNT = 50 # used to norm values between [0,1]
    MIN_SCATTER_SIZE = 10
    MULTIPLIER_SCATTER_SIZE = 800

    # Dataset noise related
    TIME_TOOMUCH = 60 # seconds

    # CHMR001 (Session 1)
    if 0:
        doTimeStuff = False

        # CHMR028
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-10-15 12-06-39 -- gracious_torvalds__Prerak-Mody-NonExpert'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-10-15 12-17-23 -- happy_carver__Prerak-Mo0dy-NonExpert'
            ]
            userName = 'Prerak Mody (Test 0)'
        
        # ?? (Marius)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-10-24 13-43-05 -- interesting_mclaren__Marius-Staring-NonExpert'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-10-24 14-00-07 -- mystifying_mendeleev__Marius-Staring-NonExpert'
            ]
            userName = 'MariusStaring'
        
        # CHMR001 (Prerak)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-03 18-41-29 -- wonderful_bose__Prerak-GaussFiltered-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-03 18-31-01 -- keen_shaw__Prera-GaussFiltered-NonExpert-AI-based'
            ]
            userName = 'Prerak Mody'

        # CHMR001 (Mark Gooding)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-05 10-41-59 -- focused_neumann__Mark-Gooding-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-11-05 10-35-46 -- beautiful_perlman__Mark-Gooding-NonExpert-AI-based'
                DIR_EXPERIMENTS / '2024-11-05 10-28-56 -- vibrant_lederberg__Mark-Gooding-NonExpert-AI-based'
            ]
            userName = 'Mark(CHMR001)'

        # ----------------------------------------------------

        # CHMR001 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-04 15-30-35 -- naughty_hopper__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-04 16-04-41 -- inspiring_chaum__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'S1(CHMR001)-U1(Yauheniya)'

        # CHMR001 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-04 11-59-29 -- upbeat_ganguly__Faeze-Gholamiankhah-NonExpert-Manual'
                # , DIR_EXPERIMENTS / '2024-11-03 18-41-29 -- wonderful_bose__Prerak-GaussFiltered-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-04 12-15-25 -- angry_yonath__Faeze-Gholamiankhah-NonExpert-AI-based'
                # , DIR_EXPERIMENTS / '2024-11-03 18-31-01 -- keen_shaw__Prera-GaussFiltered-NonExpert-AI-based'
            ]
            userName = 'S1(CHMR001)-U2(Faeze)'
            # userName = 'Faeze-Prerak (CHMR001)'
        
        # CHMR001 (U3-Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-06 12-19-40 -- relaxed_colden__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-06 12-36-16 -- naughty_antonelli__Frank-Dankers -NonExpert-AI-based'
            ]
            # userName = 'Frank(CHMR001)'
            userName = 'S1(CHMR001)-U3(Frank)'
        
        # CHMR001 (U4-Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-06 09-17-44 -- kind_wright__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-06 09-29-25 -- exciting_mendel__Alex-Vieth-NonExpert-AI-based'
            ]
            # userName = 'Alex(CHMR001)'
            userName = 'S1(CHMR001)-U4(Alex)'
                
        # CHMR001 (U5-Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-06 14-05-02 -- vigorous_germain__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-06 14-21-14 -- busy_lamarr__Patrick-de Koning-NonExpert-AI-based'
            ]
            # userName = 'Patrick (CHMR001)'
            userName = 'S1(CHMR001)-U5(Patrick)'
        
        # CHMR001 (U6-Chinmay) --> Issue with rangeOfIdxs = list(range(startIdx, endIdx+1))
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-07 09-12-30 -- vibrant_carson__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-07 09-40-56 -- modest_albattani__Chinmay-Rao-NonExpert-AI-based'
            ]
            # userName = 'Chinmay (CHMR001)'
            userName = 'S1(CHMR001)-U6(Chinmay)'
        
        # CHMR001 (U7-Ruochen) # Partial TODO
        elif 0:
            pathsManualExperiments = [
                # DIR_EXPERIMENTS / '2024-11-06 10-33-57 -- dazzling_swartz__Ruochen-Gao-NonExpert-Manual'
                DIR_EXPERIMENTS / '2024-11-28 15-34-14 -- tender_black__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 15-44-46 -- jolly_greider__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'S1(CHMR001)-U7(Ruochen)'
        
        # CHMR001 (Faeze/Jenia/Alex/Frank/Patrick/Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-04 11-59-29 -- upbeat_ganguly__Faeze-Gholamiankhah-NonExpert-Manual'
                , DIR_EXPERIMENTS / '2024-11-04 15-30-35 -- naughty_hopper__Yauheniya-Makarevich-NonExpert-Manual'
                , DIR_EXPERIMENTS / '2024-11-06 09-17-44 -- kind_wright__Alex-Vieth-NonExpert-Manual'
                , DIR_EXPERIMENTS / '2024-11-06 12-19-40 -- relaxed_colden__Frank-Dankers-NonExpert-Manual'
                , DIR_EXPERIMENTS / '2024-11-06 14-05-02 -- vigorous_germain__Patrick-de Koning-NonExpert-Manual'
                , DIR_EXPERIMENTS / '2024-11-07 09-12-30 -- vibrant_carson__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-04 12-15-25 -- angry_yonath__Faeze-Gholamiankhah-NonExpert-AI-based'
                , DIR_EXPERIMENTS / '2024-11-04 16-04-41 -- inspiring_chaum__Yauheniya-Makarevich-NonExpert-AI-based'
                , DIR_EXPERIMENTS / '2024-11-06 09-29-25 -- exciting_mendel__Alex-Vieth-NonExpert-AI-based'
                , DIR_EXPERIMENTS / '2024-11-06 12-36-16 -- naughty_antonelli__Frank-Dankers -NonExpert-AI-based'
                , DIR_EXPERIMENTS / '2024-11-06 14-21-14 -- busy_lamarr__Patrick-de Koning-NonExpert-AI-based'
                , DIR_EXPERIMENTS / '2024-11-07 09-40-56 -- modest_albattani__Chinmay-Rao-NonExpert-AI-based'
            ]
            # userName = 'Ch-Pat (CHMR001)'
            userName = 'All(CHMR001)'

        # ----------------------------------------------------

        # CHMR001 (C1-Martin)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-26 14-00-00 -- beautiful_beaver__Martin-De Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-26 14-05-20 -- cool_dirac__Martin -De Jong-Expert-AI-based'
            ]
            userName = 'S1(CHMR001)-C1(Martin)'
            useGT = False

        # CHMR001 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-08 13-25-07 -- distracted_wright__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-08 13-34-44 -- dreamy_lamarr__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'S1(CHMR001)-C2(Mischa)'
            useGT = False
        
        # CHMR001 (Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 09-54-42 -- romantic_margulis__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 10-09-38 -- sad_neumann__Niels-den Haan-Expert-AI-based'
            ]
            userName = 'S1(CHMR001)-C3(Niels)'
            useGT = False
    
    # CHMR005 (Session 2)
    elif 0:
        
        # CHMR005 (U1-Yauheniya)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-07 08-15-11 -- wonderful_diffie__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-07 08-40-11 -- youthful_kare__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'S2(CHMR005)-U1(Yauheniya)'
            TIME_TOOMUCH = 10000
        
        # CHMR005 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-11 14-33-05 -- hungry_stonebraker__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-11-11 14-50-28 -- pedantic_rhodes__Faeze-Gholamiankhah-NonExpert-AI-based'
                # DIR_EXPERIMENTS / '2024-11-12 13-09-00 -- charming_perlman__Prerak-Mody-NonExpert-AI-based' # userName = 'FaezeM-PrerakA(CHMR005)'
                DIR_EXPERIMENTS / '2024-11-13 16-35-29 -- suspicious_kilby__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'S2(CHMR005)-U2(Faeze)'
            # userName = 'Faeze(CHMR005)-v2'
            # userName = 'Faeze(CHMR005)-forGroningen'
            # showLegend = False
            TIME_TOOMUCH = 10000
        
        # CHMR005 (U3-Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-13 10-40-19 -- quirky_davinci__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-13 10-58-28 -- laughing_keldysh__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'S2(CHMR005)-U3(Frank)'
            # userName = 'Frank(CHMR005)-forGroningen'
            # showLegend = False
        
        # CHMR005 (U4-Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-11 12-05-09 -- gifted_lalande__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-11-11 12-18-20 -- hardcore_chatelet__Alex-Vieth-NonExpert-AI-based' # Old
                DIR_EXPERIMENTS / '2024-11-14 09-02-56 -- laughing_bhaskara__Alex-Vieth-NonExpert-AI-based' # New
            ]
            # userName = 'Alex(CHMR005)-v2'
            userName = 'S2(CHMR005)-U4(Alex)'

        # CHMR005 (U5-Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-11 08-36-49 -- elegant_jones__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-13 12-10-42 -- peaceful_northcutt__Patrick-de Koning-NonExpert-AI-based'
                # DIR_EXPERIMENTS / '2024-11-11 08-51-46 -- clever_golick__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'S2(CHMR005)-U5(Patrick)'
        
        # CHMR005 (U6-Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-14 12-04-47 -- kind_satoshi__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-14 12-19-36 -- fervent_gates__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'S2(CHMR005)-U6(Chinmay)'

        # CHMR005 (U7-Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-14 09-55-15 -- nice_morse__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-14 10-08-22 -- gallant_engelbart__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'S2(CHMR005)-U7(Ruochen)'
        
        # CHMR005 (Non-experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-07 08-15-11 -- wonderful_diffie__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-11 14-33-05 -- hungry_stonebraker__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-13 10-40-19 -- quirky_davinci__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-11 12-05-09 -- gifted_lalande__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-11 08-36-49 -- elegant_jones__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-14 12-04-47 -- kind_satoshi__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-14 09-55-15 -- nice_morse__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-07 08-40-11 -- youthful_kare__Yauheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-13 16-35-29 -- suspicious_kilby__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-13 10-58-28 -- laughing_keldysh__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-14 09-02-56 -- laughing_bhaskara__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-13 12-10-42 -- peaceful_northcutt__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-14 12-19-36 -- fervent_gates__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-14 10-08-22 -- gallant_engelbart__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'S2(CHMR005)-NonExperts'
            percInterp = True
            TIME_TOOMUCH = 10000

        # ----------------------------------------------------

        # CHMR005 (C1-Martin)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-12 14-03-13 -- kind_montalcini__Martin-Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-12 14-20-42 -- reverent_euclid__Martin-Jong-Expert-AI-based'
            ]
            userName = 'S2(CHMR005)-C1(Martin)'
            useGT = False
            useThisAsGT   = True # this should generally be True
            useManualAsGT = False
            useAIAsGT     = False
        
        # CHMR005 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-14 13-05-18 -- hardcore_babbage__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-14 13-12-41 -- recursing_mirzakhani__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'S2(CHMR005)-C2(Mischa)'
            useGT         = False
            useThisAsGT   = True # this should generally be True
            useManualAsGT = False
            useAIAsGT     = False
            
        # CHMR005 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-20 13-01-39 -- interesting_franklin__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-20 13-09-52 -- eloquent_merkle__Niels-den Haan-Expert-AI-based'
            ]
            userName = 'S2(CHMR005)-C3(Niels)'
            useGT = False
            useThisAsGT   = True # this should generally be True
            useManualAsGT = False
            useAIAsGT     = False
        
        # CHMR005 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 08-04-20 -- elastic_booth__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 08-11-53 -- epic_hermann__Jos-Elbers-Expert-AI-based'
            ]
            userName = 'S2(CHMR005)-C4(Jos)'
            useGT = False
            useThisAsGT   = True # this should generally be True
            useManualAsGT = False
            useAIAsGT     = False
    
    # CHMR004 (No data)
    elif 0:
        
        # CHMR004 (Prerak)
        if 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-08 12-36-46 -- nostalgic_brown__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-08 11-50-17 -- awesome_shockley__Prerak-Mody-NonExpert-AI-based'
            ]
            userName = 'Prerak (CHMR004)'

    # CHMR023 (Session 3)
    elif 0:
        
        # CHMR023 (Prerak)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 08-26-07 -- vigorous_elgamal__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 08-41-42 -- agitated_meninsky__Prerak-Mody-NonExpert-AI-based'
            ]
            userName = 'Prerak(CHMR023)'
        
        # CHMR023 (C1-Martin)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 14-09-59 -- wizardly_payne__Martin-de Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 14-21-09 -- suspicious_babbage__Martin -de Jong-Expert-AI-based' # lots of other images available for this experiment.
            ]
            userName = 'S3(CHMR023)-C1(Martin)'
            useGT    = False
            TIME_TOOMUCH = 60 # seconds
            percInterp = True
        
        # CHMR023 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 10-50-25 -- bold_mendeleev__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 11-01-11 -- exciting_rubin__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'S3(CHMR023)-C1(Mischa)'
            useGT    = False
            TIME_TOOMUCH = 60 # seconds
            percInterp = True
        
        # CHMR023 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-20 13-21-01 -- elastic_fermi__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-20 13-32-01 -- thirsty_fermat__Niels-den Haan-Expert-AI-based'
            ]
            userName = 'S3(CHMR023)-C3(Niels)'
            useGT    = False
            TIME_TOOMUCH = 60 # seconds
            percInterp = True

        # CHMR023 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 08-07-19 -- intelligent_mirzakhani__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 08-16-02 -- fervent_lamarr__Jos-Elbers-Expert-AI-based'
            ]
            userName = 'S3(CHMR023)-C4(Jos)'
            useGT    = False
            TIME_TOOMUCH = 60 # seconds
            percInterp = True

        # CHMR023 (Experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 14-09-59 -- wizardly_payne__Martin-de Jong-Expert-Manual',
                DIR_EXPERIMENTS / '2024-11-18 10-50-25 -- bold_mendeleev__Mischa-de Ridder-Expert-Manual',
                DIR_EXPERIMENTS / '2024-11-20 13-21-01 -- elastic_fermi__Niels-den Haan-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-05 08-07-19 -- intelligent_mirzakhani__Jos-Elbers-Expert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 14-21-09 -- suspicious_babbage__Martin -de Jong-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-11-18 11-01-11 -- exciting_rubin__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-11-20 13-32-01 -- thirsty_fermat__Niels-den Haan-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-05 08-16-02 -- fervent_lamarr__Jos-Elbers-Expert-AI-based',
            ]
            userName = 'S3(CHMR023)-Experts'
            useGT    = False
            percInterp = True
            useScrollData = False
            TIME_TOOMUCH = 60
        # ----------------------------------------------------

        # CHMR023 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 11-59-22 -- hopeful_bohr__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 12-29-56 -- xenodochial_moore__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'S3(CHMR023)-U1(Yauheniya)'
            percInterp  = True
            privacyMode = True
        
        # CHMR023 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 12-52-27 -- inspiring_swartz__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 13-04-03 -- compassionate_murdock__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'S3(CHMR023)-U2(Faeze)'
            percInterp  = True
            privacyMode = True
        
        # CHMR023 (U3 - Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 10-09-14 -- vigorous_joliot__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 10-20-50 -- silly_sanderson__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'S3(CHMR023)-U3(Frank)'
            percInterp  = True
            privacyMode = True
        
        # CHMR023 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 13-22-47 -- vigilant_lovelace__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 13-34-33 -- zen_austin__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'S3(CHMR023)-U4(Alex)'
            percInterp  = True
            privacyMode = True
        
        # CHMR023 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 14-35-02 -- gallant_dewdney__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 14-51-06 -- loving_noyce__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'S3(CHMR023)-U5(Patrick)'
            percInterp  = True
            privacyMode = True
        
        # CHMR023 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 15-38-44 -- elastic_grothendieck__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 15-56-55 -- fervent_pare__Chinmay-Rao-NonExpert-AI-based' # NOTE: contains a lot of individual images!
            ]
            userName = 'S3(CHMR023)-U6(Chinmay)'
            percInterp  = True
            privacyMode = True
        
        # CHMR023 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 15-03-52 -- tender_beaver__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 15-20-09 -- boring_elgamal__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'S3(CHMR023)-U7(Ruochen)'
            percInterp  = True
            privacyMode = True

        # CHMR023 (Non-experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 11-59-22 -- hopeful_bohr__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-18 12-52-27 -- inspiring_swartz__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-27 10-09-14 -- vigorous_joliot__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-18 13-22-47 -- vigilant_lovelace__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-18 14-35-02 -- gallant_dewdney__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-19 15-38-44 -- elastic_grothendieck__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-28 15-03-52 -- tender_beaver__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-18 12-29-56 -- xenodochial_moore__Yauheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-18 13-04-03 -- compassionate_murdock__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-27 10-20-50 -- silly_sanderson__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-18 13-34-33 -- zen_austin__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-18 14-51-06 -- loving_noyce__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-19 15-56-55 -- fervent_pare__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-28 15-20-09 -- boring_elgamal__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'S3(CHMR023)-NonExperts'
            percInterp = True
            useScrollData = False
            TIME_TOOMUCH = 60

    # CHMR016 (Session 4)
    elif 0:

        # CHMR016 (C1-Martin)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-21 14-14-44 -- zealous_dijkstra__Martin-de Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-21 14-08-18 -- dazzling_morse__Martin-de Jong-Expert-AI-based' 
            ]
            userName = 'S4(CHMR016)-C1(Martin)'
            useGT    = False
            useThisAsGT   = True # this should generally be True
            useManualAsGT = False
            useAIAsGT     = False

        # CHMR016 (C2-Mischa) # Partial To-Do
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / 'Meh'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-19 09-09-50 -- ecstatic_bose__Mischa-de Ridder-Expert-AI-based' 
            ]
            userName = 'S4(CHMR016)-C2(Mishca)'
            useGT    = False
            useThisAsGT   = True # this should generally be True
            useManualAsGT = False
            useAIAsGT     = False
        
        # CHMR016 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 11-29-13 -- nostalgic_mcnulty__Niels -den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 11-35-11 -- ecstatic_ganguly__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'S4(CHMR016)-C3(Niels)'
            useGT    = False

        # CHMR016 (C4-Jos)
        elif 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 08-20-54 -- sleepy_zhukovsky__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 08-24-10 -- condescending_elgamal__Jos-Elbers-Expert-AI-based'
            ]
            userName = 'S4(CHMR016)-C4(Jos)'
            useGT = False
            useThisAsGT   = True # this should generally be True
            useManualAsGT = False
            useAIAsGT     = False

        # ----------------------------------------------------

        # CHMR016 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 09-02-04 -- vigorous_elbakyan__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 09-28-56 -- laughing_burnell__Yuheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'CHMR016-U1(Yauheniya)'
        
        # CHMR016 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 12-14-09 -- zen_brahmagupta__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 12-37-55 -- gallant_keller__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'CHMR016-U2(Faeze)'
        
        # CHMR016 (U3 - Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 10-39-36 -- competent_ritchie__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 10-29-46 -- dazzling_pasteur__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'CHMR016-U3(Frank)'
        
        # CHMR016 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 12-58-23 -- angry_khayyam__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-11-25 13-16-23 -- pensive_shtern__Alex-Vieth-NonExpert-AI-based' # Code stopped responding
                DIR_EXPERIMENTS / '2024-11-27 09-02-03 -- thirsty_driscoll__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'CHMR016-U4(Alex)'
        
        # CHMR016 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 15-32-27 -- competent_jemison__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 15-52-13 -- zealous_wozniak__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'S4(CHMR016)-U5(Patrick)'
        
        # CHMR016 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 16-08-03 -- nice_mendel__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 16-25-29 -- suspicious_euler__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'S4(CHMR016)-U6(Chinmay)'
        
        # CHMR016 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 14-29-34 -- elegant_shirley__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 15-02-51 -- cranky_hypatia__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'S4(CHMR016)-U7(Ruochen)'
        
        # CHMR016 (Non-experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 09-02-04 -- vigorous_elbakyan__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-25 12-14-09 -- zen_brahmagupta__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-27 10-39-36 -- competent_ritchie__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-25 12-58-23 -- angry_khayyam__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-25 15-32-27 -- competent_jemison__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-25 16-08-03 -- nice_mendel__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-25 14-29-34 -- elegant_shirley__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 09-28-56 -- laughing_burnell__Yuheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-25 12-37-55 -- gallant_keller__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-27 10-29-46 -- dazzling_pasteur__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-27 09-02-03 -- thirsty_driscoll__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-25 15-52-13 -- zealous_wozniak__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-25 16-25-29 -- suspicious_euler__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-25 15-02-51 -- cranky_hypatia__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'S4(CHMR016)-NonExperts'
            percInterp = True
            TIME_TOOMUCH = 10000

    # CHMR020 (Session 5)
    elif 0:
        
        # CHMR020 (C3-Niels)
        if 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 11-41-26 -- inspiring_liskov__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-25 11-48-46 -- optimistic_feynman__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'S5(CHMR020)-C3(Niels)'
            useGT    = False
    
    # ----------------------------------------------------

    # CHUP059/CHUP048/CHUP005 (Session X)
    elif 0:
        
        # CHUP033 (U0-Prerak) (Dice=0.724)
        if 0:
            pathsManualExperiments = [
                # DIR_EXPERIMENTS / '2024-11-27 12-49-36 -- upbeat_ramanujan__Prerak-Mody-NonExpert-Manual'
                DIR_EXPERIMENTS / '2024-12-02 18-00-03 -- peaceful_neumann__Prerak-Mody-NonExpert-Manual' # with scrolling data
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-11-27 09-48-36 -- trusting_mclaren__Prerak-Mody-NonExpert-AI-based' 
                DIR_EXPERIMENTS / '2024-12-02 18-09-14 -- relaxed_hugle__Prerak-Mody-NonExpert-AI-based'  # with scrolling data
            ]
            userName = 'SX(CHUP033)-U0(Prerak)'

        # CHUP059 (U0-Prerak) (Dice=0.719)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-26 16-39-21 -- elastic_rubin__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-26 16-52-58 -- wonderful_dijkstra__Prerak-Mody-NonExpert-AI-based' 
            ]
            userName = 'SX(CHUP059)-U0(Prerak)'
        
        # CHUP005 (U0-Prerak) (Dice=0.714) # AIScribble-drops-DICE patient
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 08-17-28 -- vigorous_ptolemy__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 08-12-57 -- focused_mccarthy__Prerak-Mody-NonExpert-AI-based' 
            ]
            userName = 'SX(CHUP005)-U0(Prerak)'
        
        # CHUP064 (U0-Prerak) (Dice=0.696)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 08-44-05 -- brave_agnesi__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 08-52-23 -- compassionate_payne__Prerak-Mody-NonExpert-AI-based' 
            ]
            userName = 'SX(CHUP064)-U0(Prerak)'
        
        # CHUP028 (U0-Prerak) (Dice=0.690)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 09-34-25 -- compassionate_bassi__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 09-30-55 -- wizardly_lovelace__Prerak-Mody-NonExpert-AI-based' 
            ]
            userName = 'SX(CHUP028)-U0(Prerak)'
        
        # CHUP044 (U0-Prerak) (Dice=0.677)
        elif 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 13-17-08 -- affectionate_ishizaka__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-27 13-07-28 -- affectionate_taussig__Prerak-Mody-NonExpert-AI-based' 
            ]
            userName = 'SX(CHUP044)-U0(Prerak)'

        # CHUP048 (U0-Prerak) (Dice=0.615)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-26 17-24-39 -- adoring_khorana__Prerak-Mody-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-26 17-30-07 -- charming_shirley__Prerak-Mody-NonExpert-AI-based' 
            ]
            userName = 'SX(CHUP048)-U0(Prerak)'
        
    # ----------------------------------------------------
    
    # P1(CHUP-033)
    elif 0:

        # CHUP-033 (C1-Martin)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 14-40-44 -- pedantic_lehmann__Martin-De Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-11-29 09-19-41 -- busy_margulis__Martin-De Jong-Expert-AI-based' 
                DIR_EXPERIMENTS / '2024-12-12 14-50-57 -- brave_diffie__Martin-De Jong-Expert-AI-based'
            ]
            userName = 'P1(CHUP-033)-C1(Martin)'
            useGT    = False

        # CHUP-033 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 14-31-12 -- reverent_taussig__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 08-12-22 -- jolly_hopper__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'P1(CHUP-033)-C2(Mishca)'
            useGT    = False
        
        # CHUP-033 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-12 13-54-50 -- exciting_poincare__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 12-15-16 -- modest_lamarr__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'P1(CHUP-033)-C3(Niels)'
            useGT    = False
        
        # CHUP-033 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 08-47-00 -- jovial_vaughan__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 08-28-22 -- fervent_lederberg__Jos-Elbers-Expert-AI-based' 
            ]
            userName = 'P1(CHUP-033)-C4(Jos)'
            useGT    = False

        # CHUP-033 (Experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 14-40-44 -- pedantic_lehmann__Martin-De Jong-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 14-31-12 -- reverent_taussig__Mischa-de Ridder-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-12 13-54-50 -- exciting_poincare__Niels-den Haan-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 08-47-00 -- jovial_vaughan__Jos-Elbers-Expert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-12 14-50-57 -- brave_diffie__Martin-De Jong-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-03 08-12-22 -- jolly_hopper__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 12-15-16 -- modest_lamarr__Niels-den Haan-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-05 08-28-22 -- fervent_lederberg__Jos-Elbers-Expert-AI-based',
            ]
            userName = 'P1(CHUP-033)-Experts'
            useGT    = False
            percInterp = True
            useScrollData = False
            makeVideo     = False
            doAllIters    = True # [False, True] should generally be True
            showTitle     = False # for submission
            showLegend    = True

        # ----------------------------------------------------

        # CHUP-033 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-03-30 -- loving_mestorf__Yauheniya-Makarevich-NonExpert-Manual'
                # DIR_EXPERIMENTS / '2024-12-06 15-08-08 -- festive_poitras__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 10-13-51 -- elated_ride__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'P1(CHUP-033)-U1(Yauheniya)'
        
        # CHUP-033 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 12-40-37 -- quirky_allen__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 12-06-33 -- jovial_tu__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'P1(CHUP-033)-U2(Faeze)'
        
        # CHUP-033 (U3-Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 15-52-24 -- frosty_khorana__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 13-17-37 -- vibrant_banach__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'P1(CHUP-033)-U3(Frank)'
        
        # CHUP-033 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 09-27-26 -- goofy_sutherland__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 09-06-36 -- nostalgic_sammet__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'P1(CHUP-033)-U4(Alex)'
        
        # CHUP-033 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 13-27-34 -- cranky_kowalevski__patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 15-03-16 -- affectionate_galois__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'P1(CHUP-033)-U5(Patrick)'
        
        # CHUP-033 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 16-04-54 -- vigorous_wescoff__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 12-05-53 -- bold_villani__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'P1(CHUP-033)-U6(Chinmay)'
        
        # CHUP-033 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 11-49-19 -- angry_tesla__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 14-26-23 -- exciting_pasteur__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'P1(CHUP-033)-U7(Ruochen)'

        # CHUP-033 (Non-experts)
        elif 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-03-30 -- loving_mestorf__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 12-40-37 -- quirky_allen__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-03 15-52-24 -- frosty_khorana__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 09-27-26 -- goofy_sutherland__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 13-27-34 -- cranky_kowalevski__patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 16-04-54 -- vigorous_wescoff__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 11-49-19 -- angry_tesla__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 10-13-51 -- elated_ride__Yauheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-04 12-06-33 -- jovial_tu__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-28 13-17-37 -- vibrant_banach__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-02 09-06-36 -- nostalgic_sammet__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-29 15-03-16 -- affectionate_galois__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-28 12-05-53 -- bold_villani__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-04 14-26-23 -- exciting_pasteur__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'P1(CHUP-033)-NonExperts'
            percInterp    = True
            TIME_TOOMUCH  = 60
            useScrollData = False
            makeVideo     = False # to keep it fast
            showTitle     = False # for submission
            plotFigs      = True # False to get numbers fast
            showLegend    = True  

    # P2(CHUP-059)
    elif 0:
        
        # CHUP-059 (C1-Martin)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 09-34-51 -- magical_mahavira__Martin-De Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-12-03 14-32-10 -- vigilant_wilson__Martin-De Jong-Expert-AI-based' 
                DIR_EXPERIMENTS / '2024-12-12 15-08-33 -- boring_wilson__Martin-De Jong-Expert-AI-based'
            ]
            userName = 'P2(CHUP-059)-C1(Martin)'
            useGT    = False

        # CHUP-059 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 08-20-15 -- recursing_shaw__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 14-25-58 -- gallant_galileo__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'P2(CHUP-059)-C2(Mishca)'
            useGT    = False
        
        # CHUP-059 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 12-22-15 -- awesome_shamir__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-12 14-04-32 -- tender_wilbur__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'P2(CHUP-059)-C3(Niels)'
            useGT    = False
        
        # CHUP-059 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 08-33-14 -- happy_elgamal__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 08-53-44 -- clever_bartik__Jos-Elbers-Expert-AI-based' 
            ]
            userName = 'P2(CHUP-059)-C4(Jos)'
            useGT    = False
        
        # CHUP-059 (Experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 09-34-51 -- magical_mahavira__Martin-De Jong-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-03 08-20-15 -- recursing_shaw__Mischa-de Ridder-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 12-22-15 -- awesome_shamir__Niels-den Haan-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-05 08-33-14 -- happy_elgamal__Jos-Elbers-Expert-Manual',
                # DIR_EXPERIMENTS / '',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-12 15-08-33 -- boring_wilson__Martin-De Jong-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 14-25-58 -- gallant_galileo__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-12 14-04-32 -- tender_wilbur__Niels-den Haan-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 08-53-44 -- clever_bartik__Jos-Elbers-Expert-AI-based',
                # DIR_EXPERIMENTS / '',
            ]
            userName      = 'P2(CHUP-059)-Experts'
            useGT         = False
            percInterp    = True
            useScrollData = False
            makeVideo     = False # False to keep it fast
            makeRender    = True
            doAllIters    = True # [False, True] should generally be True
            showTitle     = False # for submission
            showLegend    = False

        # ----------------------------------------------------

        # CHUP-059 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 10-29-31 -- magical_blackburn__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 15-56-44 -- zen_driscoll__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'P2(CHUP-059)-U1(Yauheniya)'
        
        # CHUP-059 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 12-18-46 -- nifty_wu__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 12-37-15 -- blissful_hopper__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'P2(CHUP-059)-U2(Faeze)'
        
        # CHUP-059 (U3 - Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 13-25-01 -- nostalgic_swanson__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 15-38-16 -- gifted_murdock__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'P2(CHUP-059)-U3(Frank)'
        
        # CHUP-059 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 09-15-07 -- wizardly_pike__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 09-06-22 -- condescending_einstein__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'P2(CHUP-059)-U4(Alex)'
        
        # CHUP-059 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 15-12-36 -- sweet_sinoussi__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 10-03-09 -- reverent_mirzakhani__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'P2(CHUP-059)-U5(Patrick)'
        
        # CHUP-059 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 12-15-42 -- nice_mccarthy__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 16-20-34 -- eloquent_lalande__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'P2(CHUP-059)-U6(Chinmay)'
        
        # CHUP-059 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 14-37-23 -- mystifying_albattani__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 08-11-36 -- quirky_ishizaka__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'P2(CHUP-059)-U7(Ruochen)'

        # CHUP-059 (Non-experts)
        elif 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 10-29-31 -- magical_blackburn__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 12-18-46 -- nifty_wu__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-28 13-25-01 -- nostalgic_swanson__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-02 09-15-07 -- wizardly_pike__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-29 15-12-36 -- sweet_sinoussi__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-28 12-15-42 -- nice_mccarthy__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 14-37-23 -- mystifying_albattani__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 15-56-44 -- zen_driscoll__Yauheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-05 12-37-15 -- blissful_hopper__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-03 15-38-16 -- gifted_murdock__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 09-06-22 -- condescending_einstein__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 10-03-09 -- reverent_mirzakhani__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-04 16-20-34 -- eloquent_lalande__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-13 08-11-36 -- quirky_ishizaka__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'P2(CHUP-059)-NonExperts'
            percInterp    = True
            TIME_TOOMUCH  = 60
            useScrollData = False
            makeVideo     = False # to keep it fast
            showTitle     = False # for submission
            plotFigs      = True # False to get numbers fast
            showLegend    = False  

    # P3(CHUP-005)
    elif 0:
        
        # CHUP-005 (C1-Martin)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-00-58 -- tender_hugle__Martin-De Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 09-43-36 -- distracted_feistel__Martin-De Jong-Expert-AI-based' 
            ]
            userName = 'P3(CHUP-005)-C1(Martin)'
            useGT    = False

        # CHUP-005 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 15-03-46 -- dreamy_ritchie__Mischa-de ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-12-03 08-33-21 -- nostalgic_bhabha__Mischa-de Ridder-Expert-AI-based'
                DIR_EXPERIMENTS / '2024-12-18 15-38-23 -- silly_goodall__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'P3(CHUP-005)-C2(Mischa)'
            useGT    = False
        
        # CHUP-005 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-16 12-00-19 -- great_jennings__Niels -den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 12-26-54 -- crazy_cohen__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'P3(CHUP-005)-C3(Niels)'
            useGT    = False
        
        # CHUP-005 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 09-05-17 -- gracious_ride__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 08-36-55 -- epic_leavitt__Jos-Elbers-Expert-AI-based' 
            ]
            userName = 'P3(CHUP-005)-C4(Jos)'
            useGT    = False

        # CHUP-005 (Experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-00-58 -- tender_hugle__Martin-De Jong-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 15-03-46 -- dreamy_ritchie__Mischa-de ridder-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-16 12-00-19 -- great_jennings__Niels -den Haan-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-13 09-05-17 -- gracious_ride__Jos-Elbers-Expert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 09-43-36 -- distracted_feistel__Martin-De Jong-Expert-AI-based',
                # DIR_EXPERIMENTS / '2024-12-03 08-33-21 -- nostalgic_bhabha__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-18 15-38-23 -- silly_goodall__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 12-26-54 -- crazy_cohen__Niels-den Haan-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-05 08-36-55 -- epic_leavitt__Jos-Elbers-Expert-AI-based',
            ]
            userName = 'P3(CHUP-005)-Experts'
            useGT         = False
            percInterp    = True
            useScrollData = False
            makeVideo     = False
            doAllIters    = True # [False, True] should generally be True
            showTitle     = False # for submission
            showLegend    = False
        
        # ----------------------------------------------------

        # CHUP-005 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 15-34-30 -- reverent_dhawan__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 10-58-32 -- vibrant_villani__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'P3(CHUP-005)-U1(Yauheniya)'
        
        # CHUP-005 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 14-05-59 -- kind_perlman__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 12-32-33 -- elegant_turing__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'P3(CHUP-005)-U2(Faeze)'
        
        # CHUP-005 (U3 - Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 13-01-56 -- objective_almeida__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 13-41-26 -- competent_haibt__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'P3(CHUP-005)-U3(Frank)'
        
        # CHUP-005 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 09-16-43 -- compassionate_clarke__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 09-27-56 -- boring_bhabha__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'P3(CHUP-005)-U4(Alex)'
        
        # CHUP-005 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 10-13-58 -- quizzical_blackburn__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 15-25-20 -- objective_bose__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'P3(CHUP-005)-U5(Patrick)'
        
        # CHUP-005 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 16-43-31 -- sweet_boyd__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 12-33-57 -- vibrant_proskuriakova__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'P3(CHUP-005)-U6(Chinmay)'
        
        # CHUP-005 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 08-23-33 -- elated_hugle__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 14-49-01 -- gallant_gould__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'P3(CHUP-005)-U7(Ruochen)'

        # CHUP-005 (Non-experts)
        elif 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 15-34-30 -- reverent_dhawan__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 14-05-59 -- kind_perlman__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 13-01-56 -- objective_almeida__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 09-16-43 -- compassionate_clarke__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 10-13-58 -- quizzical_blackburn__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 16-43-31 -- sweet_boyd__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-13 08-23-33 -- elated_hugle__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 10-58-32 -- vibrant_villani__Yauheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-04 12-32-33 -- elegant_turing__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-28 13-41-26 -- competent_haibt__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-02 09-27-56 -- boring_bhabha__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-29 15-25-20 -- objective_bose__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-11-28 12-33-57 -- vibrant_proskuriakova__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-04 14-49-01 -- gallant_gould__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'P3(CHUP-005)-NonExperts'
            percInterp    = True
            TIME_TOOMUCH  = 60
            useScrollData = False
            makeVideo     = False # to keep it fast
            showTitle     = False # for submission
            plotFigs      = True # False to get numbers fast
            showLegend    = False  

    # P4(CHUP-064)
    elif 0:

        # CHUP-064 (C1-Martin)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 09-57-03 -- fervent_hopper__Martin-De Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-06-46 -- nervous_taussig__Martin-De Jong-Expert-AI-based' 
            ]
            userName = 'P4(CHUP-064)-C1(Martin)'
            useGT    = False            

        # CHUP-064 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 08-41-05 -- gifted_meitner__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 15-11-38 -- gallant_hellman__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'P4(CHUP-064)-C2(Mischa)'
            useGT    = False
        
        # CHUP-064 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 12-31-40 -- frosty_hertz__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-16 12-05-24 -- hopeful_sinoussi__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'P4(CHUP-064)-C3(Niels)'
            useGT    = False
        
        # CHUP-064 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 08-42-52 -- epic_cartwright__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 09-09-11 -- practical_davinci__Jos-Elbers-Expert-AI-based' 
            ]
            userName = 'P4(CHUP-064)-C4(Jos)'
            useGT    = False
        
        # CHUP-064 (Experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-29 09-57-03 -- fervent_hopper__Martin-De Jong-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-03 08-41-05 -- gifted_meitner__Mischa-de Ridder-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 12-31-40 -- frosty_hertz__Niels-den Haan-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-05 08-42-52 -- epic_cartwright__Jos-Elbers-Expert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-06-46 -- nervous_taussig__Martin-De Jong-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 15-11-38 -- gallant_hellman__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-16 12-05-24 -- hopeful_sinoussi__Niels-den Haan-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-13 09-09-11 -- practical_davinci__Jos-Elbers-Expert-AI-based',
            ]
            userName = 'P4(CHUP-064)-Experts'
            useGT         = False
            percInterp    = True
            useScrollData = False
            makeVideo     = False
            doAllIters    = True # [False, True] should generally be True
            showTitle     = False # for submission
            showLegend    = False
            plotFigs      = True
        
        # ----------------------------------------------------

        # CHUP-064 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-24-29 -- elated_sammet__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-11 12-06-16 -- blissful_gagarin__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'P4(CHUP-064)-U1(Yauheniya)'
        
        # CHUP-064 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 12-03-21 -- heuristic_wing__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 14-19-43 -- infallible_galois__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'P4(CHUP-064)-U2(Faeze)'
        
        # CHUP-064 (U3 - Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-11-28 13-50-05 -- nifty_hodgkin__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 13-16-41 -- vigorous_khayyam__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'P4(CHUP-064)-U3(Frank)'
        
        # CHUP-064 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                # DIR_EXPERIMENTS / '2024-12-02 09-37-14 -- flamboyant_maxwell__Alex-Vieth-NonExpert-Manual'
                DIR_EXPERIMENTS / '2024-12-04 09-02-17 -- festive_nash__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 09-27-30 -- exciting_newton__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'P4(CHUP-064)-U4(Alex)'
        
        # CHUP-064 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 13-00-30 -- upbeat_bell__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 10-24-44 -- quirky_cohen__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'P4(CHUP-064)-U5(Patrick)'
        
        # CHUP-064 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 15-30-03 -- romantic_galileo__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-10 16-05-10 -- cranky_jackson__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'P4(CHUP-064)-U6(Chinmay)'
        
        # CHUP-064 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 14-57-34 -- jovial_carver__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 08-36-43 -- relaxed_cori__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'P4(CHUP-064)-U7(Ruochen)'

        # CHUP-064 (Non-experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-24-29 -- elated_sammet__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-05 12-03-21 -- heuristic_wing__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-11-28 13-50-05 -- nifty_hodgkin__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 09-02-17 -- festive_nash__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 13-00-30 -- upbeat_bell__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-02 15-30-03 -- romantic_galileo__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 14-57-34 -- jovial_carver__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-11 12-06-16 -- blissful_gagarin__Yauheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 14-19-43 -- infallible_galois__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 13-16-41 -- vigorous_khayyam__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 09-27-30 -- exciting_newton__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 10-24-44 -- quirky_cohen__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-10 16-05-10 -- cranky_jackson__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-13 08-36-43 -- relaxed_cori__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'P4(CHUP-064)-NonExperts'
            percInterp    = True
            TIME_TOOMUCH  = 60
            useScrollData = False
            makeVideo     = False # to keep it fast
            showTitle     = False # for submission
            plotFigs      = True # False to get numbers fast
            showLegend    = False  

    # P5(CHUP-028)
    elif 0:

        # CHUP-028 (C1-Martin)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-12-15 -- zealous_rhodes__Martin-De Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 14-10-13 -- objective_austin__Martin-De Jong-Expert-AI-based' 
            ]
            userName = 'P5(CHUP-028)-C1(Martin)'
            useGT    = False

        # CHUP-028 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 15-16-47 -- elastic_thompson__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                # DIR_EXPERIMENTS / '2024-12-06 14-03-32 -- gifted_chaum__Mischa-de Ridder-Expert-AI-based'
                DIR_EXPERIMENTS / '2024-12-18 15-43-51 -- exciting_grothendieck__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'P5(CHUP-028)-C2(Mischa)'
            useGT    = False
        
        # CHUP-028 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-16 12-08-22 -- great_noether__Niels-den Haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-12 13-43-11 -- nostalgic_payne__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'P5(CHUP-028)-C3(Niels)'
            useGT    = False
        
        # CHUP-028 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 09-12-00 -- mystifying_shamir__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 08-35-46 -- focused_austin__Jos-Elbers-Expert-AI-based' 
            ]
            userName = 'P5(CHUP-028)-C4(Jos)'
            useGT    = False
        
        # CHUP-028 (Experts)
        elif 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-12-15 -- zealous_rhodes__Martin-De Jong-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 15-16-47 -- elastic_thompson__Mischa-de Ridder-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-16 12-08-22 -- great_noether__Niels-den Haan-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-13 09-12-00 -- mystifying_shamir__Jos-Elbers-Expert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 14-10-13 -- objective_austin__Martin-De Jong-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 14-03-32 -- gifted_chaum__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-12 13-43-11 -- nostalgic_payne__Niels-den Haan-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 08-35-46 -- focused_austin__Jos-Elbers-Expert-AI-based',
            ]
            userName = 'P5(CHUP-028)-Experts'
            useGT         = False
            percInterp    = True
            useScrollData = False
            makeVideo     = False
            doAllIters    = True # [False, True] should generally be True
            showTitle     = False # for submission
            showLegend    = False 
            plotFigs      = True # False to get numbers fast

        # ----------------------------------------------------

        # CHUP-028 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-11 12-16-05 -- affectionate_wilson__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-39-05 -- jolly_cray__Yuaheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'P5(CHUP-028)-U1(Yauheniya)'
        
        # CHUP-028 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 14-26-21 -- determined_rubin__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 12-13-17 -- strange_bardeen__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'P5(CHUP-028)-U2(Faeze)'
        
        # CHUP-028 (U3 - Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 13-22-54 -- admiring_darwin__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 15-10-43 -- gracious_elbakyan__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'P5(CHUP-028)-U3(Frank)'
        
        # CHUP-028 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 09-32-40 -- objective_bell__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 09-10-37 -- practical_mestorf__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'P5(CHUP-028)-U4(Alex)'
        
        # CHUP-028 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 10-31-58 -- wonderful_galileo__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 13-11-40 -- exciting_spence__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'P5(CHUP-028)-U5(Patrick)'
        
        # CHUP-028 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-10 16-13-52 -- jolly_goldstine__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 15-40-11 -- quirky_blackburn__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'P5(CHUP-028)-U6(Chinmay)'
        
        # CHUP-028 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 08-45-25 -- frosty_cartwright__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 11-25-53 -- beautiful_joliot__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'P5(CHUP-028)-U7(Ruochen)'

        # CHUP-028 (Non-experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-11 12-16-05 -- affectionate_wilson__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 14-26-21 -- determined_rubin__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 13-22-54 -- admiring_darwin__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-09 09-32-40 -- objective_bell__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 10-31-58 -- wonderful_galileo__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-10 16-13-52 -- jolly_goldstine__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-13 08-45-25 -- frosty_cartwright__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-39-05 -- jolly_cray__Yuaheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-05 12-13-17 -- strange_bardeen__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-03 15-10-43 -- gracious_elbakyan__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-04 09-10-37 -- practical_mestorf__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-04 13-11-40 -- exciting_spence__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-02 15-40-11 -- quirky_blackburn__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 11-25-53 -- beautiful_joliot__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName = 'P5(CHUP-028)-NonExperts'
            percInterp    = True
            TIME_TOOMUCH  = 60
            useScrollData = False
            makeVideo     = False # to keep it fast
            showTitle     = False # for submission
            plotFigs      = True # False to get numbers fast
            showLegend    = False  

    # P6(CHUP-044)
    elif 0:
        
        # CHUP-044 (C1-Martin)
        if 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 14-15-38 -- compassionate_nightingale__Martin-De Jong-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-22-27 -- kind_curie__Martin-De Jong-Expert-AI-based' 
            ]
            userName = 'P6(CHUP-044)-C1(Martin)'
            useGT    = False

        # CHUP-044 (C2-Mischa)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 14-08-30 -- confident_lamarr__Mischa-de Ridder-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 15-25-59 -- vibrant_lumiere__Mischa-de Ridder-Expert-AI-based'
            ]
            userName = 'P6(CHUP-044)-C2(Mishca)'
            useGT    = False
        
        # CHUP-044 (C3-Niels)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-12 13-46-01 -- musing_merkle__Niels-den haan-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-16 12-15-50 -- lucid_fermi__Niels-den Haan-Expert-AI-based' 
            ]
            userName = 'P6(CHUP-044)-C3(Niels)'
            useGT    = False

        # CHUP-044 (C4-Jos)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 08-38-29 -- intelligent_sutherland__Jos-Elbers-Expert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 09-18-35 -- festive_allen__Jos-Elbers-Expert-AI-based' 
            ]
            userName = 'P6(CHUP-044)-C4(Jos)'
            useGT    = False
        
        # CHUP-044 (Experts)
        elif 1:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 14-15-38 -- compassionate_nightingale__Martin-De Jong-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 14-08-30 -- confident_lamarr__Mischa-de Ridder-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-12 13-46-01 -- musing_merkle__Niels-den haan-Expert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 08-38-29 -- intelligent_sutherland__Jos-Elbers-Expert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 13-22-27 -- kind_curie__Martin-De Jong-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 15-25-59 -- vibrant_lumiere__Mischa-de Ridder-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-16 12-15-50 -- lucid_fermi__Niels-den Haan-Expert-AI-based',
                DIR_EXPERIMENTS / '2024-12-13 09-18-35 -- festive_allen__Jos-Elbers-Expert-AI-based',
            ]
            userName = 'P6(CHUP-044)-Experts'
            useGT         = False
            percInterp    = True
            useScrollData = False
            makeVideo     = False
            doAllIters    = True # [False, True] should generally be True
            showTitle     = False # for submission
            showLegend    = False
            plotFigs      = True # False to get numbers fast

        # ----------------------------------------------------

        # CHUP-044 (U1-Yauheniya)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-51-32 -- hungry_archimedes__Yauheniya-Makarevich-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-11 12-26-05 -- great_ellis__Yauheniya-Makarevich-NonExpert-AI-based'
            ]
            userName = 'P6(CHUP-044)-U1(Yauheniya)'
        
        # CHUP-044 (U2-Faeze)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-05 12-19-19 -- relaxed_proskuriakova__Faeze-Gholamiankhah-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 14-34-13 -- gallant_hoover__Faeze-Gholamiankhah-NonExpert-AI-based'
            ]
            userName = 'P6(CHUP-044)-U2(Faeze)'
        
        # CHUP-044 (U3 - Frank)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-03 15-15-34 -- nice_greider__Frank-Dankers-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 13-30-58 -- amazing_lehmann__Frank-Dankers-NonExpert-AI-based'
            ]
            userName = 'P6(CHUP-044)-U3(Frank)'
        
        # CHUP-044 (U4 - Alex)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 09-15-03 -- focused_fermat__Alex-Vieth-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-09 09-39-20 -- vibrant_carver__Alex-Vieth-NonExpert-AI-based'
            ]
            userName = 'P6(CHUP-044)-U4(Alex)'
        
        # CHUP-044 (U5 - Patrick)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 13-16-27 -- hopeful_kilby__Patrick-de Koning-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 10-40-13 -- musing_franklin__Patrick-de Koning-NonExpert-AI-based'
            ]
            userName = 'P6(CHUP-044)-U5(Patrick)'
        
        # CHUP-044 (U6 - Chinmay)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-02 15-45-46 -- jolly_mestorf__Chinmay-Rao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-10 16-23-57 -- xenodochial_cori__Chinmay-Rao-NonExpert-AI-based'
            ]
            userName = 'P6(CHUP-044)-U6(Chinmay)'
        
        # CHUP-044 (U7 - Ruochen)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-06 11-33-17 -- nice_kepler__Ruochen-Gao-NonExpert-Manual'
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-13 08-55-56 -- optimistic_kare__Ruochen-Gao-NonExpert-AI-based'
            ]
            userName = 'P6(CHUP-044)-U7(Ruochen)'

        # CHUP-044 (Non-experts)
        elif 0:
            pathsManualExperiments = [
                DIR_EXPERIMENTS / '2024-12-04 10-51-32 -- hungry_archimedes__Yauheniya-Makarevich-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-05 12-19-19 -- relaxed_proskuriakova__Faeze-Gholamiankhah-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-03 15-15-34 -- nice_greider__Frank-Dankers-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 09-15-03 -- focused_fermat__Alex-Vieth-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-04 13-16-27 -- hopeful_kilby__Patrick-de Koning-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-02 15-45-46 -- jolly_mestorf__Chinmay-Rao-NonExpert-Manual',
                DIR_EXPERIMENTS / '2024-12-06 11-33-17 -- nice_kepler__Ruochen-Gao-NonExpert-Manual',
            ]
            pathsSemiAutoExperiments = [
                DIR_EXPERIMENTS / '2024-12-11 12-26-05 -- great_ellis__Yauheniya-Makarevich-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 14-34-13 -- gallant_hoover__Faeze-Gholamiankhah-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 13-30-58 -- amazing_lehmann__Frank-Dankers-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-09 09-39-20 -- vibrant_carver__Alex-Vieth-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-06 10-40-13 -- musing_franklin__Patrick-de Koning-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-10 16-23-57 -- xenodochial_cori__Chinmay-Rao-NonExpert-AI-based',
                DIR_EXPERIMENTS / '2024-12-13 08-55-56 -- optimistic_kare__Ruochen-Gao-NonExpert-AI-based',
            ]
            userName      = 'P6(CHUP-044)-NonExperts'
            percInterp    = True
            TIME_TOOMUCH  = 60
            useScrollData = False
            makeVideo     = False # to keep it fast
            showTitle     = False # for submission
            plotFigs      = True # False to get numbers fast
            showLegend    = False  

    # for cropLastSlices in [False, True]:
    for cropLastSlices in [False]:
        eval(pathsManualExperiments, pathsSemiAutoExperiments, userName, cropLastSlices=cropLastSlices, useGT=useGT)

"""
2024-11-09
 - What are the axis distributions (order of axial, coronal, sagittal) for mask data in the following cases:
   1. hecktor data (irrelevant here, as we are not using it in this code) 
        --> (sagittal, coronal, axial)
   2. orthanc .dcm 
        --> np.fliplr(np.rot90(x, k=3)) on axial sliceIdx for PET/CT
        --> np.moveaxis(maskArray, [0,1,2], [2,1,0]) (sagittal, coronal, axial) --> (axial, coronal, sagittal)
            --> I am reading this in this code
   3. /prepare .dcm (stored in Orthanc too, so semiauto can be compared with GT without any transformations)
        --> np.rot90(np.flipud(maskArrayCopy[:,:,idx]), k=3) on axial sliceIdx of (sagittal, coronal, axial)
        --> np.moveaxis(maskArrayCopy,[0,1,2], [2,1,0]) (sagittal, coronal, axial) --> (axial, coronal, sagittal)
   4. /uploadManualRefinement .dcm
        --> None (??)
"""

"""
 - conda activate interactive-refinement
 - To run this you need to ensure your DICOm server needs to be on at localhost:8042
"""

"""
*On Windows
**start Docker machine too (and test localhost:8042, else orthanc will not work)
cd "D:\HCAI\Project 5 - Interactive Contour Refinement\code\cornerstone3D-trials"
conda activate interactive-refinement
python src/backend/utils/evalUtils.py

"""