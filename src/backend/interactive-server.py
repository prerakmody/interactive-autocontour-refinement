import os, sys
import pdb
import time
import yaml
import json
import http
import timeit
import psutil
import logging
import imageio
import platform
import warnings
import datetime
import traceback
import tracemalloc
import setproctitle
import numpy as np
from pathlib import Path

import matplotlib.colors
import skimage.morphology
import matplotlib.pyplot as plt

import requests
import pydicom
import pydicom_seg
import directory_tree
import dicomweb_client
import SimpleITK as sitk
logging.getLogger('dicomweb_client').setLevel(logging.ERROR)

import re
import copy
import ssl
import typing
import fastapi
import uvicorn
import pydantic
import starlette
import fastapi.middleware.cors
import starlette.middleware.sessions
from contextlib import asynccontextmanager

import termcolor

import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import edt
import torch
import monai
import scipy.ndimage
torch.manual_seed(42)
np.random.seed(42)

import gc
import onnx
import onnxruntime
gc.collect() # TODO: Is this the right place to keep it?

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.onnx")
logging.getLogger('onnxscript').setLevel(logging.WARNING)

######################## KEYS ########################

if 1:
    # Keys - DICOM
    KEY_STUDY_INSTANCE_UID  = 'StudyInstanceUID'
    KEY_SERIES_INSTANCE_UID = 'SeriesInstanceUID'
    KEY_SOP_INSTANCE_UID    = 'SOPInstanceUID'
    KEY_WADO_RS_ROOT        = 'wadoRsRoot'

    # Keys - Model load
    KEY_MODEL_STATE_DICT = 'model_state_dict'
    EPOCH_STR            = 'epoch_{:04d}'

    # Keys - Input + session-data json
    KEY_DATA                 = 'data'
    KEY_TORCH_DATA           = 'torchData'
    KEY_SCRIBBLE_MAP         = 'scribbleMap' # a 3D map of 1's for bgd and 2's for fgd
    KEY_CLIENT_IDENTIFIER    = 'clientIdentifier'
    KEY_DCM_LIST             = 'dcmList'
    KEY_SCRIBBLE_COUNTER     = 'scribbleCounter'
    KEY_MANUAL_COUNTER       = 'manualCounter'
    KEY_DATETIME             = 'dateTime'
    KEY_PATH_SAVE            = 'pathSave'
    KEY_SEG_ARRAY_GT         = 'segArrayGT'
    KEY_SEG_SOP_INSTANCE_UID    = 'segSOPInstanceUID'
    KEY_SEG_SERIES_INSTANCE_UID = 'segSeriesInstanceUID'
    KEY_SEG_ORTHANC_ID          = 'segOrthancID'
    KEY_PATIENT_NAME            = 'patientName'
    KEY_SLICE_IDXS_SCROLLED_OBJ = 'scrolledSliceIdxsObj'

    VALUE_INT_FGD = 1
    VALUE_INT_BGD = 2

    KEY_VIEW_TYPE     = 'viewType'
    KEY_SCRIBBLE_TYPE = 'scribbleType'
    KEY_TIME_TO_SCRIBBLE = 'timeToScribble'
    KEY_SCRIBBLE_START_EPOCH = 'scribbleStartEpoch'
    KEY_SCRIBBLE_FGD  = 'fgd'
    KEY_SCRIBBLE_BGD  = 'bgd'
    KEY_POINTS_3D     = 'points3D'

    # Keys - For DICOM server
    KEY_CASE_NAME          = 'caseName'
    KEY_USERNAME           = 'userName'
    KEY_SEARCH_OBJ_CT      = 'searchObjCT'
    KEY_SEARCH_OBJ_PET     = 'searchObjPET'
    KEY_SEARCH_OBJ_RTSGT   = 'searchObjRTSGT'
    KEY_SEARCH_OBJ_RTSPRED = 'searchObjRTSPred'

    # Keys - For response json
    KEY_STATUS = 'status'
    KEY_RESPONSE_DATA = 'responseData'

    # Keys - For saving
    fileNameForSave = lambda name, counter, viewType, sliceId: '-'.join([str(name), SERIESDESC_SUFFIX_REFINE, '{:03d}'.format(counter), viewType, 'slice{:03d}'.format(sliceId)])

    # Keys - for extensions
    KEY_EXT_ONNX = '.onnx'

    # Key - for views
    KEY_AXIAL    = 'Axial'
    KEY_CORONAL  = 'Coronal'
    KEY_SAGITTAL = 'Sagittal'

    # Keys - for colors
    COLORSTR_RED   = 'red'
    COLORSTR_GREEN = 'green'
    COLORSTR_PINK  = 'pink'
    COLORSTR_GRAY  = 'gray'
    SAVE_DPI = 200 # 200=3MB, 150=?]

    # Keys - For platforms
    KEY_PLATFORM_LINUX   = 'Linux'
    KEY_PLATFORM_WINDOWS = 'Windows'
    KEY_PLATFORM_DARWIN  = 'Darwin'

    # Vars - For logger
    LOG_CONFIG = None

    # Keys - for ram and gpu usage
    KEY_RAM_USAGE_IN_GB  = 'usedRAMInGB'
    KEY_TOTAL_RAM_IN_GB  = 'totalRAMInGB'
    KEY_GPU_USAGE_IN_GB  = 'usedGPUInGB'
    KEY_TOTAL_GPU_IN_GB  = 'totalGPUInGB'
    KEY_PLATFORM         = 'platform'
    KEY_DEVICE_MODEL     = 'deviceModel'
    KEY_TIME             = 'timeTaken'

    # Other keys
    MSG_RETURN_PREFIX = "[clientIdentifier={}, patientName={}, loadOnnx={}]"

    # Private Dicom Tags (https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.datadict.add_dict_entry.html) # Decimal String
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
    # pydicom.datadict.add_private_dict_entry('Mody', 0x10010001, VALUEREP_FLOAT32, 'TimeToBrush', 'Time To Brush')
    # pydicom.datadict.add_private_dict_entry('Mody', 0x10011010, VALUEREP_FLOAT32, 'TimeToScribble', 'Time To Scribble')
    # pydicom.datadict.add_private_dict_entry('Mody', 0x10011011, VALUEREP_FLOAT32, 'TimeToDistMap', 'Time for Distance Map')
    # pydicom.datadict.add_private_dict_entry('Mody', 0x10011012, VALUEREP_FLOAT32, 'TimeToInfer', 'Time for Inference')

    # Dev vs Prod
    KEY_JOHN = 'John'
    KEY_DOE  = 'Doe'

    # Jsons/Suffixs
    SUFFIX_SLICE_SCROLL_JSON = 'slice-scroll.json'


######################## User-defined settings ########################
if 1:
    # Settings - Python server
    USE_HTTPS = os.getenv('USE_HTTPS', True)
    if type(USE_HTTPS) == str:
        if USE_HTTPS.lower() in ['True','true', '1', 'yes', '']: USE_HTTPS = True
        else: USE_HTTPS = False

    HOST = '0.0.0.0' # ['localhost', 0.0.0.0] # NOTE: using 0.0.0.0 for docker situations
    PORT_PYTHON = 55000
    PORT_NODEJS = 50000
    MODE_DEBUG  = False

    # Settings - Model Input
    SHAPE_TENSOR  = (1, 5, 144, 144, 144)
    HU_MIN, HU_MAX   = -250, 250
    SUV_MIN_v1, SUV_MAX_v1 = 0   ,25000
    SUV_MIN_v2, SUV_MAX_v2 = 0   ,25

    # Settings - Model Type
    KEY_UNET_V1          = 'unet_v1'

    # Settings - Distance Map
    DISTMAP_Z = 3
    DISTMAP_SIGMA = 0.005

    # Settings - Paths and filenames
    DIR_THIS        = Path(__file__).parent.absolute() # <root>/src/backend/
    # DIR_SRC         = DIR_THIS.parent.absolute() # <root>/src/
    DIR_SRC         = DIR_THIS
    DIR_ASSETS      = DIR_SRC / 'assets/'
    DIR_UTILS       = DIR_SRC / 'backend' / 'utils/'
    # DIR_MAIN        = DIR_SRC.parent.absolute() # <root>/
    DIR_MAIN        = DIR_THIS
    DIR_MODELS      = DIR_MAIN / '_models/'
    DIR_LOGS        = DIR_MAIN / '_logs/'
    DIR_KEYS        = DIR_MAIN / '_keys/'
    DIR_EXPERIMENTS = DIR_MAIN / '_experiments/'
    DIR_EXPERIMENTS_JOHNDOE = DIR_MAIN / '_experiments/experiments-john-doe'
    
    Path(DIR_LOGS).mkdir(parents=True, exist_ok=True)
    Path(DIR_EXPERIMENTS).mkdir(parents=True, exist_ok=True)
    Path(DIR_EXPERIMENTS_JOHNDOE).mkdir(parents=True, exist_ok=True)

    FILENAME_PATIENTS_UUIDS_JSON = 'patients-uuids.json'
    FILENAME_METAINFO_SEG_JSON   = 'metainfo-segmentation.json'
    SERIESDESC_SUFFIX_REFINE     = 'Series-SEG-Refine'
    SERIESDESC_SUFFIX_REFINE_MAN = 'Series-SEG-RefineManual'
    CREATORNAME_REFINE           = 'Modys Refinement model: ' + str(KEY_UNET_V1)
    SERIESNUM_REFINE             = 5
    SERIESNUM_REFINE_MANUAL      = 6
    SUFIX_REFINE                 = 'Refine'

    # PATH_HOSTCERT  = DIR_ASSETS / 'hostCert.pem'
    # PATH_HOSTKEY   = DIR_ASSETS / 'hostKey.pem'
    PATH_HOSTCERT  = DIR_KEYS / 'hostCert.pem'
    PATH_HOSTKEY   = DIR_KEYS / 'hostKey.pem'
    PATH_LOGCONFIG = DIR_ASSETS / 'logConfigCustom.yaml'

    PATH_CWD = Path.cwd()
    PATHS_UTILS_PYTHONFILES = list(DIR_UTILS.glob('*.py'))
    PATHS_EXCLUDE_UTILS_RELATIVE = [str(path.relative_to(PATH_CWD)) for path in PATHS_UTILS_PYTHONFILES]
    PATHS_INCLUDE_UTILS_RELATIVE = []
    
    # Settings - Dicom Client
    HOST_DCM = '0.0.0.0'
    if os.getenv('USE_DOCKER', True):
        HOST_DCM = 'database'
    DCM_SERVER_URL = f'http://{HOST_DCM}:8042/dicom-web'
    print (' - [configureFastAPIApp()] DCM_SERVER_URL: ', DCM_SERVER_URL)

#################################################################
#                             UTILS
#################################################################

class StripLeadingSlashMiddleware(starlette.middleware.base.BaseHTTPMiddleware):
    async def dispatch(self, request: fastapi.Request, call_next):
        # Option 1: Rewrite the scope's "path" directly
        path = request.scope.get("path", "")
        request.state.original_path = path  # save if needed
        request.scope["path"] = path.rstrip("/")
        
        # Option 2: Alternatively, if you do not want to affect route matching,
        # you can just store the modified path for use in your endpoints:
        # request.state.strip_path = path.rstrip("/")
        
        response = await call_next(request)
        return response

class CustomCORSMiddleware(starlette.middleware.base.BaseHTTPMiddleware):
    def __init__(self, app, allow_origins=None, allow_credentials=True, allow_methods=None, allow_headers=None):
        super().__init__(app)
        self.allow_origins = allow_origins or []
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or ["*"]
        self.allow_headers = allow_headers or ["*"]

    async def dispatch(self, request: fastapi.Request, call_next):
        origin = request.headers.get("origin")
        print (' - [CustomCORSMiddleware] Origin: ', origin)
        if origin and self.is_allowed_origin(origin):
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = str(self.allow_credentials).lower()
            response.headers["Access-Control-Allow-Methods"] = ",".join(self.allow_methods)
            response.headers["Access-Control-Allow-Headers"] = ",".join(self.allow_headers)
            return response
        return await call_next(request)

    def is_allowed_origin(self, origin):
        for allowed_origin in self.allow_origins:
            if re.match(allowed_origin, origin):
                return True
        return False

class LogOriginMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            request = fastapi.Request(scope, receive)
            origin = request.headers.get('origin')
            print (' ===========================================>>')
            logging.info(f"Incoming request origin: {origin}")
            # print (' - [LogOriginMiddleware] Origin: ', origin)
        await self.app(scope, receive, send)

def getPublicIP():
    try:
        return requests.get('https://api64.ipify.org').text
    except:
        return None

def configureFastAPIApp(app):
    
    # Step 1 - Get hosts
    publicIP = getPublicIP()
    hostsAll = ['127.0.0.1', 'localhost'] #, 'http://37-97-228-132.colo.transip.net/hcai-rt/']
    if publicIP is not None:
        hostsAll.append(publicIP)
    
    # Step 2 - Get ports
    ports        = [PORT_NODEJS, PORT_PYTHON] # has to be able to accept requests from the nodejs server (otherwise you get CORS error)
    
    # Step 3 - Combine protocol + host + port
    origins      = [f"http://{host}:{port}" for host in hostsAll for port in ports]
    origins      += [f"https://{host}:{port}" for host in hostsAll for port in ports]
    # print (' - [configureFastAPIApp()] Origins: ', origins[:10])
    
    # Step 4 - Add middleware
    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=origins, #["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
        # allow_origin_regex=allow_origin_regex,
    )

    # Step 99 - Debug
    # app.router.redirect_slashes = False

    return app

def getTorchDevice(verbose=False):

    # Step 0 - Init
    device = torch.device('cpu')

    # Step 1 - Get device
    if platform.system() == KEY_PLATFORM_DARWIN:
        if torch.backends.mps.is_available(): 
            device = torch.device('mps')
            device = torch.device('cpu'); print ('\n - [getTorchDevice()] MPS on torch does not seem to work on MacOS. So using cpu.')
    elif platform.system() in [KEY_PLATFORM_LINUX, KEY_PLATFORM_WINDOWS]:
        if torch.cuda.is_available(): device = torch.device('cuda')
    else:
        print (' - Unknown platform: {}'.format(platform.system()))

    # Step 2 - Debug
    if 0:
        device = torch.device('cpu')

    # Step 99 - Final
    if verbose:
        print ('\n - Device: {}\n'.format(device))
    return device

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

ramUsageInGBPrev = 0
gpuUsageinGBPrev = 0
def getMemoryUsage(printFlag=True, returnFlag=False, updatePrevFlag=True):

    t0 = time.time()
    ramUsageInMB = None
    totalRAMInMB = None
    gpuUsageInMB = None
    totalGPUInMB = None

    ramUsageInGB = None
    totalRAMInGB = None
    gpuUsageInGB = None
    totalGPUInGB = None

    platformStr = None
    deviceModel = getTorchDevice(verbose=False)

    try:

        # Step 0 - Init
        pid  = os.getpid()
        proc = psutil.Process(pid)
        platformStr = platform.system()

        # Step 1 - Get RAM usage
        try:
            ramUsageInMB  = proc.memory_info().rss / 1024 / 1024 # in MB
            totalRAMInMB  = psutil.virtual_memory().total / 1024 / 1024
        except:
            traceback.print_exc()

        # Step 2 - Get GPU usage
        try:
            if platformStr == KEY_PLATFORM_DARWIN: # [TODO: need to redo this for MacOS]
                # gpuUsageInMB = proc.memory_info().vms / 1024 / 1024
                pass
            elif platformStr in [KEY_PLATFORM_LINUX, KEY_PLATFORM_WINDOWS]:
                if torch.cuda.is_available():
                    totalGPUInMB = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
                    import nvitop
                    nvDevices = nvitop.Device.all()
                    myNVProcess = None
                    for nvDevice in nvDevices:
                        nvProcesses = nvDevice.processes()
                        if pid in nvProcesses:
                            myNVProcess = nvProcesses[pid]
                            break
                    if myNVProcess is not None:
                        # print (dir(myNVProcess))
                        # print (myNVProcess.gpu_memory()) # shows N/A
                        # print (myNVProcess.gpu_memory_human()) # shows N/A
                        # print (myNVProcess.gpu_memory_percent()) # shows N/A
                        # print (myNVProcess.gpu_sm_utilization()) # shows N/A
                        gpuUsageInMB = float(myNVProcess.host_memory_human().split('MiB')[0]) # this seems to be the same as proc.memory_info().rss / 1024 / 1024
                           
        except:
            traceback.print_exc()

        # Step 3 - Get final outputs
        if ramUsageInMB is not None:
            ramUsageInGB  = '{:.2f}'.format(ramUsageInMB / 1024.0) # in GB
        if totalRAMInMB is not None:
            totalRAMInGB = '{:.2f}'.format(totalRAMInMB / 1024.0)
        if gpuUsageInMB is not None:
            gpuUsageInGB = '{:.2f}'.format(gpuUsageInMB / 1024.0)   
        if totalGPUInMB is not None:
            totalGPUInGB = '{:.2f}'.format(totalGPUInMB / 1024.0)
        
        # Step 4 - Get deltas
        global ramUsageInGBPrev, gpuUsageinGBPrev
        if ramUsageInGBPrev is not None and ramUsageInGB is not None:
            ramUsageInGBDelta = '{:.2f}'.format(float(ramUsageInGB) - float(ramUsageInGBPrev))
        else:
            ramUsageInGBDelta = None
        
        if gpuUsageinGBPrev is not None and gpuUsageInGB is not None:
            gpuUsageInGBDelta = '{:.2f}'.format(float(gpuUsageInGB) - float(gpuUsageinGBPrev))
        else:
            gpuUsageInGBDelta = None
        strToReturn = ' ** [{}][{}][{}] Memory usage: RAM ({}(Δ={})/{} GB), GPU ({}(Δ={})/{} GB)'.format(platformStr, deviceModel, pid, ramUsageInGB, ramUsageInGBDelta, totalRAMInGB, gpuUsageInGB, gpuUsageInGBDelta, totalGPUInGB)
        # strToReturn += ' | SESSIONSGLOBAL={:.2f} GB'.format(get_size(SESSIONSGLOBAL)/1024.0/1024.0/1024.0) # in GB
        timeTaken = time.time() - t0
        strToReturn += ' ({:.2f} s)'.format(timeTaken)
        strToReturn = termcolor.colored(strToReturn, 'green')
        if updatePrevFlag:
            ramUsageInGBPrev = ramUsageInGB
            gpuUsageinGBPrev = gpuUsageInGB

        if printFlag:
            # print (strToReturn)
            logger.info(strToReturn)
        
        if returnFlag:
            resMemory = {
                KEY_RAM_USAGE_IN_GB: ramUsageInGB, KEY_TOTAL_RAM_IN_GB: totalRAMInGB,
                KEY_GPU_USAGE_IN_GB: gpuUsageInGB, KEY_TOTAL_GPU_IN_GB: totalGPUInGB,
                KEY_PLATFORM: platformStr, KEY_DEVICE_MODEL: deviceModel.type,
                KEY_TIME: timeTaken
            }
            return resMemory
    
    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
def getRequestInfo(request):
    userAgent = request.headers.get('user-agent', 'userAgentIsNone')
    referer   = request.headers.get('referer', 'refererIsNone')
    return userAgent, referer

def epoch_to_datetime(epoch_ms):
  """Converts epoch time in milliseconds to a datetime object.

  Args:
    epoch_ms: Epoch time in milliseconds.

  Returns:
    A datetime object representing the given epoch time.
  """

  dt = datetime.datetime.fromtimestamp(int(epoch_ms) / 1000)
  return dt


#################################################################
#                        DATA MODELS
#################################################################


class SearchObj(pydantic.BaseModel):
    StudyInstanceUID: str = pydantic.Field(...)
    SeriesInstanceUID: str = pydantic.Field(...)
    SOPInstanceUID: str = pydantic.Field(...)
    wadoRsRoot: str = pydantic.Field(...)

class PreparedData(pydantic.BaseModel):
    searchObjCT: SearchObj = pydantic.Field(...)
    searchObjPET: SearchObj = pydantic.Field(...)
    searchObjRTSGT: SearchObj = pydantic.Field(...)
    searchObjRTSPred: SearchObj = pydantic.Field(...)
    caseName: str = pydantic.Field(...)

# Incoming packet
class PayloadPrepare(pydantic.BaseModel):
    data: PreparedData = pydantic.Field(...)
    identifier: str = pydantic.Field(...)
    user: str = pydantic.Field(...)

class ProcessData(pydantic.BaseModel):
    points3D: typing.List[typing.Tuple[int, int, int]] = pydantic.Field(...)
    viewType: str = pydantic.Field(...)
    scribbleType: str = pydantic.Field(...)
    # viewType: typing.Literal[KEY_AXIAL, KEY_SAGITTAL, KEY_CORONAL] = pydantic.Field(...)  # Example allowed values
    # scribbleType: typing.Literal['type1', 'type2'] = pydantic.Field(...)  # Example allowed values
    caseName: str = pydantic.Field(...)
    timeToScribble: float = pydantic.Field(...)
    scribbleStartEpoch: str = pydantic.Field(...)

# Incoming packet
class PayloadProcess(pydantic.BaseModel):
    data: ProcessData = pydantic.Field(...)
    identifier: str = pydantic.Field(...)
    user: str = pydantic.Field(...)

class PayloadCloseSession(pydantic.BaseModel):
    identifier: str = pydantic.Field(...) # random name generated for each webpage 
    user: str = pydantic.Field(...) # firstname-lastname-role-mode

class ScrolledSliceIdxsData(pydantic.BaseModel):
    caseName: str = pydantic.Field(...)
    scrolledSliceIdxsObj: typing.Dict[str, object] = pydantic.Field(...)

# Incoming packet
class PayloadScrolledSliceIdxs(pydantic.BaseModel):
    data: ScrolledSliceIdxsData = pydantic.Field(...)
    identifier: str = pydantic.Field(...)
    user: str = pydantic.Field(...)

def getClientIdentifier(identifier, user):
    try:
        return identifier + '__' + user
    
    except:
        traceback.print_exc()
        return 'Meh'
    
#################################################################
#                        NNET MODELS
#################################################################

class ModelWithSigmoidAndThreshold(torch.nn.Module):

    def __init__(self, model, threshold=0.5):
        super(ModelWithSigmoidAndThreshold, self).__init__()
        self.model     = model
        self.threshold = threshold

    def forward(self, x):

        y = self.model(x) # [B,C=1, H,W,D]
        y = torch.sigmoid(y)
        y = torch.where(y <= self.threshold, torch.tensor(0.0), torch.tensor(1.0))
        return x

def sigmoidAndThresholdForward(self, x, threshold=0.5):
    """
    This will only work with Monai's UNet model becuase its forward function is very simple: https://docs.monai.io/en/0.4.0/_modules/monai/networks/nets/unet.html
    NOTE: No print, time.time() functions here since torch.onnx.dynamo_export fails!
    """
    y = self.model(x) # [B,C=1, H,W,D]
    y = torch.sigmoid(y)
    y = torch.round(y)
    # y = torch.where(y <= threshold, torch.tensor(0.0), torch.tensor(1.0))
    return y

def getModel(modelName, device=None):

    model = None

    try:

        # Step 1 - Get neural arch
        if modelName == KEY_UNET_V1:
            # https://docs.monai.io/en/stable/networks.html#unet
            # https://github.com/Project-MONAI/MONAI/blob/1.3.1/monai/networks/nets/unet.py#L30
            model = monai.networks.nets.UNet(in_channels=5, out_channels=1, spatial_dims=3, channels=[16, 32, 64, 128], strides=[2, 2, 2], num_res_units=2) # [CT,PET,Pred,Fgd,Bgd] --> [Refined-Pred] # 1.2M params

        # Step 99 - Move to device
        if device is not None:
            model = model.to(device)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return model

def loadModel(modelPath, modelName=None, model=None, device=None, loadOnnx=False):  
    
    loadedModel = None
    ortSession  = None

    try:

        # Step 1 - Get model
        if model is None and modelName is not None:
            model = getModel(modelName)

        # Step 2 - Load the model
        checkpoint = None
        if model is not None:

            # Step 2.1 - Get checkpoint
            if Path(modelPath).exists():

                checkpoint = torch.load(modelPath, map_location=device, weights_only=True)
                print ('\n - [loadModel()] Setting weights_only=True\n')

                if KEY_MODEL_STATE_DICT in checkpoint:
                    model.load_state_dict(checkpoint[KEY_MODEL_STATE_DICT])
                else:
                    model.load_state_dict(checkpoint)
            
                # Step 2.2 - Add steps for a) move to device b) post-processing, c) eval mode and d) warm-up
                # model = ModelWithSigmoidAndThreshold(model, threshold=0.5) # does not export the unet model weights when exporting to onnx format
                if device is not None:
                    model = model.to(device)
                model.forward = sigmoidAndThresholdForward.__get__(model, monai.networks.nets.UNet) # bind a method (sigmoidAndThresholdForward) to an instance of a class
                model.eval()
                randomInput     = torch.randn(SHAPE_TENSOR, device=device)
                
                # Step 2.3 - Check for onnx loading
                if not loadOnnx:
                    _ = model(randomInput) # warm-up
                
                else:
                    
                    # Step 2.3.1 - Convert existing model to onnx
                    try:
                        modelOnnx = torch.onnx.dynamo_export(model, randomInput) # type(loadedModel) == torch.onnx.ONNXProgram
                    except:
                        traceback.print_exc()
                        pdb.set_trace()
                    # Step 2.3.2 - Make sure .onnx model exists
                    modelPathOnnx = Path(modelPath).with_suffix(KEY_EXT_ONNX)
                    # Path(modelPathOnnx).unlink(missing_ok=True)
                    if not Path(modelPathOnnx).exists(): # NOTE: why do I do this if I am not even using it?
                        convertToOnnxAndSaveToDisk(modelOnnx, modelPathOnnx)
                    
                    # Step 2.3.3 - Get onnxruntime session
                    ortSession = onnxruntime.InferenceSession(modelPathOnnx, providers=['CPUExecutionProvider'])

                    # Step 2.3.4 - Warmpup
                    randomInputOnnx        = modelOnnx.adapt_torch_inputs_to_onnx(randomInput) # [B,C,H,W,D] --> ([B,C,H,W,D],) essentially a tuple
                    randomInputOnnxRuntime = {k.name: to_numpy(v) for k, v in zip(ortSession.get_inputs(), randomInputOnnx)}
                    _ = ortSession.run(None, randomInputOnnxRuntime) # warm-up

                    if 0:
                    
                        if 0:

                            randomInputOnnx        = modelOnnx.adapt_torch_inputs_to_onnx(randomInput) # [B,C,H,W,D] --> ([B,C,H,W,D],) essentially a tuple
                            randomInputOnnxRuntime = {k.name: to_numpy(v) for k, v in zip(ortSession.get_inputs(), randomInputOnnx)}

                            print (' - [loadModel()] ONNX Inference time: ', timeit.timeit(lambda: ortSession.run(None, randomInputOnnxRuntime), number=10))
                            t0 = time.time()
                            randomOutputOnnxRuntime = ortSession.run(None, randomInputOnnxRuntime)
                            print (' - [loadModel()] ONNX Inference time: ', time.time() - t0)

                            print (' - [loadModel()] Torch Inference time: ', timeit.timeit(lambda: model(randomInput), number=10))
                            t0 = time.time()
                            randomOutputTorch = model(randomInput)
                            print (' - [loadModel()] Torch Inference time: ', time.time() - t0)

                            print (' - [loadModel()] ONNX Output: ', randomOutputOnnxRuntime[0].max(), randomOutputOnnxRuntime[0].sum(), type(randomOutputOnnxRuntime[0]))
                            print (' - [loadModel()] Torch Output: ', randomOutputTorch.max(), randomOutputTorch.sum(), type(randomOutputTorch))
                            difference = np.abs(randomOutputOnnxRuntime[0] - to_numpy(randomOutputTorch)).sum()
                            print (' - [loadModel()] Difference: ', difference)
                            pdb.set_trace()
                        
                        elif 1:
                            print (' - [loadModel()] ONNX Inference time: ', timeit.timeit(lambda: doInferenceNew(modelOnnx, ortSession, randomInput), number=10))
                            t0 = time.time()
                            randomOutputTorchOnnxRuntime, randomOutputNumpyOnnxRuntime = doInferenceNew(modelOnnx, ortSession, randomInput)
                            print (' - [loadModel()] ONNX Inference time: ', time.time() - t0)

                            print (' - [loadModel()] Torch Inference time: ', timeit.timeit(lambda: doInferenceNew(model, None, randomInput), number=10))
                            t0 = time.time()
                            randomOutputTorch, randomOutputNumpy = doInferenceNew(model, None, randomInput)
                            print (' - [loadModel()] Torch Inference time: ', time.time() - t0)

                            print (' - [loadModel()] ONNX Output: ', randomOutputNumpyOnnxRuntime.shape, randomOutputNumpyOnnxRuntime.max(), randomOutputNumpyOnnxRuntime.sum(), type(randomOutputNumpyOnnxRuntime))
                            print (' - [loadModel()] non-ONNX Output: ', randomOutputNumpy.shape, randomOutputNumpy.max(), randomOutputNumpy.sum(), type(randomOutputNumpy))
                            difference = np.abs(randomOutputNumpyOnnxRuntime - randomOutputNumpy).sum()
                            print (' - [loadModel()] Difference: ', difference)
            
            else:
                print (' - [loadModel()] Model not found at: ', modelPath)

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    if loadOnnx:
        loadedModel = modelOnnx
    else:
        loadedModel = model

    return loadedModel, ortSession

def loadModelUsingUserPath(device, expNameParam, epochParam, modelTypeParam, loadOnnx):
    """
    Step 1/3 - FInd model path and give to loadModel() function
      - Step 2: loadModel()
      - Step 3 - doInferenceNew()
    """

    model = None
    ortSession = None
    try:

        print ('\n =========================== [loadModelUsingUserPath()] =========================== \n')
    
        # Step 1 - Load model
        getMemoryUsage()
        modelPath = Path(DIR_MODELS) / expNameParam / EPOCH_STR.format(epochParam) / EPOCH_STR.format(epochParam)
        
        if Path(modelPath).exists():
            print (' - [loadModel()] Loading model from: ', modelPath)
            print (' - [loadModel()] Device  : ', device)
            print (' - [loadModel()] loadOnnx: ', loadOnnx)
            
            model, ortSession = loadModel(modelPath, modelTypeParam, device=device, loadOnnx=loadOnnx)
            if model is not None:
                getMemoryUsage()
            else:
                print (' - [loadModel()] Model not loaded')
                print (' - Exiting...')
                exit(0)
        
            print ('\n =========================== [loadModelUsingUserPath()] =========================== \n')
        
        else:
            print (' - [loadModel()] Model not found at: ', modelPath)
            print (' - Exiting...')
            exit(0)

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return model, ortSession

def doInferenceNew(model, ortSession, preparedDataTorch):
    """
    Params
    ------
    model            : torch.nn.Module
    ortSession       : onnxruntime.InferenceSession
    preparedDataTorch: torch.Tensor

    Returns
    -------
    segArrayRefinedTorch: torch.Tensor, shape=(1,5,H,W,D) # NOTE: is this (1,5,sagittal, coronal, axial) or (1,5,coronal, sagittal, axial)?
    segArrayRefinedNumpy: np.ndarray, shape=(H,W,D) --> to be used for .dcm creation --> upload to Orthanc --> display in cornerstone3D.js viewer

    Notes
    -----
     - make sure to use torch.inference_mode() to avoid autograd overhead
    """

    segArrayRefinedNumpy = None
    segArrayRefinedTorch = None
    try:
        if model is not None:
            with torch.inference_mode():
                if ortSession is None:
                    segArrayRefinedTorch = model(preparedDataTorch)
                    segArrayRefinedNumpy = to_numpy(segArrayRefinedTorch)[0,0]
                else:
                    tOnnxInferenceStart = time.time()
                    preparedDataOnnx         = model.adapt_torch_inputs_to_onnx(preparedDataTorch) # very little time on linux cpu
                    preparedDataOnnxRuntime  = {k.name: to_numpy(v) for k, v in zip(ortSession.get_inputs(), preparedDataOnnx)}
                    tOnnxConversion = time.time()
                    segArrayRefinedNumpy     = ortSession.run(None, preparedDataOnnxRuntime)[0] # very high time on linux cpu ~ 0.7s
                    tOnnxInference = time.time()
                    segArrayRefinedNumpy     = segArrayRefinedNumpy[0,0]
                    # print (' - [doInferenceNew()] Time taken for ORT inference: {:.2f}s (conversion={:.2f}s + inference={:.2f}s)'.format(time.time() - tOnnxInferenceStart, tOnnxConversion-tOnnxInferenceStart, tOnnxInference-tOnnxConversion))
                    tDevice = time.time()
                    segArrayRefinedTorch     = torch.tensor(segArrayRefinedNumpy, device=DEVICE)
                    # print (' - [doInferenceNew()] Time taken to move to device: ', time.time() - tDevice) # very little time (to send to linux cpu)
        else:
            print (' - [doInferenceNew()] Model is None!')

    except:
        traceback.print_exc()
        if MODE_DEBUG: 
            print (' - [doInferenceNew()] Error in inference')
            getMemoryUsage()
            pdb.set_trace()

    return segArrayRefinedTorch, segArrayRefinedNumpy

def doInference(model, ortSession, preparedDataTorch):
    """
    Defunct function. DO not use.
    """
    segArrayRefinedNumpy = None
    segArrayRefinedTorch = None
    try:
        if model is not None:
            with torch.inference_mode():
                if ortSession is None:
                    segArrayRefinedTorch  = model(preparedDataTorch)
                    segArrayRefinedTorch  = torch.sigmoid(segArrayRefinedTorch).detach()
                    segArrayRefinedTorch[segArrayRefinedTorch <= 0.5] = 0
                    segArrayRefinedTorch[segArrayRefinedTorch > 0.5] = 1
                    segArrayRefinedNumpy = segArrayRefinedTorch.cpu().numpy()[0,0]
                else:
                    preparedDataOnnx = model.adapt_torch_inputs_to_onnx(preparedDataTorch)
                    preparedDataOnnxRuntime = {k.name: to_numpy(v) for k, v in zip(ORT_SESSION.get_inputs(), preparedDataOnnx)}
                    segArrayRefinedOnnxRuntime = ortSession.run(None, preparedDataOnnxRuntime)
                    pdb.set_trace()

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()

    return segArrayRefinedTorch, segArrayRefinedNumpy

def convertToOnnxAndSaveToDisk(modelOnnx, modelPathOnnx):
    """
    Params
    ------
    model        : torch.Module 
    modelPathOnnx: Path
    randomInput  : torch.Tensor
    """

    try:
        
        modelOnnx.save(str(modelPathOnnx))

    except:
        traceback.print_exc()
        pdb.set_trace()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def fillHolesIn3DBinaryMask(binaryMask):
    """
    binaryMask: np.ndarray, shape=(H,W,D), (coronal, sagittal, axial)
    """
    try:
        
        binaryMaskSitk   = sitk.GetImageFromArray(binaryMask.astype(np.uint8))
        holeFilter       = sitk.BinaryFillholeImageFilter()
        binaryMaskFilled = holeFilter.Execute(binaryMaskSitk)
        binaryMask       = sitk.GetArrayFromImage(binaryMaskFilled)

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return binaryMask

def fillHolesIn2DBinaryMaskByView(binaryMask, viewType):
    """
    Params
    ------
        binaryMask: np.ndarray, shape=(H,W,D), (coronal, sagittal, axial)
        viewType  : str, one of [KEY_AXIAL, KEY_SAGITTAL, KEY_CORONAL]
    """
    try:
        # Step 0 - Init
        binaryMaskFilled = np.zeros_like(binaryMask, dtype=np.uint8) # empty array to store the filled binary mask
        holeFilter       = sitk.BinaryFillholeImageFilter()

        # Step 0.1 - Init (get sliceIdxs)
        if viewType == KEY_AXIAL:
            sliceIdxs = np.argwhere(np.sum(binaryMask, axis=(0,1))).flatten() # get the slice with the most number of pixels
        elif viewType == KEY_SAGITTAL:
            sliceIdxs = np.argwhere(np.sum(binaryMask, axis=(0,2))).flatten()
        elif viewType == KEY_CORONAL:
            sliceIdxs = np.argwhere(np.sum(binaryMask, axis=(1,2))).flatten()
        # print (f' - [fillHolesInBinaryMaskByAxialSlices()][view={viewType}] sliceIdxs: ', sliceIdxs)

        # Step 1 - Iterate through each slice along the axial view
        # for i in range(binaryMask.shape[2]):
        for i in sliceIdxs:
            
            # Convert the 2D slice to a SimpleITK image
            if viewType == KEY_AXIAL:
                slice_sitk = sitk.GetImageFromArray(binaryMask[:, :, i].astype(np.uint8))
            elif viewType == KEY_SAGITTAL:
                slice_sitk = sitk.GetImageFromArray(binaryMask[:, i, :].astype(np.uint8))
            elif viewType == KEY_CORONAL:
                slice_sitk = sitk.GetImageFromArray(binaryMask[i, :, :].astype(np.uint8))
            
            # Apply the BinaryFillholeImageFilter to the 2D slice
            slice_filled_sitk = holeFilter.Execute(slice_sitk)
            
            # Convert the filled 2D slice back to a NumPy array
            slice_filled = sitk.GetArrayFromImage(slice_filled_sitk)
            
            # Store the filled 2D slice in the corresponding position of the output array
            if viewType == KEY_AXIAL:
                binaryMaskFilled[:, :, i] = slice_filled
            elif viewType == KEY_SAGITTAL:
                binaryMaskFilled[:, i, :] = slice_filled
            elif viewType == KEY_CORONAL:
                binaryMaskFilled[i, :, :] = slice_filled

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return binaryMaskFilled

#################################################################
#                           DCM SERVER
#################################################################

def getDCMClient(wadoRsRoot):
    
    client = None

    try:

        # Step 1 - Init
        dicomSessionObj = requests.Session()
        dicomSessionObj.auth = ('orthanc', 'orthanc')
        client            = dicomweb_client.api.DICOMwebClient(url=DCM_SERVER_URL, session=dicomSessionObj)

    except:
        traceback.print_exc()
    
    return client

def getCTArray(client, patientData):
    """
    This function fills in patientData[KEY_TORCH_DATA][:,0] with the CT array
     - for instance in ctInstances:ctArray[:, :, int(instance.InstanceNumber)-1] = instance.pixel_array
      - so this makes [H,W,Depth]
    """
    
    ctArray, ctArrayProcessed, ctArrayProcessedBool = None, None, False
    patientName = None

    try:

        # Step 0 - Init
        patientName = patientData[KEY_DATA][KEY_CASE_NAME]
        preparedData = patientData[KEY_DATA]

        # Step 1 - Get CT instances
        ctInstances = client.retrieve_series(
            study_instance_uid=preparedData[KEY_SEARCH_OBJ_CT][KEY_STUDY_INSTANCE_UID],
            series_instance_uid=preparedData[KEY_SEARCH_OBJ_CT][KEY_SERIES_INSTANCE_UID]
        )

        # Step 2 - Sort instances
        ctInstances = sorted(ctInstances, key=lambda x: int(x.InstanceNumber))

        # Step 3 - Get CT array
        if len(ctInstances) == 0:
            print (' - [prepare()] No CT instances found')
            return ctArray, patientData
        
        ctArray = np.zeros((len(ctInstances), ctInstances[0].Rows, ctInstances[0].Columns), dtype=np.int16)
        # ctArray = np.zeros((ctInstances[0].Rows, ctInstances[0].Columns, len(ctInstances)), dtype=np.int16) # [NOTE: It should be this!]
        for instance in ctInstances:
            ctArray[:, :, int(instance.InstanceNumber)-1] = instance.pixel_array
        
        # Step 3.1 - Perform min-max crop and then z-normalization
        ctArrayProcessed = np.clip(copy.deepcopy(ctArray), HU_MIN, HU_MAX) # NOTE: make sure these numbers match with the ones used during training (in def getPatientData() -->  Step 1.8)
        ctArrayProcessed = (ctArrayProcessed - np.mean(ctArrayProcessed)) / np.std(ctArrayProcessed)

        # Step 4 - Update sessionsGlobal
        thisShapeTensor = list(copy.deepcopy(SHAPE_TENSOR))
        thisShapeTensor[2] = ctArray.shape[0]
        thisShapeTensor[3] = ctArray.shape[1]
        thisShapeTensor[4] = ctArray.shape[2]
        
        patientData[KEY_TORCH_DATA]                = torch.zeros(thisShapeTensor, dtype=torch.float32, device=DEVICE)
        patientData[KEY_TORCH_DATA][0, 0, :, :, :] = torch.tensor(ctArrayProcessed, dtype=torch.float32, device=DEVICE)
        patientData[KEY_DCM_LIST]                  = ctInstances
        patientData[KEY_SCRIBBLE_MAP]              = np.zeros_like(ctArray)

        ctArrayProcessedBool = True

    except:
        print (' - [getCTArray()] Could not get CT array for patient: ', patientName)
        print ('    --------------------------- CT ERROR ---------------------------')
        traceback.print_exc()
        print ('    --------------------------- CT ERROR ---------------------------')
    
    return ctArrayProcessedBool, ctArray, ctArrayProcessed, patientData

def getPTArray(client, patientData):
    """
    This function fills in patientData[KEY_TORCH_DATA][:,1] with the PT array
    """
    
    ptArray = None

    try:

        # Step 0 - Init
        preparedData = patientData[KEY_DATA]
        patientName  = patientData.get(KEY_PATIENT_NAME, '')

        # Step 1 - Get PT instances
        ptInstances = client.retrieve_series(
            study_instance_uid=preparedData[KEY_SEARCH_OBJ_PET][KEY_STUDY_INSTANCE_UID],
            series_instance_uid=preparedData[KEY_SEARCH_OBJ_PET][KEY_SERIES_INSTANCE_UID]
        )

        # Step 2 - Sort instances
        ptInstances = sorted(ptInstances, key=lambda x: int(x.InstanceNumber))

        # Step 3 - Get PT array
        if len(ptInstances) == 0:
            print (' - [getPTArray()] No PT instances found')
            return ptArray, patientData
        
        ptArray = np.zeros((len(ptInstances), ptInstances[0].Rows, ptInstances[0].Columns), dtype=np.int16)
        for instance in ptInstances:
            ptArray[:, :, int(instance.InstanceNumber)-1] = instance.pixel_array
        
        # Step 3.1 - Perform min-max crop and then z-normalization
        if ptArray.max() > 1000:
            print (f' - [getPTArray({patientName})] Using np.clip(SUV_MIN_v1:{SUV_MIN_v1}, SUV_MAX_v1:{SUV_MAX_v1}) as ptArray.max() > 1000: {ptArray.max()}')
            ptArrayProcessed = np.clip(copy.deepcopy(ptArray), SUV_MIN_v1, SUV_MAX_v1) # NOTE: make sure these numbers match with the ones used during training (in def getPatientData() -->  Step 1.8)
        else:
            ptArrayProcessed = np.clip(copy.deepcopy(ptArray), SUV_MIN_v2, SUV_MAX_v2)
        ptArrayProcessed = (ptArrayProcessed - np.mean(ptArrayProcessed)) / np.std(ptArray)

        # Step 4 - Update sessionsGlobal
        patientData[KEY_TORCH_DATA][0, 1, :, :, :] = torch.tensor(ptArrayProcessed, dtype=torch.float32, device=DEVICE)
        
    except:
        traceback.print_exc()
    
    return ptArray, ptArrayProcessed, patientData

def getSEGs(client, patientData): # preparedData, sessionsGlobal, clientIdentifier, debug=False):
    """
    This function fills in patientData[KEY_TORCH_DATA][:,2] with the SEG-GT array
     - also patientData[KEY_SEG_ARRAY_GT] with the SEG-GT array
    """
    segArrayGT   = None
    segArrayPred = None

    try:

        # Step 0 - Init
        preparedData = patientData[KEY_DATA]

        # Step 1 - Get SEG-GT instance
        studyInstanceUIDGT = preparedData[KEY_SEARCH_OBJ_RTSGT][KEY_STUDY_INSTANCE_UID]
        if studyInstanceUIDGT != '' and studyInstanceUIDGT != None:

            try:
                segInstanceGT = client.retrieve_instance(
                    study_instance_uid=preparedData[KEY_SEARCH_OBJ_RTSGT][KEY_STUDY_INSTANCE_UID],
                    series_instance_uid=preparedData[KEY_SEARCH_OBJ_RTSGT][KEY_SERIES_INSTANCE_UID],
                    sop_instance_uid=preparedData[KEY_SEARCH_OBJ_RTSGT][KEY_SOP_INSTANCE_UID]
                )

                # Step 1.2 - Read GT array
                reader = pydicom_seg.SegmentReader()
                resultGT = reader.read(segInstanceGT)

                for segment_number in resultGT.available_segments:
                    segArrayGT = resultGT.segment_data(segment_number)  # directly available
                    segArrayGT = np.moveaxis(segArrayGT, [0,1,2], [2,1,0]) # (axial, coronal, sagittal) --> (sagittal, coronal, axial)
                    # NOTE: Dirty hack to make the orientation of the SEG correct 
                    for idx in range(segArrayGT.shape[2]):
                        segArrayGT[:,:,idx] = np.rot90(segArrayGT[:,:,idx], k=1)
                        segArrayGT[:,:,idx] = np.flipud(segArrayGT[:,:,idx])

            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print (' - [getSEGs(studyUID={})] No SEG-GT instance found'.format(studyInstanceUIDGT))
        
        # Step 2 - Get SEG-Pred instance
        studyInstanceUIDPred = preparedData[KEY_SEARCH_OBJ_RTSPRED][KEY_STUDY_INSTANCE_UID]
        if studyInstanceUIDPred != '' and studyInstanceUIDPred != None:
            try:
                segInstancePred = client.retrieve_instance(
                    study_instance_uid=studyInstanceUIDPred,
                    series_instance_uid=preparedData[KEY_SEARCH_OBJ_RTSPRED][KEY_SERIES_INSTANCE_UID],
                    sop_instance_uid=preparedData[KEY_SEARCH_OBJ_RTSPRED][KEY_SOP_INSTANCE_UID]
                )

                # Step 2.2 - Read Pred array
                reader = pydicom_seg.SegmentReader()
                resultPred = reader.read(segInstancePred)

                for segment_number in resultPred.available_segments:
                    segArrayPred = resultPred.segment_data(segment_number)
                    segArrayPred = np.moveaxis(segArrayPred, [0,1,2], [2,1,0]) # [z,y,x] --> [x,y,z]
                    # NOTE: Dirty hack to make the orientation of the SEG correct
                    for idx in range(segArrayPred.shape[2]):
                        segArrayPred[:,:,idx] = np.rot90(segArrayPred[:,:,idx], k=1)
                        segArrayPred[:,:,idx] = np.flipud(segArrayPred[:,:,idx])
                                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    print (' - [getSEGs(studyUID={})] No SEG-Pred instance found'.format(studyInstanceUIDPred))
            
        # Step 3 - Update sessionsGlobal
        if segArrayPred is not None:
            patientData[KEY_TORCH_DATA][0, 2, :, :, :] = torch.tensor(segArrayPred, dtype=torch.float32, device=DEVICE)
        
        if segArrayGT is not None:
            patientData[KEY_SEG_ARRAY_GT] = segArrayGT # store as numpy array
        
    except:
        traceback.print_exc()
    
    return segArrayGT, segArrayPred, patientData

def plotHistograms(ctArray, ctArrayProcessed, ptArray, ptArrayProcessed, segArrayGT, segArrayPred, patientName, saveFolderPath):
    
        try:
    
            # Step 1 - Plot histograms
            f,axarr = plt.subplots(3,2, figsize=(10,10))
            axarr[0,0].hist(ctArray.flatten(), bins=100, color='black', alpha=0.5, label='CT')
            axarr[0,0].set_title('CT')
            axarr[0,1].hist(ptArray.flatten(), bins=100, color='black', alpha=0.5, label='PT')
            axarr[0,1].set_title('PT')
            axarr[1,0].hist(ctArrayProcessed.flatten(), bins=100, color='black', alpha=0.5, label='CT-Processed')
            axarr[1,0].set_title('CT-Processed')
            axarr[1,1].hist(ptArrayProcessed.flatten(), bins=100, color='black', alpha=0.5, label='PT-Processed')
            axarr[1,1].set_title('PT-Processed')
            axarr[2,0].hist(segArrayGT.flatten(), bins=100, color='black', alpha=0.5, label='SEG-GT')
            axarr[2,0].set_title('SEG-GT')
            axarr[2,1].hist(segArrayPred.flatten(), bins=100, color='black', alpha=0.5, label='SEG-Pred')
            axarr[2,1].set_title('SEG-Pred')
            plt.suptitle(patientName)

            Path(saveFolderPath).mkdir(parents=True, exist_ok=True)
            plt.savefig(str(Path(saveFolderPath).joinpath(patientName + '_histograms.png')), bbox_inches='tight')
    
        except:
            traceback.print_exc()
            if MODE_DEBUG: pdb.set_trace()

def postInstanceToOrthanc(requestBaseURL, dcmPath):
    """
    Here we post the dcmPath file to the Orthanc server
    Params
    ------
    requestBaseURL: str
    dcmPath: str

    Returns
    -------
    postDICOMStatus: bool
    instanceOrthanID: str
    postInstanceStatus: str, ['Success', 'AlreadyStored']
    """

    # Step 0 - Init
    postDICOMStatus = False
    instanceOrthanID = None
    postInstanceStatus = ''

    try:

        # Step 1 - Post .dcm(modality=SEG) instance
        sendInstanceURL = requestBaseURL + '/instances'
        with open(dcmPath, 'rb') as file:

            # Step 1.1 - Read .dcm
            tReadStart = time.time()
            dcmPathContent     = file.read()
            tReadTotal = time.time() - tReadStart

            # Step 1.2 - Send .dcm
            tSendStart = time.time()
            postRequestObj       = requests.post(sendInstanceURL, data=dcmPathContent)
            tSendTotal = time.time() - tSendStart
            
            # Step 1.3 - Parse response
            if postRequestObj.status_code == 200:
                postRequestResponse = postRequestObj.json()
                postInstanceStatus   = postRequestResponse['Status'] # ['Success', 'AlreadyStored']
                instanceOrthanID     = postRequestResponse['ID']
                postDICOMStatus = True
            else:
                print (f' - [postInstanceToOrthanc()][code={postRequestObj.status_code}, Reason={postRequestObj.reason}] Could not post instance: {postRequestObj.text}')
                if MODE_DEBUG: pdb.set_trace() 
            # print (' - [postInstanceToOrthanc()] Read time: {:.4f} seconds, Send time: {:.4f} seconds'.format(tReadTotal, tSendTotal))
                
    except:
        traceback.print_exc()
        print (' - [makeSEGDicom()] Could not post instance')
    
    return postDICOMStatus, instanceOrthanID, postInstanceStatus

def deleteInstanceFromOrthanc(requestBaseURL, instanceOrthanID):

    deleteInstanceStatus = False

    try:
        deleteInstanceURL = requestBaseURL + '/instances/' + str(instanceOrthanID)
        deleteResponse = requests.delete(deleteInstanceURL)
        if deleteResponse.status_code == 404:
            print (' - [makeSEGDicom()] Instance not found: ', deleteInstanceURL)
            pass # instance not found
        if deleteResponse.status_code == 200:
            # print (' - [makeSEGDicom()] Instance deleted')
            deleteInstanceStatus = True
            pass # instance deleted
    except:
        traceback.print_exc()
        print (' - [makeSEGDicom()] Could not delete instance')
    
    return deleteInstanceStatus

def addAndGetPrivateBlockToDcm(ds):
    
    block = None

    try:

        block = ds.private_block(PRIVATE_BLOCK_GROUP, PRIVATE_BLOCK_CREATOR, create=True)

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return block

def makeSEGDicom(maskArray, patientSessionData, viewType, sliceId, scribbleStartEpoch, timeToScribble, tDistMap, tInfer):
    """
    Params
    ------
    maskArray: np.ndarray, [H,W,Depth] (sagittal, coronal, axial)
    patientSessionData: dict
    viewType: str
    sliceId: int
    timeToScribble: float, time taken to scribble
    tDistMap: float, time taken to make distance map
    tInfer: float, time taken to make inference

    Notes
    -----
    1. This function is called by "async def process()", so this uploads the ai-based contour refinement.
     - For manual refinement refer to "async def uploadManualRefinement()"
    """

    makeDICOMStatus = False
    tDCMMakeTotal, tPostTotal = -1, -1
    try:

        # Step 1 - Make dicom (and save to disk)
        if 1:
            # Step 0 - Init
            tDCMMakeStart = time.time()
            def set_segment_color(ds, segment_index, rgb_color):

                def rgb_to_cielab(rgb):
                    import skimage
                    import skimage.color
                    # Normalize RGB values to the range 0-1
                    rgb_normalized = np.array(rgb) / 255.0
                    # Convert RGB to CIELab
                    cielab = skimage.color.rgb2lab(np.array([rgb_normalized]))
                    return cielab.flatten()
                
                # Convert RGB to DICOM CIELab
                cielab = rgb_to_cielab(rgb_color)
                # DICOM CIELab values need to be scaled and converted to unsigned 16-bit integers
                L_star = int((cielab[0] / 100) * 65535)  # L* from 0 to 100
                a_star = int(((cielab[1] + 128) / 255) * 65535)  # a* from -128 to +127
                b_star = int(((cielab[2] + 128) / 255) * 65535)  # b* from -128 to +127
                
                # Set the color for the specified segment
                if 'SegmentSequence' in ds:
                    segment = ds.SegmentSequence[segment_index]
                    segment.RecommendedDisplayCIELabValue = [L_star, a_star, b_star]
                
                # Save the modified DICOM file
                return ds

            floatify = lambda x: [float(each) for each in x] 
            patientName       = patientSessionData[KEY_DATA][KEY_CASE_NAME]
            ctDicomsList      = patientSessionData[KEY_DCM_LIST]
            pathFolderMask    = patientSessionData[KEY_PATH_SAVE]
            counter           = patientSessionData[KEY_SCRIBBLE_COUNTER]
            sopInstanceUID    = patientSessionData[KEY_SEG_SOP_INSTANCE_UID]
            seriesInstanceUID = patientSessionData[KEY_SEG_SERIES_INSTANCE_UID]

            # Step 1 - Convert to sitk image
            dsCT        = ctDicomsList[0]
            maskSpacing = floatify(dsCT.PixelSpacing) + [float(dsCT.SliceThickness)]
            maskOrigin  = floatify(dsCT.ImagePositionPatient)
            if 0:
                sliceId = 72
                f,axarr = plt.subplots(1,3)
                axarr[0].imshow(maskArray[:,:,sliceId], cmap='gray'); axarr[0].set_title('maskArray[:,:,{}]'.format(sliceId))
                axarr[1].imshow(np.moveaxis(maskArray, [0,1,2], [2,1,0])[sliceId,:,:], cmap='gray'); axarr[1].set_title('np.moveaxis(maskArray, [0,1,2], [2,1,0])[sliceId,:,:]')
                axarr[2].imshow(np.moveaxis(maskArray, [0,1,2], [1,2,0])[sliceId,:,:], cmap='gray'); axarr[2].set_title('np.moveaxis(maskArray, [0,1,2], [1,2,0])[sliceId,:,:]')
                plt.show()

            maskArrayCopy = copy.deepcopy(maskArray)
            for idx in range(maskArrayCopy.shape[2]):
                maskArrayCopy[:,:,idx] = np.flipud(maskArrayCopy[:,:,idx]) # NOTE: only applies to AI-assisted refinement, maybe not for manual
                maskArrayCopy[:,:,idx] = np.rot90(maskArrayCopy[:,:,idx], k=3)
                
            maskArrayForImage = np.moveaxis(maskArrayCopy, [0,1,2], [2,1,0]); # print (" - Doing makeSEGDICOM's np.moveaxis() as always") # np([H,W,D]) -> np([D,W,H]) -> sitk([H,W,D])
            maskImage   = sitk.GetImageFromArray(maskArrayForImage.astype(np.uint8)) # np([H,W,D]) -> np([D,W,H]) -> sitk([H,W,D])
            maskImage.SetSpacing(maskSpacing)
            maskImage.SetOrigin(maskOrigin)
            
            # Step 2 - Create a basic dicom dataset        
            template                    = pydicom_seg.template.from_dcmqi_metainfo(Path(DIR_ASSETS) / FILENAME_METAINFO_SEG_JSON)
            if MODE_DEBUG:
                template.SeriesDescription  = fileNameForSave(patientName, counter, str(viewType), int(sliceId))  # '-'.join([patientName, SERIESDESC_SUFFIX_REFINE, str(counter)])
            else:
                template.SeriesDescription  = '-'.join([patientName, SERIESDESC_SUFFIX_REFINE, Path(pathFolderMask).parts[-1], '{:03d}'.format(counter)])
            template.SeriesNumber       = SERIESNUM_REFINE
            template.ContentCreatorName = CREATORNAME_REFINE
            # template.ContentLabel       = maskType
            writer                      = pydicom_seg.MultiClassWriter(template=template, inplane_cropping=False, skip_empty_slices=False, skip_missing_segment=False)
            dcm                         = writer.write(maskImage, ctDicomsList)
            # print (' - rows: {} | cols: {} | numberofframes:{}'.format(dcm.Rows, dcm.Columns, dcm.NumberOfFrames))
            
            # Step 3 - Set UIDs specific to patient
            set_segment_color(dcm, 0, [255, 192, 203]) # pink
            dcm.StudyInstanceUID        = dsCT.StudyInstanceUID
            dcm.SeriesInstanceUID       = seriesInstanceUID
            dcm.SOPInstanceUID          = sopInstanceUID

            privateBlock = addAndGetPrivateBlockToDcm(dcm)
            privateBlock.add_new(TAG_TIME_TO_SCRIBBLE, VALUEREP_FLOAT32, float(timeToScribble))
            privateBlock.add_new(TAG_TIME_TO_DISTMAP, VALUEREP_FLOAT32, float(tDistMap))
            privateBlock.add_new(TAG_TIME_TO_INFER, VALUEREP_FLOAT32, float(tInfer))
            privateBlock.add_new(TAG_TIME_EPOCH, VALUEREP_STRING, scribbleStartEpoch)
            print ('  | - [makeSEGDicom]() timeToScribble: {:.4f} | tDistMap: {:.4f} | tInfer: {:.4f} | scribbleStartEpoch: {}'.format(
                dcm[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_SCRIBBLE)].value
                , dcm[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_DISTMAP)].value
                , dcm[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_INFER)].value
                , epoch_to_datetime(dcm[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_EPOCH)].value)
            ))
            
            # Step 4 - Save the dicom file 
            Path(pathFolderMask).mkdir(parents=True, exist_ok=True)
            dcmPath = str(Path(pathFolderMask).joinpath('-'.join([patientName, SUFIX_REFINE, '{:03d}'.format(counter)]) + '.dcm'))
            tWriteStart = time.time()
            dcm.save_as(dcmPath)
            tWriteTotal = time.time() - tWriteStart
            print ('  | - [makeSEGDicom()] Saving SEG with SeriesDescription: ', dcm.SeriesDescription)
            tDCMMakeTotal = time.time() - tDCMMakeStart
        
        # Step 4 - Post to DICOM server
        if 1:
            tPostStart = time.time()
            instanceOrthancID = patientSessionData[KEY_SEG_ORTHANC_ID]
            global DCMCLIENT
            if DCMCLIENT is not None:
                requestBaseURL = str(DCMCLIENT.protocol) + '://' + str(DCMCLIENT.host) + ':' + str(DCMCLIENT.port)

                if instanceOrthancID is None:
                    # print (' - [makeSEGDicom()] First AI scribble for this patient. Posting SEG to DICOM server')
                    postDICOMStatus, instanceOrthancID, postInstanceStatus = postInstanceToOrthanc(requestBaseURL, dcmPath)
                    if postDICOMStatus:
                        if postInstanceStatus == 'AlreadyStored':
                            deleteInstanceStatus = deleteInstanceFromOrthanc(requestBaseURL, instanceOrthancID)
                            if deleteInstanceStatus:
                                postDICOMStatus, instanceOrthancID, postInstanceStatus = postInstanceToOrthanc(requestBaseURL, dcmPath)
                                if postDICOMStatus:
                                    patientSessionData[KEY_SEG_ORTHANC_ID] = instanceOrthancID # this is so that the dicom data is not crowded. Only the latest instance is stored
                                    makeDICOMStatus = True
                        else:
                            makeDICOMStatus = True
                    else:
                        print ('  | - [ERROR][makeSEGDicom()] Could not post SEG to DICOM server')
                
                elif instanceOrthancID is not None:
                    # print (' - [makeSEGDicom()] >1 AI scribble for this patient. Deleting existing SEG and posting new SEG to DICOM server')
                    deleteInstanceStatus = deleteInstanceFromOrthanc(requestBaseURL, instanceOrthancID)
                    if deleteInstanceStatus:
                        postDICOMStatus, instanceOrthancID, postInstanceStatus = postInstanceToOrthanc(requestBaseURL, dcmPath)
                        if postDICOMStatus:
                            patientSessionData[KEY_SEG_ORTHANC_ID] = instanceOrthancID
                            makeDICOMStatus = True
                    else:
                        print ('  | - [ERROR][makeSEGDicom()] Could not delete existing SEG from DICOM server')

            else:
                print ('  | - [ERROR][makeSEGDicom()] DCMCLIENT is None. Not posting SEG to DICOM server')
            
            tPostTotal = time.time() - tPostStart

        # Step 99 - Print time
        print ('  | - [makeSEGDicom()] Total time for make: {:.4f}s (write={:.4f}s), post: {:.4f}s'.format(tDCMMakeTotal, tWriteTotal, tPostTotal)) 

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()

    
    return makeDICOMStatus, patientSessionData

def getPatientUUIDs(patientID):

    seriesInstanceUUID = None
    sopInstanceUUID    = None
    try:
        # Step 0 - Init
        pathPatientsUUIDJson = DIR_ASSETS / FILENAME_PATIENTS_UUIDS_JSON
        Path(pathPatientsUUIDJson.parent).mkdir(parents=True, exist_ok=True)
        if not Path(pathPatientsUUIDJson).exists():
            with open(pathPatientsUUIDJson, 'w') as fp:
                json.dump({}, fp, indent=4)
        
        # Step 1.1 - Get data (if it exists)
        patientsUUIDs = {}
        with open(pathPatientsUUIDJson, 'r') as fp:
            patientsUUIDs = json.load(fp)

            # Step 1 - Get patient UUIDs
            if patientID in patientsUUIDs:
                seriesInstanceUUID = patientsUUIDs[patientID].get(KEY_SERIES_INSTANCE_UID, None)
                sopInstanceUUID    = patientsUUIDs[patientID].get(KEY_SOP_INSTANCE_UID, None)
            else:
                print (' - [getPatientUUIDs()] No patient found with patientID: ', patientID)

        # Step 2 - Make data (if it does not exist)
        if seriesInstanceUUID == None or sopInstanceUUID == None:
            seriesInstanceUUID, sopInstanceUUID = str(pydicom.uid.generate_uid()), str(pydicom.uid.generate_uid())
            with open(pathPatientsUUIDJson, 'w') as fp:
                patientsUUIDs[patientID] = {KEY_SERIES_INSTANCE_UID: seriesInstanceUUID, KEY_SOP_INSTANCE_UID: sopInstanceUUID}
                json.dump(patientsUUIDs, fp, indent=4)

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return seriesInstanceUUID, sopInstanceUUID

#################################################################
#                        DIST MAP UTILS
#################################################################

def getViewTypeAndSliceId(points3D, viewType):
    """
    points3D comes from cornerstone3D and  (or H,W,Depth)
    """
    sliceId = None
    try:
        
        if viewType is None:
            for viewIdx in [0,1,2]:
                points3DAtIdx = points3D[:,viewIdx]
                if np.unique(points3DAtIdx).shape[0] == 1:
                    if viewIdx == 0:
                        viewType = KEY_SAGITTAL
                    elif viewIdx == 1:
                        viewType = KEY_CORONAL
                    elif viewIdx == 2:
                        viewType = KEY_AXIAL
                    sliceId = points3DAtIdx[0]
                    break
        else:
            if viewType == KEY_AXIAL:
                sliceId = points3D[0][2]
            elif viewType == KEY_SAGITTAL:
                sliceId = points3D[0][0]
            elif viewType == KEY_CORONAL:
                sliceId = points3D[0][1]

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return viewType, sliceId

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

def getGaussianDistanceMapOld(ctArrayShape, points3D, distZ, sigma, viewType=None):

    gaussianDistanceMap = None
    sliceId = None
    try:
        
        # Step 0 - Identify viewType and sliceID
        if viewType is None:
            viewType, sliceId = getViewTypeAndSliceId(points3D, viewType)
        else:
            _, sliceId = getViewTypeAndSliceId(points3D, viewType)
        if viewType is None or sliceId is None:
            return gaussianDistanceMap

        # Step 1 - Put points3D in an array
        points3DInVolume = np.zeros(ctArrayShape)
        # points3DInVolume[points3D[:,0], points3D[:,1], points3D[:,2]] = 1
        points3DInVolume[points3D[:,1], points3D[:,0], points3D[:,2]] = 1

        # Step 2 - Get distance map
        if viewType == KEY_AXIAL     : sampling = (1,1,distZ)
        elif viewType == KEY_SAGITTAL: sampling = (distZ,1,1)
        elif viewType == KEY_CORONAL : sampling = (1,distZ,1)
        euclideanDistanceMap = scipy.ndimage.distance_transform_edt(1-points3DInVolume, sampling=sampling)
        maxVal               = euclideanDistanceMap.max()
        euclideanDistanceMap = 1 - (euclideanDistanceMap / maxVal)
        
        # Step 2 - Get gaussian distance map
        gaussianDistanceMap = np.exp(-(1-euclideanDistanceMap)**2 / (2 * sigma**2))

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return gaussianDistanceMap, viewType, sliceId

def getGaussianDistanceMap(scribbleMapData, distZ, sigma, callerFunc=''):
    """
    Params
    ------
    scribbleMapData: np.ndarray, [H,W,D], containing 1's and 0's
    distZ: float
    sigma: float
    """

    # Step 0 - Init
    gaussianDistanceMap = None

    try:
        
        # Step 1 - Get distance map
        scribbleDataSum = np.sum(scribbleMapData)
        # euclideanDistanceMap = scipy.ndimage.distance_transform_edt(1-scribbleMapData, sampling=distZ)
        # euclideanDistanceMap = scipy.ndimage.distance_transform_cdt(1-scribbleMapData); print (' - [getGaussianDistanceMap()] Doing cdt instead of edt ')
        if scribbleDataSum == 0:
            # print (f' - [getGaussianDistanceMap({callerFunc})][scribbleDataSum={scribbleDataSum}] Doing distance_transform_edt')
            euclideanDistanceMap = euclideanDistanceMap = scipy.ndimage.distance_transform_edt(1-scribbleMapData, sampling=distZ)
        else:
            # print (f' - [getGaussianDistanceMap(callerFunc={callerFunc})][scribbleDataSum={scribbleDataSum}] Doing edt.edt() instead of distance_transform_edt() ')
            euclideanDistanceMap = edt.edt(1-scribbleMapData, anisotropy=(distZ, distZ, distZ))
        
        maxVal               = euclideanDistanceMap.max()
        euclideanDistanceMap = 1 - (euclideanDistanceMap / maxVal)
        
        # Step 2 - Get gaussian distance map
        gaussianDistanceMap = np.exp(-(1-euclideanDistanceMap)**2 / (2 * sigma**2))
        
        # Step 3 - Check
        if np.any(scribbleMapData == 1):
            if not np.any(gaussianDistanceMap == 1):
                print (' - [getGaussianDistanceMap()] Found 1 in scribbleMapData, but no 1 in gaussianDistanceMap')

    except:
        print (f' - [ERROR][getGaussianDistanceMap(callerFunc={callerFunc})][scribbleDataSum={scribbleDataSum}] Could not get gaussian distance map')
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return gaussianDistanceMap

def getDistanceMap(scribbleMapData, preparedDataTorch, scribbleType, points3D, distMapZ, distMapSigma, viewType):
    """
    Params
    ------
    scribbleMapData: np.ndarray, [H,W,D], containing either VALUE_INT_FGD or VALUE_INT_BGD
    preparedDataTorch: torch.tensor, [1,5,H,W,D]
    scribbleType: str
    points3D: np.ndarray, [N,3]: its emptiness has already been checked in prepare(), gives points in [sagittal, coronal, axial]
    distMapZ: float
    distMapSigma: float
    """
    # Step 0 - Init
    sliceId = None # [NOTE: only for viz purposes]

    try:
        
        # Step 1 - Identify viewType and sliceID
        viewType, sliceId = getViewTypeAndSliceId(points3D, viewType=viewType) # [TODO: Do I need this?]

        # Step 2 - Update scribbleMapData
        # Step 2.1 - Remove points that are outside the range of scribbleMapData.shape (scribbleMapData.shape = [H,W,D] set in getCTArray())
        originalLen = points3D.shape[0]
        points3D = points3D[(points3D[:,0] >= 0) & (points3D[:,0] < scribbleMapData.shape[1])]
        points3D = points3D[(points3D[:,1] >= 0) & (points3D[:,1] < scribbleMapData.shape[0])]
        points3D = points3D[(points3D[:,2] >= 0) & (points3D[:,2] < scribbleMapData.shape[2])]
        finalLen = points3D.shape[0]
        if originalLen != finalLen:
            print (' - [getDistanceMap()] Removed {} points that were outside the range of scribbleMapData.shape'.format(originalLen-finalLen))
        
        # Step 2.2 - Update scribbleMapData
        if finalLen > 0:
            if scribbleType == KEY_SCRIBBLE_FGD:
                scribbleMapData[points3D[:,1], points3D[:,0], points3D[:,2]] = VALUE_INT_FGD
            elif scribbleType == KEY_SCRIBBLE_BGD:
                scribbleMapData[points3D[:,1], points3D[:,0], points3D[:,2]] = VALUE_INT_BGD

        # Step 2 - Get gaussian distance maps
        ctArrayShape   = tuple(preparedDataTorch[0,0].shape)
        fgdMap, bgdMap = np.zeros(ctArrayShape), np.zeros(ctArrayShape)

        # Step 2.1 - Get fgd map
        scribbleMapDataFgd = copy.deepcopy(scribbleMapData)
        scribbleMapDataFgd[scribbleMapDataFgd != VALUE_INT_FGD] = 0
        scribbleMapDataFgd[scribbleMapDataFgd == VALUE_INT_FGD] = 1
        fgdMap                 = getGaussianDistanceMap(scribbleMapDataFgd, distZ=distMapZ, sigma=distMapSigma, callerFunc='getDistanceMap().fgdMap')
        preparedDataTorch[0,3] = torch.tensor(fgdMap, dtype=torch.float32, device=DEVICE)
        
        # Step 2.2 - Get bgd map
        scribbleMapDataBgd = copy.deepcopy(scribbleMapData)
        scribbleMapDataBgd[scribbleMapDataBgd != VALUE_INT_BGD] = 0
        scribbleMapDataBgd[scribbleMapDataBgd == VALUE_INT_BGD] = 1
        bgdMap                 = getGaussianDistanceMap(scribbleMapDataBgd, distZ=distMapZ, sigma=distMapSigma, callerFunc='getDistanceMap().bgdMap')
        preparedDataTorch[0,4] = torch.tensor(bgdMap, dtype=torch.float32, device=DEVICE)

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()
    
    return points3D, scribbleMapData, preparedDataTorch, viewType, sliceId

#################################################################
#                        PLOTTING UTILS
#################################################################

def plotData(scribbleMapData, ctArray, ptArray, gtArray, predArray, refineArray, sliceId, caseName, counter, points3D, viewType, scribbleType, extraSlices=7, saveFolderPath=False):
    """
    Function is called in the following functions:
    - prepare()

    Params
    ------
    scribbleMapData: np.ndarray, [H,W,D], containing either VALUE_INT_FGD or VALUE_INT_BGD
    ctArray, ptArray, gtArray, predArray, refineArray: np.ndarray, [H,W,depth]
    sliceId: int
    caseName: str
    counter: int
    points3D: np.ndarray, [N,3]
    viewType: str
    scribbleType: str
    """

    points3DDistanceMap, sliceId = None, None
    pngName = None

    try:

        import matplotlib.colors
        import skimage.morphology
        import matplotlib.pyplot as plt
        

        # Step 0 - Define constants
        rotAxial    = lambda x: x
        rotSagittal = lambda x: np.rot90(x, k=1)
        rotCoronal  = lambda x: np.rot90(x, k=1)

        CMAP_DEFAULT      = plt.cm.Oranges
        RGBA_ARRAY_BLUE   = np.array([0   ,0 ,255,255])/255.
        RGBA_ARRAY_YELLOW = np.array([218,165,32 ,255])/255.

        # Step 0 - Identify viewType and sliceID
        points3DDistanceMap = None
        # viewType = None
        if points3D is not None:
            ctArrayShape = tuple(ctArray.shape)
            points3DDistanceMap, _, sliceId = getGaussianDistanceMapOld(ctArrayShape, points3D, distZ=DISTMAP_Z, sigma=DISTMAP_SIGMA, viewType=viewType)
        
        scribbleMapDataFgd = copy.deepcopy(scribbleMapData)
        scribbleMapDataFgd[scribbleMapDataFgd != VALUE_INT_FGD] = 0
        scribbleMapDataFgd[scribbleMapDataFgd == VALUE_INT_FGD] = 1
        scribbleMapDataBgd = copy.deepcopy(scribbleMapData)
        scribbleMapDataBgd[scribbleMapDataBgd != VALUE_INT_BGD] = 0
        scribbleMapDataBgd[scribbleMapDataBgd == VALUE_INT_BGD] = 1
        points3DDistanceMapFgd = getGaussianDistanceMap(scribbleMapDataFgd, distZ=DISTMAP_Z, sigma=DISTMAP_SIGMA, callerFunc='plotData.scribbleMapDataFgd')
        points3DDistanceMapBgd = getGaussianDistanceMap(scribbleMapDataBgd, distZ=DISTMAP_Z, sigma=DISTMAP_SIGMA, callerFunc='plotData.scribbleMapDataBgd')
                    
        # Step 1 - Set up figure
        rows         = 3
        baseColumns  = 3
        totalColumns = baseColumns
        extraSliceIdsAndColumnIds = []
        if points3D is not None:
            totalColumns += extraSlices # +3,-3 slices for each view
            for sliceDelta in range(-extraSlices//2+1, extraSlices//2+1):
                sliceNeighborId = sliceId + sliceDelta
                columnId        = baseColumns + extraSlices//2 + sliceDelta
                if sliceNeighborId >= 0 and sliceNeighborId < ctArray.shape[2]:
                    extraSliceIdsAndColumnIds.append((sliceNeighborId, columnId))
        if extraSlices > 0 or extraSlices is not None:
            f,axarr = plt.subplots(rows,totalColumns, figsize=(30, 8))
        else:
            f,axarr = plt.subplots(rows,totalColumns)
        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.05, hspace=0.05)
        
        # Step 2 - Show different views (Axial/Sagittal/Coronal)
        if 1:
            LINEWIDTHS = 0.25
            # Step 2.1 - Axial slice
            axarr[0,0].set_ylabel('Axial')
            axarr[0,0].imshow(ctArray[:, :, sliceId], cmap=COLORSTR_GRAY)
            axarr[0,1].imshow(ptArray[:, :, sliceId], cmap=COLORSTR_GRAY)
            axarr[0,2].imshow(ctArray[:, :, sliceId], cmap=COLORSTR_GRAY)
            if gtArray is not None:
                axarr[0,0].contour(gtArray[:, :, sliceId], colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                axarr[0,1].contour(gtArray[:, :, sliceId], colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
            if predArray is not None:
                axarr[0,0].contour(predArray[:, :, sliceId], colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                axarr[0,1].contour(predArray[:, :, sliceId], colors=COLORSTR_RED, linewidths=LINEWIDTHS)
            if refineArray is not None:
                axarr[0,0].contour(refineArray[:, :, sliceId], colors=COLORSTR_PINK, linestyle='dotted', linewidths=LINEWIDTHS)
                axarr[0,1].contour(refineArray[:, :, sliceId], colors=COLORSTR_PINK, linestyle='dotted', linewidths=LINEWIDTHS)
            for (sliceNeighborId, columnId) in extraSliceIdsAndColumnIds:
                axarr[0,columnId].imshow(ctArray[:, :, sliceNeighborId], cmap=COLORSTR_GRAY)
                axarr[0,columnId].imshow(ptArray[:, :, sliceNeighborId], cmap=COLORSTR_GRAY, alpha=0.3)
                if gtArray is not None:
                    axarr[0,columnId].contour(gtArray[:, :, sliceNeighborId], colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                if predArray is not None:
                    axarr[0,columnId].contour(predArray[:, :, sliceNeighborId], colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                if refineArray is not None:
                    axarr[0,columnId].contour(refineArray[:, :, sliceNeighborId], colors=COLORSTR_PINK, linestyle='dotted', linewidths=LINEWIDTHS)
                
                sliceDrawStr = ''
                if sliceNeighborId == sliceId:
                    sliceDrawStr = '**'
                axarr[0,columnId].set_title('Slice: {}{}'.format(sliceNeighborId+1, sliceDrawStr))
            
            # Step 2.2 - Sagittal slice
            axarr[1,0].set_ylabel('Sagittal')
            axarr[1,0].imshow(rotSagittal(ctArray[:, sliceId, :]), cmap=COLORSTR_GRAY)
            axarr[1,1].imshow(rotSagittal(ptArray[:, sliceId, :]), cmap=COLORSTR_GRAY)
            axarr[1,2].imshow(rotSagittal(ctArray[:, sliceId, :]), cmap=COLORSTR_GRAY)
            if gtArray is not None:
                axarr[1,0].contour(rotSagittal(gtArray[:, sliceId, :]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                axarr[1,1].contour(rotSagittal(gtArray[:, sliceId, :]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
            if predArray is not None:
                axarr[1,0].contour(rotSagittal(predArray[:, sliceId, :]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                axarr[1,1].contour(rotSagittal(predArray[:, sliceId, :]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
            if refineArray is not None:
                axarr[1,0].contour(rotSagittal(refineArray[:, sliceId, :]), colors=COLORSTR_PINK, linestyle='dashed', linewidths=LINEWIDTHS)
                axarr[1,1].contour(rotSagittal(refineArray[:, sliceId, :]), colors=COLORSTR_PINK, linestyle='dashed', linewidths=LINEWIDTHS)
            for (sliceNeighborId, columnId) in extraSliceIdsAndColumnIds:
                axarr[1,columnId].imshow(rotSagittal(ctArray[:, sliceNeighborId, :]), cmap=COLORSTR_GRAY)
                axarr[1,columnId].imshow(rotSagittal(ptArray[:, sliceNeighborId, :]), cmap=COLORSTR_GRAY, alpha=0.3)
                if gtArray is not None:
                    axarr[1,columnId].contour(rotSagittal(gtArray[:, sliceNeighborId, :]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                if predArray is not None:
                    axarr[1,columnId].contour(rotSagittal(predArray[:, sliceNeighborId, :]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                if refineArray is not None:
                    axarr[1,columnId].contour(rotSagittal(refineArray[:, sliceNeighborId, :]), colors=COLORSTR_PINK, linestyle='dotted', linewidths=LINEWIDTHS)

            # Step 2.3 - Coronal slice
            axarr[2,0].set_ylabel('Coronal')
            axarr[2,0].imshow(rotCoronal(ctArray[sliceId, :, :]), cmap=COLORSTR_GRAY)
            axarr[2,1].imshow(rotCoronal(ptArray[sliceId, :, :]), cmap=COLORSTR_GRAY)
            axarr[2,2].imshow(rotCoronal(ctArray[sliceId, :, :]), cmap=COLORSTR_GRAY)
            if gtArray is not None:
                axarr[2,0].contour(rotCoronal(gtArray[sliceId, :, :]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                axarr[2,1].contour(rotCoronal(gtArray[sliceId, :, :]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
            if predArray is not None:
                axarr[2,0].contour(rotCoronal(predArray[sliceId, :, :]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                axarr[2,1].contour(rotCoronal(predArray[sliceId, :, :]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
            if refineArray is not None:
                axarr[2,0].contour(rotCoronal(refineArray[sliceId, :, :]), colors=COLORSTR_PINK, linestyle='dashed', linewidths=LINEWIDTHS)
                axarr[2,1].contour(rotCoronal(refineArray[sliceId, :, :]), colors=COLORSTR_PINK, linestyle='dashed', linewidths=LINEWIDTHS)
            for (sliceNeighborId, columnId) in extraSliceIdsAndColumnIds:
                axarr[2,columnId].imshow(rotCoronal(ctArray[sliceNeighborId, :, :]), cmap=COLORSTR_GRAY)
                axarr[2,columnId].imshow(rotCoronal(ptArray[sliceNeighborId, :, :]), cmap=COLORSTR_GRAY, alpha=0.3)
                if gtArray is not None:
                    axarr[2,columnId].contour(rotCoronal(gtArray[sliceNeighborId, :, :]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                if predArray is not None:
                    axarr[2,columnId].contour(rotCoronal(predArray[sliceNeighborId, :, :]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                if refineArray is not None:
                    axarr[2,columnId].contour(rotCoronal(refineArray[sliceNeighborId, :, :]), colors=COLORSTR_PINK, linestyle='dotted', linewidths=LINEWIDTHS)
        
        # Step 3 - Show distance map
        if 1:
            if points3DDistanceMap is not None:
                
                # Step 3.1 - Get colormaps
                if scribbleType == KEY_SCRIBBLE_FGD:
                    scribbleColor = RGBA_ARRAY_YELLOW
                    scribbleColorStr = 'yellow'
                elif scribbleType == KEY_SCRIBBLE_BGD:
                    scribbleColor = RGBA_ARRAY_BLUE
                    scribbleColorStr = 'blue'
                scribbleColorMapBase = matplotlib.colors.ListedColormap([scribbleColor for _ in range(256)])
                scribbleColorMap, scribbleNorm     = getScribbleColorMap(scribbleColorMapBase, opacityBoolForScribblePoints=True)
                cmapScribbleDist, normScribbleDist = getScribbleColorMap(CMAP_DEFAULT, opacityBoolForScribblePoints=False)

                scribbleColorMapBaseFgd              = matplotlib.colors.ListedColormap([RGBA_ARRAY_YELLOW for _ in range(256)])
                scribbleColorMapFgd, scribbleNormFgd = getScribbleColorMap(scribbleColorMapBaseFgd, opacityBoolForScribblePoints=True)
                scribbleColorMapBaseBgd              = matplotlib.colors.ListedColormap([RGBA_ARRAY_BLUE for _ in range(256)])
                scribbleColorMapBgd, scribbleNormBgd = getScribbleColorMap(scribbleColorMapBaseBgd, opacityBoolForScribblePoints=True)

                # Step 3.2 - Get binary distance map
                points3DDistanceMapBinary = copy.deepcopy(points3DDistanceMap)
                points3DDistanceMapBinary[points3DDistanceMapBinary < 1] = 0

                if viewType == KEY_AXIAL:
                    axial2DSlice = skimage.morphology.binary_dilation(points3DDistanceMapBinary[:, :, sliceId])
                    axarr[0,0].imshow(axial2DSlice, cmap=scribbleColorMap, norm=scribbleNorm)
                    axarr[0,1].imshow(axial2DSlice, cmap=scribbleColorMap, norm=scribbleNorm)
                    axarr[0,2].imshow(skimage.morphology.binary_dilation(scribbleMapDataFgd[:,:,sliceId]), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                    axarr[0,2].imshow(skimage.morphology.binary_dilation(scribbleMapDataBgd[:,:,sliceId]), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                    for (sliceNeighborId, columnId) in extraSliceIdsAndColumnIds:
                        # axarr[0,columnId].imshow(points3DDistanceMap[:, :, sliceNeighborId], cmap=cmapScribbleDist, norm=normScribbleDist)
                        if scribbleType == KEY_SCRIBBLE_FGD  : axarr[0,columnId].imshow(points3DDistanceMapFgd[:, :, sliceNeighborId], cmap=cmapScribbleDist, norm=normScribbleDist)
                        elif scribbleType == KEY_SCRIBBLE_BGD: axarr[0,columnId].imshow(points3DDistanceMapBgd[:, :, sliceNeighborId], cmap=cmapScribbleDist, norm=normScribbleDist)
                elif viewType == KEY_SAGITTAL:
                    sagittal2DSlice = skimage.morphology.binary_dilation(rotSagittal(points3DDistanceMapBinary[:, sliceId, :]))
                    axarr[1,0].imshow(sagittal2DSlice, cmap=scribbleColorMap, norm=scribbleNorm)
                    axarr[1,1].imshow(sagittal2DSlice, cmap=scribbleColorMap, norm=scribbleNorm)
                    axarr[1,2].imshow(skimage.morphology.binary_dilation(rotSagittal(scribbleMapDataFgd[:,sliceId,:])), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                    axarr[1,2].imshow(skimage.morphology.binary_dilation(rotSagittal(scribbleMapDataBgd[:,sliceId,:])), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                    for (sliceNeighborId, columnId) in extraSliceIdsAndColumnIds:
                        # axarr[1,columnId].imshow(rotSagittal(points3DDistanceMap[:, sliceNeighborId, :]), cmap=cmapScribbleDist, norm=normScribbleDist)
                        if scribbleType == KEY_SCRIBBLE_FGD  : axarr[1,columnId].imshow(rotSagittal(points3DDistanceMapFgd[:, sliceNeighborId, :]), cmap=cmapScribbleDist, norm=normScribbleDist)
                        elif scribbleType == KEY_SCRIBBLE_BGD: axarr[1,columnId].imshow(rotSagittal(points3DDistanceMapBgd[:, sliceNeighborId, :]), cmap=cmapScribbleDist, norm=normScribbleDist)
                elif viewType == KEY_CORONAL:
                    coronal2DSlice = skimage.morphology.binary_dilation(rotCoronal(points3DDistanceMapBinary[sliceId, :, :]))
                    axarr[2,0].imshow(coronal2DSlice, cmap=scribbleColorMap, norm=scribbleNorm)
                    axarr[2,1].imshow(coronal2DSlice, cmap=scribbleColorMap, norm=scribbleNorm)
                    axarr[2,2].imshow(skimage.morphology.binary_dilation(rotCoronal(scribbleMapDataFgd[sliceId,:,:])), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                    axarr[2,2].imshow(skimage.morphology.binary_dilation(rotCoronal(scribbleMapDataBgd[sliceId,:,:])), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                    for (sliceNeighborId, columnId) in extraSliceIdsAndColumnIds:
                        # axarr[2,columnId].imshow(rotCoronal(points3DDistanceMap[sliceNeighborId, :, :]), cmap=cmapScribbleDist, norm=normScribbleDist)
                        if scribbleType == KEY_SCRIBBLE_FGD  : axarr[2,columnId].imshow(rotCoronal(points3DDistanceMapFgd[sliceNeighborId, :, :]), cmap=cmapScribbleDist, norm=normScribbleDist)
                        elif scribbleType == KEY_SCRIBBLE_BGD: axarr[2,columnId].imshow(rotCoronal(points3DDistanceMapBgd[sliceNeighborId, :, :]), cmap=cmapScribbleDist, norm=normScribbleDist)
                else:
                    print (' - [plotData()] Unknown viewType: {}'.format(viewType))
        
        supTitleStr = 'CaseName: {} | SliceIdx: {} | SlideID: (per GUI): {}'.format(caseName, sliceId, sliceId+1)
        if points3D is not None:
            supTitleStr += '\n ( scribbleType: {} in view: {})'.format(scribbleType, viewType) 
        supTitleStr += r'\n(\\textcolor{GT}{green}, \\textcolor{Prev Pred}{red}, \textcolor{Refined Pred}{pink}, \textcolor{distance-hmap}{orange}'
        plt.suptitle(supTitleStr) #, usetex=True)
        
        # if saveFolderPath is None:
        #     plt.show()
        if saveFolderPath is not None:
            Path(saveFolderPath).mkdir(parents=True, exist_ok=True)
            pngName     = fileNameForSave(caseName, counter, str(viewType), int(sliceId))
            saveFigPath = Path(saveFolderPath).joinpath('{}.png'.format(pngName))
            plt.savefig(str(saveFigPath), bbox_inches='tight', dpi=SAVE_DPI)
            plt.close()
        

        # For paper
        if 0:
            try:
                if viewType == KEY_AXIAL: 
                    indexing = (slice(None), slice(None), sliceId)
                    rotFunc  = rotAxial
                elif viewType == KEY_SAGITTAL: 
                    indexing = (slice(None), sliceId, slice(None))
                    rotFunc  = rotSagittal
                elif viewType == KEY_CORONAL: 
                    indexing = (sliceId, slice(None), slice(None))
                    rotFunc  = rotCoronal
                
                forPaperDPI = 500

                plt.imshow(rotFunc(ctArray[indexing]), cmap=COLORSTR_GRAY)
                plt.imshow(rotFunc(ptArray[indexing]), cmap=COLORSTR_GRAY, alpha=0.6)
                plt.contour(rotFunc(gtArray[indexing]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                plt.contour(rotFunc(predArray[indexing]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataFgd[indexing])), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataBgd[indexing])), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_0_CT_PET_GT_Pred_Scribbles.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                plt.close()

                # plt.imshow(rotFunc(ctArray[indexing]), cmap=COLORSTR_GRAY)
                # plt.contour(rotFunc(gtArray[indexing]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                # plt.contour(rotFunc(predArray[indexing]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                # plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataFgd[indexing])), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                # plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataBgd[indexing])), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                # plt.xticks([]); plt.yticks([])
                # saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_CT_Pred_Scribbles.png'.format(pngName))
                # plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                # plt.close()

                # plt.imshow(rotFunc(ptArray[indexing]), cmap=COLORSTR_GRAY)
                # plt.contour(rotFunc(gtArray[indexing]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                # plt.contour(rotFunc(predArray[indexing]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                # plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataFgd[indexing])), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                # plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataBgd[indexing])), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                # plt.xticks([]); plt.yticks([])
                # saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_PT_Pred_Scribbles.png'.format(pngName))
                # plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                # plt.close()

                plt.imshow(rotFunc(ctArray[indexing]), cmap=COLORSTR_GRAY)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_1_CT_only.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                plt.close()

                plt.imshow(rotFunc(ptArray[indexing]), cmap=COLORSTR_GRAY)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_2_PET_only.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                plt.close()

                plt.imshow(np.zeros_like(ptArray[indexing]), cmap=COLORSTR_GRAY)
                plt.contour(rotFunc(predArray[indexing]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_3_Pred_only.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                plt.close()

                plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataFgd[indexing])), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_4_FGD_only.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)

                plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataBgd[indexing])), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_5_BGD_only.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)

                plt.imshow(np.zeros_like(ptArray[indexing]), cmap=COLORSTR_GRAY)
                plt.contour(rotFunc(refineArray[indexing]), colors=COLORSTR_PINK, linewidths=LINEWIDTHS)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_6_Refine_only.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                plt.close()
                
                try:
                    # scribbleColorMapBase = matplotlib.colors.ListedColormap([RGBA_ARRAY_YELLOW for _ in range(256)])
                    # scribbleColorMap, scribbleNorm     = getScribbleColorMap(scribbleColorMapBase, opacityBoolForScribblePoints=True)
                    cmapScribbleDist, normScribbleDist = getScribbleColorMap(CMAP_DEFAULT, opacityBoolForScribblePoints=False)
                    plt.imshow(rotFunc(points3DDistanceMapFgd[indexing]), cmap=cmapScribbleDist, norm=normScribbleDist)
                    plt.xticks([]); plt.yticks([])
                    saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_7_FGD_distanceMap.png'.format(pngName))
                    plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)

                    # scribbleColorMapBase = matplotlib.colors.ListedColormap([RGBA_ARRAY_BLUE for _ in range(256)])
                    # scribbleColorMap, scribbleNorm     = getScribbleColorMap(scribbleColorMapBase, opacityBoolForScribblePoints=True)
                    cmapScribbleDist, normScribbleDist = getScribbleColorMap(CMAP_DEFAULT, opacityBoolForScribblePoints=False)
                    plt.imshow(rotFunc(points3DDistanceMapBgd[indexing]), cmap=cmapScribbleDist, norm=normScribbleDist)
                    plt.xticks([]); plt.yticks([])
                    saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_8_BGD_distanceMap.png'.format(pngName))
                    plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                except:
                    traceback.print_exc()

                plt.imshow(rotFunc(ctArray[indexing]), cmap=COLORSTR_GRAY)
                plt.imshow(rotFunc(ptArray[indexing]), cmap=COLORSTR_GRAY, alpha=0.6)
                plt.contour(rotFunc(gtArray[indexing]), colors=COLORSTR_GREEN, linewidths=LINEWIDTHS)
                plt.contour(rotFunc(predArray[indexing]), colors=COLORSTR_RED, linewidths=LINEWIDTHS)
                plt.contour(rotFunc(refineArray[indexing]), colors=COLORSTR_PINK, linestyle='dotted', linewidths=LINEWIDTHS)
                plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataFgd[indexing])), cmap=scribbleColorMapFgd, norm=scribbleNormFgd)
                plt.imshow(rotFunc(skimage.morphology.binary_dilation(scribbleMapDataBgd[indexing])), cmap=scribbleColorMapBgd, norm=scribbleNormBgd)
                plt.xticks([]); plt.yticks([])
                saveFigPathForPaper = Path(saveFolderPath).joinpath('{}_paper_input_9_CT_PET_GT_Pred_Scribbles_rEFINE.png'.format(pngName))
                plt.savefig(str(saveFigPathForPaper), bbox_inches='tight', dpi=forPaperDPI)
                plt.close()



            except:
                traceback.print_exc()

    except:
        plt.close()
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()

    return points3DDistanceMap, viewType, sliceId, pngName

def plot2DInteractionAsRGB(points3DDistanceMap, viewType, sliceId, pngName, counter, ctArray, segArrayGT, segArrayPred, refineArray=None, saveFolderPath=None):
    """
    In this function, we plot the interaction between the prediction, GT, and scribble in RGB format
    Here R=Pred, G=GT, B=Interaction
    """
    try:
        print (' - [plot2DInteractionAsRGB()] viewType: {} | sliceId: {}'.format(viewType, sliceId))
        if points3DDistanceMap is not None and viewType is not None and sliceId is not None and pngName is not None:
            
            # Step 0 - Init
            rotAxial    = lambda x: x
            rotSagittal = lambda x: np.rot90(x, k=1)
            rotCoronal  = lambda x: np.rot90(x, k=1)

            # Step 1 - Create image placeholder
            image = np.zeros((ctArray.shape[0], ctArray.shape[1], 3))

            # Step 2 - Chose prediction map on basis of counter
            if counter == 1:
                segArrayPredThis = segArrayPred
            else:
                segArrayPredThis = refineArray
            
            # Step 3 - Create binary distance map (from gaussian)
            points3DDistanceMapBinary = copy.deepcopy(points3DDistanceMap)
            points3DDistanceMapBinary[points3DDistanceMapBinary < 1] = 0

            # Step 4 - Add RGB channels at pred, GT, and scribble
            if viewType == KEY_AXIAL:
                image[:,:,0] = segArrayPred[:,:,sliceId] # R
                image[:,:,1] = segArrayGT[:,:,sliceId]   # G
                image[:,:,2] = points3DDistanceMapBinary[:,:,sliceId] # B
            elif viewType == KEY_CORONAL:
                image[:,:,0] = rotSagittal(segArrayPred[sliceId, :, :])
                image[:,:,1] = rotSagittal(segArrayGT[sliceId, :, :])
                image[:,:,2] = rotSagittal(points3DDistanceMapBinary[sliceId, :, :])
            elif viewType == KEY_SAGITTAL:
                image[:,:,0] = rotCoronal(segArrayPred[:, sliceId, :])
                image[:,:,1] = rotCoronal(segArrayGT[:, sliceId, :])
                image[:,:,2] = rotCoronal(points3DDistanceMapBinary[:,sliceId, :])

            # Step 5 - Save image
            imageInt = (image*255).astype(np.uint8)
            pngNameForInteraction = Path(saveFolderPath, '{}-interaction.png'.format(pngName))
            imageio.imwrite(pngNameForInteraction, imageInt)

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()

def plot(scribbleMapData, preparedDataTorch, segArrayGT, caseName, counter, points3D, viewType, scribbleType, refineArray=None, saveFolderPath=None):
    """
    Params
    ------
    scribbleMapData: np.ndarray, [H,W,D], containing either VALUE_INT_FGD or VALUE_INT_BGD
    preparedDataTorch: torch.Tensor, shape: (batch_size, 3, height, width)

    """
    try:
        ctArray      = np.array(preparedDataTorch[0,0])
        ptArray      = np.array(preparedDataTorch[0,1])
        segArrayPred = np.array(preparedDataTorch[0,2])

        # PLotting data    
        if 1:
            points3DDistanceMap, _, sliceId, pngName = plotData(scribbleMapData, ctArray, ptArray, segArrayGT, segArrayPred, refineArray, None, caseName, counter, points3D, viewType, scribbleType, saveFolderPath=saveFolderPath)

        # Saving interactions in RGB format (R=Pred, G=GT, B=Interaction)
        if 1:
            plot2DInteractionAsRGB(points3DDistanceMap, viewType, sliceId, pngName, counter, ctArray, segArrayGT, segArrayPred, refineArray, saveFolderPath)
            

    except:
        traceback.print_exc()
        if MODE_DEBUG: pdb.set_trace()

def plotUsingThread(plotFunc, *args):

    thread = threading.Thread(target=run_executor_in_thread, args=(plotFunc, *args))
    thread.daemon = True  # This makes the thread a daemon thread
    thread.start()

def run_executor_in_thread(func, *args):
    with ProcessPoolExecutor() as executor:
        future= executor.submit(func, *args)

#################################################################
#                        API ENDPOINTS
#################################################################

# Step 2 - Global Vars-related
SESSIONSGLOBAL = {}
DCMCLIENT      = None
MODEL          = None
DEVICE         = None
ORT_SESSION    = None
LOAD_ONNX      = False

## -------------------------------------------------->>> Entry point
@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):

    ################################################################## Step 0 - Init
    tracemalloc.start()
    import socket
    hostname   = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print ('\n =========================== [lifespan()] =========================== \n')
    print(f" - Server ({hostname}) is running on IP: {ip_address} (USE_HTTPS: {USE_HTTPS})")
    print ('   --> SESSIONSGLOBAL: ', SESSIONSGLOBAL.keys())
    # checkAssetPaths()
    print ('\n =========================== [lifespan()] =========================== \n')
    

    ################################################################## Step 1 - On startup
    global DEVICE
    global MODEL
    global ORT_SESSION
    global LOAD_ONNX
    DEVICE = getTorchDevice()
    
    ######################## Experiment-wise settings ########################
    if 0:
        expName   = 'UNetv1__DICE-LR1e3__Class1__Trial1'
        epoch     = 100
        modelType = KEY_UNET_V1 # type == <class 'monai.networks.nets.unet.UNet'>
        # DEVICE    = torch.device('cpu')
        loadOnnx  = True
    
    elif 0:
        expName   = 'UNetv1__DICE-LR1e3-B12__Cls1-Pt-Scr__Trial1'
        epoch     = 100
        modelType = KEY_UNET_V1 # type == <class 'monai.networks.nets.unet.UNet'>
        # DEVICE    = torch.device('cpu')
        loadOnnx  = True
    
    elif 0:
        expName   = 'UNetv1__DICE-LR1e3-B12__Cls1-Pt-Scr__Trial2'
        epoch     = 150
        modelType = KEY_UNET_V1 # type == <class 'monai.networks.nets.unet.UNet'>
        # DEVICE    = torch.device('cpu')
        loadOnnx  = False # True, False
    
    elif 0:
        expName   = 'UNetv1__DICE-LR1e3__W5-B32__Cls1-Pt-Scr__Trial5'
        epoch     = 500
        modelType = KEY_UNET_V1 # type == <class 'monai.networks.nets.unet.UNet'>
        # DEVICE    = torch.device('cpu')
        loadOnnx  = True # True, False
    
    # 2024-10-02
    elif 1:
        expName   = 'UNetv1__DICE-LR1e3__W5-B32-MoreData__Cls1-Pt-Scr__Trial1'
        epoch     = 90 # [80,90(best),190]
        modelType = KEY_UNET_V1 # type == <class 'monai.networks.nets.unet.UNet'>
        # DEVICE    = torch.device('cpu')
        loadOnnx  = False # True, False
    
    # 2024-10-14 (Obedience Loss)
    elif 0:
        expName = 'UNetv1__1DICEep1-1Obedep2-LR1e3__W3-B32-MoreData__Cls1-Pt-Scr__Trial2_ep50'
        epoch     = 100
        modelType = KEY_UNET_V1
        loadOnnx  = True

    MODEL, ORT_SESSION = loadModelUsingUserPath(DEVICE, expName, epoch, modelType, loadOnnx)
    LOAD_ONNX = loadOnnx

    
    yield
    
    ################################################################## Step 1 - On startup
    print (' - [on_shutdown()] Nothing here!')

# Step 1 - App related
app     = fastapi.FastAPI(lifespan=lifespan, title="FastAPI: Interactive Server Python App")
configureFastAPIApp(app)
setproctitle.setproctitle("interactive-server.py") # set process name
logger = logging.getLogger(__name__)
loggerFileHandler = logging.FileHandler(DIR_LOGS / 'interactive-server-{:%Y-%m-%d-%H-%M-%S}.log'.format(datetime.datetime.now()), encoding='utf-8')
logger.addHandler(loggerFileHandler)
app.add_middleware(StripLeadingSlashMiddleware)

def checkAssetPaths(verbose=False):

    logBool, certBool, keyBool = False, False, False
    fail = False

    try:    

        # LogConfig
        if Path(PATH_LOGCONFIG).exists():
            if verbose: print (' - [checkPath()] logConfig file exists!')
            logBool = True
        else:
            print (' - [checkPath()] logConfig file does not exist!: ', PATH_LOGCONFIG)
        
        # Keys
        if Path(PATH_HOSTCERT).exists():
            if verbose: print (' - [checkPath()] hostCert file exists!')
            certBool = True
        else:
            fail = True
            print (' - [checkPath()] hostCert file does not exist!: ', PATH_HOSTCERT)
        
        if Path(PATH_HOSTKEY).exists():
            if verbose: print (' - [checkPath()] hostKey file exists!')
            keyBool = True
        else:
            fail = True
            print (' - [checkPath()] hostKey file does not exist!: ', PATH_HOSTKEY)
        
        if fail:
            print (' - [checkPath()] Please check the below tree structure!')
            print (directory_tree.DisplayTree('.'))
            print ('--'*20)

    except:
        traceback.print_exc()
    
    return logBool and certBool and keyBool

@app.middleware("http")
async def logging_middleware(request: fastapi.Request, call_next: typing.Callable[[fastapi.Request], typing.Awaitable[fastapi.Response]]) -> fastapi.Response:
    
    # Production mode
    if 1:
        start    = time.time()
        response = await call_next(request)
        duration = (time.time() - start)
        
        if request.url.path != '/serverHealth':
        # if 1:
            source   = termcolor.colored(f"{request.client.host}:{request.client.port}", "blue")
            # source   = (request.headers.get('origin', None))
            resource = termcolor.colored(f"{request.method} {request.url.path}", "green")
            result   = termcolor.colored(f"{response.status_code}", "yellow")
            duration = termcolor.colored(f"[{duration:.1f}s]", "magenta")
            message  = f"{source} => {resource} => {result} {duration}"
            logger.info(message)
    
    # Development mode (to see all requests)
    else:
        source   = termcolor.colored(f"{request.client.host}:{request.client.port}", "blue")
        resource = termcolor.colored(f"{request.method} {request.url.path}", "green")
        message  = f"Incoming request: {source} => {resource}"
        logger.info(message)

        start    = time.time()
        response = await call_next(request)
        duration = (time.time() - start)
        
        # Log the response
        result   = termcolor.colored(f"{response.status_code} ({http.HTTPStatus(response.status_code)})", "yellow")
        duration = termcolor.colored(f"[{duration:.1f}s]", "magenta")
        message  = f"{source} => {resource} => {result} {duration}"
        logger.info(message)
        
    return response
    
# Step 3 - API Endpoints
@app.post("/prepare")
async def prepare(payload: PayloadPrepare, request: starlette.requests.Request):
    
    global DCMCLIENT
    global SESSIONSGLOBAL

    try:

        # Step 0 - Init
        print ('|----------------------------------------------')
        getMemoryUsage()
        tStart             = time.time()
        userAgent, referer = getRequestInfo(request)
        clientIdentifier   = getClientIdentifier(payload.identifier, payload.user) # payload.identifier + '__' + clientUserName
        preparePayloadData = payload.data.dict()
        patientName        = preparePayloadData[KEY_CASE_NAME]
        # user         = request.user # AuthenticationMiddleware must be installed to access request.user

        if clientIdentifier not in SESSIONSGLOBAL:
            SESSIONSGLOBAL[clientIdentifier] = {'userAgent': userAgent, KEY_CLIENT_IDENTIFIER: clientIdentifier}
        
        if patientName not in SESSIONSGLOBAL[clientIdentifier]:
            SESSIONSGLOBAL[clientIdentifier][patientName] = {KEY_DATA:{}, KEY_TORCH_DATA: [], KEY_SCRIBBLE_MAP: []
                                                , KEY_DCM_LIST: [], KEY_SCRIBBLE_COUNTER: 0, KEY_MANUAL_COUNTER: 0
                                                , KEY_SEG_SOP_INSTANCE_UID: None, KEY_SEG_SERIES_INSTANCE_UID: None
                                                , KEY_SEG_ORTHANC_ID: None
                                                , KEY_DATETIME: datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                                                , KEY_SEG_ARRAY_GT: None
                                                , KEY_PATIENT_NAME: patientName
                                                , KEY_SLICE_IDXS_SCROLLED_OBJ: {KEY_AXIAL: {}, KEY_SAGITTAL: {}, KEY_CORONAL: {}}
                                                }
            if KEY_JOHN in payload.user and KEY_DOE in payload.user:
                print (' - [prepare()] JohnDoe user detected!')
                SESSIONSGLOBAL[clientIdentifier][patientName][KEY_PATH_SAVE] = Path(DIR_EXPERIMENTS_JOHNDOE).joinpath(SESSIONSGLOBAL[clientIdentifier][patientName][KEY_DATETIME] + ' -- ' + clientIdentifier)
            else:      
                SESSIONSGLOBAL[clientIdentifier][patientName][KEY_PATH_SAVE] = Path(DIR_EXPERIMENTS).joinpath(SESSIONSGLOBAL[clientIdentifier][patientName][KEY_DATETIME] + ' -- ' + clientIdentifier)
            patientSeriesInstanceUID, patientSOPInstanceUID = getPatientUUIDs(patientName)
            SESSIONSGLOBAL[clientIdentifier][patientName][KEY_SEG_SERIES_INSTANCE_UID] = patientSeriesInstanceUID
            SESSIONSGLOBAL[clientIdentifier][patientName][KEY_SEG_SOP_INSTANCE_UID]    = patientSOPInstanceUID

        # Step 1 - Check if new scans are selected on the client side
        dataAlreadyPresent = True
        patientData = SESSIONSGLOBAL[clientIdentifier][patientName]
        if patientData[KEY_DATA] != preparePayloadData:
            dataAlreadyPresent = False
            patientData[KEY_DATA] = preparePayloadData
            patientData[KEY_TORCH_DATA] = []

            if DCMCLIENT == None:
                DCMCLIENT = getDCMClient(preparePayloadData[KEY_SEARCH_OBJ_CT][KEY_WADO_RS_ROOT])
                
            if DCMCLIENT != None:
                ctArrayProcessedBool, ctArray, ctArrayProcessed, patientData = getCTArray(DCMCLIENT, patientData)
                if ctArrayProcessedBool:
                    ptArray, ptArrayProcessed, patientData = getPTArray(DCMCLIENT, patientData)
                    if ptArray is not None:
                        segArrayGT, segArrayPred, patientData = getSEGs(DCMCLIENT, patientData)
                        if segArrayPred is not None:
                            if ctArray.shape == ptArray.shape == segArrayPred.shape:                               
                                if 0:
                                    plotHistograms(ctArray, ctArrayProcessed, ptArray, ptArrayProcessed, segArrayGT, segArrayPred, patientName, patientData[KEY_PATH_SAVE])
                                    plotUsingThread(plotHistograms, ctArray, ctArrayProcessed, ptArray, ptArrayProcessed, segArrayGT, segArrayPred, patientName, patientData[KEY_PATH_SAVE])
                                if 0:
                                    saveFolderPath = patientData[KEY_PATH_SAVE]
                                    # TODO: plotData() has changed a lot since this was written. Need to update this
                                    plotData(ctArray, ptArray, segArrayGT, segArrayPred, sliceId=95, caseName=patientName, saveFolderPath=saveFolderPath)
                                    plotData(ctArray, ptArray, segArrayGT, segArrayPred, sliceId=82, caseName=patientName, saveFolderPath=saveFolderPath)
                                    plotData(ctArray, ptArray, segArrayGT, segArrayPred, sliceId=62, caseName=patientName, saveFolderPath=saveFolderPath)
                                
                            else:
                                raise fastapi.HTTPException(status_code=500, detail="shapes dont match for patientName: {}".format(patientName))
                        else:
                            raise fastapi.HTTPException(status_code=500, detail="getSEGs() failed for patientName: {}".format(patientName))
                    else:
                        raise fastapi.HTTPException(status_code=500, detail="getPTArray() failed for patientName: {}".format(patientName))
                else:
                    raise fastapi.HTTPException(status_code=500, detail="getCTArray() failed for patientName: {}".format(patientName))
                
                SESSIONSGLOBAL[clientIdentifier][patientName] = patientData

        else:
            dataAlreadyPresent = True
        
        # Step 2 - Logging        
        print (' - /prepare (for {}) (dataAlreadyPresent:{}): {}'.format(clientIdentifier, dataAlreadyPresent, patientName))

        # Step 99 - Return
        getMemoryUsage()
        print ('|----------------------------------------------')
        tTotal = time.time() - tStart
        if dataAlreadyPresent:
            return {"status": "[clientIdentifier={}, patientName={}] Data already loaded into python server ({:.2f}s)".format(clientIdentifier, patientName, tTotal)}
        else:
            return {"status": "[clientIdentifier={}, patientName={}] Fresh data loaded into python server ({:.2f}s)".format(clientIdentifier, patientName, tTotal)}
        
    except pydantic.ValidationError as e:
        print (' - /prepare (from {},{}): {}'.format(referer, userAgent, e))
        logging.error(e)
        raise fastapi.HTTPException(status_code=500, detail="Error in /prepare for patientName: {} => {}".format(patientName, e))
    
    except Exception as e:
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail="Error in /prepare for patientName: {} => {}".format(patientName, e))

@app.post("/process")
async def process(payload: PayloadProcess, request: starlette.requests.Request):

    global MODEL
    global ORT_SESSION
    global DCMCLIENT
    global SESSIONSGLOBAL
    global LOAD_ONNX

    # import memray
    # with memray.Tracker(DIR_LOGS / "memray-{}.bin".format(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))):
    if 1:
        try:
            # Step 0 - Init
            print ('|----------------------------------------------')
            getMemoryUsage()
            tStart = time.time()
            userAgent, referer = getRequestInfo(request)
            clientUserName     = payload.user
            clientIdentifier   = getClientIdentifier(payload.identifier, payload.user) # payload.identifier + '__' + clientUserName
            processPayloadData = payload.data.dict()
            patientName        = processPayloadData[KEY_CASE_NAME]
            returnMessagePrefix = MSG_RETURN_PREFIX.format(clientIdentifier, patientName, LOAD_ONNX)

            # Step 1 - Check if session data is available
            dataAlreadyPresent = False
            if clientIdentifier not in SESSIONSGLOBAL:
                dataAlreadyPresent = False
            elif patientName not in SESSIONSGLOBAL[clientIdentifier]:
                dataAlreadyPresent = False
            else:
                dataAlreadyPresent = True
        
            # Step 2 - Logging
            print (' - /process (for {}): {} with MODEL: '.format(clientIdentifier, patientName, MODEL))
            

            # Step 3 - Process scribble data
            if dataAlreadyPresent:
                
                # Step 3.0 - Extract data
                points3D     = processPayloadData[KEY_POINTS_3D] # [(h/w, h/w, d), (), ..., ()] [NOTE: cornerstone3D sends array-indexed data, so now +1/-1 needed]
                if len(points3D) == 0:
                    raise fastapi.HTTPException(status_code=500, detail="{} No scribble points selected. Please select scribble points.".format(returnMessagePrefix))
                points3D       = np.array([list(x) for x in points3D])
                viewType       = processPayloadData[KEY_VIEW_TYPE]
                scribbleType   = processPayloadData[KEY_SCRIBBLE_TYPE]
                timeToScribble     = processPayloadData[KEY_TIME_TO_SCRIBBLE]
                scribbleStartEpoch = processPayloadData[KEY_SCRIBBLE_START_EPOCH]

                # Step 3.0 - Init
                patientData       = SESSIONSGLOBAL[clientIdentifier][patientName]
                preparedData      = patientData[KEY_DATA]
                preparedDataTorch = patientData[KEY_TORCH_DATA]
                preparedDataTorch[0,3] = torch.zeros_like(preparedDataTorch[0,1])
                preparedDataTorch[0,4] = torch.zeros_like(preparedDataTorch[0,1])
                scribbleMapData   = patientData[KEY_SCRIBBLE_MAP]
                
                # Step 3.2 - Get distance map
                tDistMapStart = time.time()
                newPoints3D, scribbleMapData, preparedDataTorch, viewType, sliceId = getDistanceMap(scribbleMapData, preparedDataTorch, scribbleType, points3D, DISTMAP_Z, DISTMAP_SIGMA, viewType)
                if viewType is None or sliceId is None:
                    print ('|----------------------------------------------')
                    raise fastapi.HTTPException(status_code=500, detail="Error in /process => getDistanceMap() failed")
                # print (' - [process()] torch.sum(preparedDataTorch, dim=(2,3,4)): ', torch.sum(preparedDataTorch, dim=(2,3,4)))
                tDistMap = time.time() - tDistMapStart

                # Step 4.1 - Get refined segmentation
                tInferStart  = time.time()
                segArrayRefinedTorch, segArrayRefinedNumpy = doInferenceNew(MODEL, ORT_SESSION, preparedDataTorch)
                try:
                    segArrayRefinedNumpy = fillHolesIn3DBinaryMask(segArrayRefinedNumpy)
                    segArrayRefinedNumpy = fillHolesIn2DBinaryMaskByView(segArrayRefinedNumpy, viewType)

                    pass
                except:
                    traceback.print_exc()
                tInfer      = time.time() - tInferStart
                if segArrayRefinedNumpy is None or segArrayRefinedTorch is None:
                    print ('|----------------------------------------------')
                    raise fastapi.HTTPException(status_code=500, detail="Error in /process => doInferenceNew() failed")
                
                # Step 4.2 - Update counter for patient
                patientData[KEY_SCRIBBLE_COUNTER] += 1
                
                # Step 4.2 - Save refined segmentation
                tMakeSegDCMStart = time.time()
                makeSEGDICOMStatus, patientData = makeSEGDicom(segArrayRefinedNumpy, patientData, viewType, sliceId, scribbleStartEpoch, timeToScribble, tDistMap, tInfer)
                tMakeSegDCM = time.time() - tMakeSegDCMStart

                # Step 4.99 - Plot refined segmentation
                if 1:
                    try: 
                        segArrayGT     = patientData[KEY_SEG_ARRAY_GT]
                        preparedDataNp = copy.deepcopy(to_numpy(preparedDataTorch))
                        thread = threading.Thread(target=run_executor_in_thread, args=(plot, scribbleMapData, preparedDataNp, segArrayGT, patientName, patientData[KEY_SCRIBBLE_COUNTER], newPoints3D, viewType, scribbleType, segArrayRefinedNumpy, patientData[KEY_PATH_SAVE]))
                        thread.daemon = True  # This makes the thread a daemon thread
                        thread.start()
                    except:
                        traceback.print_exc()
                        if MODE_DEBUG: pdb.set_trace()
                
                # Step 5 - Update global data
                preparedDataTorch[0,2]        = segArrayRefinedTorch # update prediction in preparedDataTorch
                patientData[KEY_TORCH_DATA]   = preparedDataTorch
                patientData[KEY_SCRIBBLE_MAP] = scribbleMapData
                SESSIONSGLOBAL[clientIdentifier][patientName] = patientData

                if not makeSEGDICOMStatus:
                    raise fastapi.HTTPException(status_code=500, detail="Error in /process => makeSEGDicom failed")
                
                # Step 5 - Return
                tTotal = time.time() - tStart
                timeTakenStr = "[{}] Time taken: (tScribble={:.2f}s, tDistMap={:.2f}s, tInfer={:.2f}s, tMakeSegDCM={:.2f}s, tTotal={:.2f}s)".format(
                    patientName, timeToScribble, tDistMap, tInfer, tMakeSegDCM, tTotal)
                # print ('  - Time taken: ', timeTakenStr)
                logger.info(timeTakenStr)
                returnObj = {"status": "{} Scribble processed in python server {}".format(returnMessagePrefix, timeTakenStr)}
                returnObj[KEY_RESPONSE_DATA] = {
                    KEY_STUDY_INSTANCE_UID : preparedData[KEY_SEARCH_OBJ_CT][KEY_STUDY_INSTANCE_UID],
                    KEY_SERIES_INSTANCE_UID: patientData[KEY_SEG_SERIES_INSTANCE_UID],
                    KEY_SOP_INSTANCE_UID   : patientData[KEY_SEG_SOP_INSTANCE_UID],
                    KEY_WADO_RS_ROOT       : preparedData[KEY_SEARCH_OBJ_CT][KEY_WADO_RS_ROOT]
                }
                getMemoryUsage()
                print ('|----------------------------------------------')
                return returnObj
            
            else:
                print ('|----------------------------------------------')
                raise fastapi.HTTPException(status_code=500, detail="{} No data present in python server. Reload page.".format(returnMessagePrefix))
        
        except pydantic.ValidationError as e:
            print (' - /process (from {},{}): {}'.format(referer, userAgent, e))
            print ('|----------------------------------------------')
            logging.error(e)
            raise fastapi.HTTPException(status_code=500, detail="{} Error in /process => {}".format(returnMessagePrefix, str(e)))

        except Exception as e:
            traceback.print_exc()
            print ('|----------------------------------------------')
            raise fastapi.HTTPException(status_code=500, detail=" {} Error in /process => {}".format(returnMessagePrefix, str(e)))

@app.get('/')
async def root():

    try:
        dateStr = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        return {"message": "Hello World. This is Mody's AI interactive server! Valid POST endpoints are /prepare and /process. Time={}".format(dateStr)}
    except Exception as e:
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=" Error in / => {}".format(str(e)))

@app.get('/serverHealth')
async def serverHealth():
    
    try:
        dateStr = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        return {"message": getMemoryUsage(printFlag=False, returnFlag=True, updatePrevFlag=False), "time": dateStr}
    except Exception as e:
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=" Error in /serverHealth => {}".format(str(e)))

@app.post('/uploadManualRefinement')
async def uploadManualRefinement(file: fastapi.UploadFile = fastapi.File(...), identifier: str = fastapi.Form(...), user: str = fastapi.Form(...), caseName: str = fastapi.Form(...)
                                 , timeToBrush: float = fastapi.Form(...), brushStartEpoch: str = fastapi.Form(...)):

    try:
        
        # Step 0 - Init
        print ('|----------------------------------------------')

        # Step 1 - Read the file content
        try:
            file_content = await file.read()
        except Exception as e:
            traceback.print_exc()
            raise fastapi.HTTPException(status_code=500, detail="Error in /uploadManualRefinement => {}".format(str(e)))

        # Step 2 - Get file save path
        try:
            clientIdentifier = getClientIdentifier(identifier, user)
            pathFolderMask   = Path(SESSIONSGLOBAL[clientIdentifier][caseName][KEY_PATH_SAVE])
            if not pathFolderMask.exists():
                pathFolderMask.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            traceback.print_exc()
            raise fastapi.HTTPException(status_code=500, detail="Error in /uploadManualRefinement => clientIdentifier or caseName not found in SESSIONSGLOBAL")

        # Step 3 - Dicom tag modification
        try:
            # Step 3.1 - Make dicom file from memory
            dicom_file     = pydicom.filebase.DicomBytesIO(file_content)
            dicom_dataset  = pydicom.dcmread(dicom_file)

            # Step 3.2 - Modify dicom tags
            SESSIONSGLOBAL[clientIdentifier][caseName][KEY_MANUAL_COUNTER] += 1
            counter = SESSIONSGLOBAL[clientIdentifier][caseName][KEY_MANUAL_COUNTER]
            dicom_dataset.SeriesDescription = '-'.join([caseName, SERIESDESC_SUFFIX_REFINE_MAN, Path(pathFolderMask).parts[-1], '{:03d}'.format(counter)])
            dicom_dataset.SeriesNumber      = SERIESNUM_REFINE_MANUAL
            dicom_dataset.PatientName       = caseName
            dicom_dataset.PatientID         = caseName

            # Step 3.3 - Add private tags
            block = addAndGetPrivateBlockToDcm(dicom_dataset)
            block.add_new(TAG_TIME_TO_BRUSH, VALUEREP_FLOAT32, float(timeToBrush))
            block.add_new(TAG_TIME_EPOCH, VALUEREP_STRING, brushStartEpoch)
            timeToBrushFromDataset = dicom_dataset[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_TO_BRUSH)].value
            timeOfBrushStart = epoch_to_datetime(dicom_dataset[(PRIVATE_BLOCK_GROUP, TAG_OFFSET + TAG_TIME_EPOCH)].value)
            
            # Step 3.99 - Logging
            logStr = termcolor.colored("[{}] ".format(caseName), "blue") + termcolor.colored("timeToBrush: {:.2f}s".format(timeToBrushFromDataset), "green") + termcolor.colored(" | timeOfBrushStart: {}".format(timeOfBrushStart), "green")
            logger.info(logStr)

        except Exception as e:
            traceback.print_exc()
            raise fastapi.HTTPException(status_code=500, detail="Error in /uploadManualRefinement => dicom tag modification failed")

        # Step 4 - Save path
        try:
            pathFile = pathFolderMask.joinpath(file.filename.format(counter))
            try:
                dicom_dataset.save_as(pathFile)
                # TODO: why no return here?
            except Exception as e:
                traceback.print_exc()
                raise fastapi.HTTPException(status_code=500, detail="Error in /uploadManualRefinement => {}".format(str(e)))

        except:
            traceback.print_exc()
            raise fastapi.HTTPException(status_code=500, detail="Error in /uploadManualRefinement => clientIdentifier or caseName not found in SESSIONSGLOBAL")

    except Exception as e:
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=" Error in /uploadManualRefinement => {}".format(str(e)))
    
    print ('|----------------------------------------------')

@app.post('/uploadScrolledSliceIdxs')
async def uploadScrolledSliceIdxs(payload: PayloadScrolledSliceIdxs, request: starlette.requests.Request):

    try:
        
        # Step 0 - Init
        print ('|----------------------------------------------')
        clientIdentifier                   = getClientIdentifier(payload.identifier, payload.user) # payload.identifier + '__' + clientUserName
        uploadScrolledSliceIdxsPayloadData = payload.data.dict()
        patientName                        = uploadScrolledSliceIdxsPayloadData[KEY_CASE_NAME]
        returnMessagePrefix                = MSG_RETURN_PREFIX.format(clientIdentifier, patientName, LOAD_ONNX)

        # Step 1 - Check if session data is available
        dataAlreadyPresent = False
        if clientIdentifier not in SESSIONSGLOBAL:
            dataAlreadyPresent = False
        elif patientName not in SESSIONSGLOBAL[clientIdentifier]:
            dataAlreadyPresent = False
        else:
            dataAlreadyPresent = True

        # Step 2 - Extract scroll data
        if dataAlreadyPresent:
            
            # Step 2.1 - Get data
            uploadScrolledSliceIdxs = uploadScrolledSliceIdxsPayloadData[KEY_SLICE_IDXS_SCROLLED_OBJ]
            patientSliceIdxsObj     = SESSIONSGLOBAL[clientIdentifier][patientName][KEY_SLICE_IDXS_SCROLLED_OBJ]
            existingObjAxial = patientSliceIdxsObj[KEY_AXIAL]
            existingObjSag   = patientSliceIdxsObj[KEY_SAGITTAL]
            existingObjCor   = patientSliceIdxsObj[KEY_CORONAL]
            newObjAxial      = uploadScrolledSliceIdxs[KEY_AXIAL]
            newObjSag        = uploadScrolledSliceIdxs[KEY_SAGITTAL]
            newObjCor        = uploadScrolledSliceIdxs[KEY_CORONAL]
            print (' - [uploadScrolledSliceIdxs()] len(existingObjAxial): {}, len(existingObjSag): {}, len(existingObjCor): {}'.format(len(existingObjAxial), len(existingObjSag), len(existingObjCor)))
            print (' - [uploadScrolledSliceIdxs()] len(newObjAxial)     : {}, len(newObjSag)     : {}, len(newObjCor)     : {}'.format(len(newObjAxial), len(newObjSag), len(newObjCor)))

            # Step 2.2 - Update data
            existingObjAxial.update(newObjAxial)
            existingObjSag.update(newObjSag)
            existingObjCor.update(newObjCor)
            SESSIONSGLOBAL[clientIdentifier][patientName][KEY_SLICE_IDXS_SCROLLED_OBJ] = {KEY_AXIAL: existingObjAxial, KEY_SAGITTAL: existingObjSag, KEY_CORONAL: existingObjCor}
            # print (' - [uploadScrolledSliceIdxs()] SESSIONSGLOBAL[clientIdentifier][patientName][KEY_SLICE_IDXS_SCROLLED_OBJ]: {}'.format(SESSIONSGLOBAL[clientIdentifier][patientName][KEY_SLICE_IDXS_SCROLLED_OBJ]))

            # Step 2.3 - Dump .json
            pathFolderMask = Path(SESSIONSGLOBAL[clientIdentifier][patientName][KEY_PATH_SAVE])
            Path(pathFolderMask).mkdir(parents=True, exist_ok=True) 
            pathJSON       = pathFolderMask.joinpath('{}__{}'.format(patientName, SUFFIX_SLICE_SCROLL_JSON))
            with open(pathJSON, 'w') as f:
                json.dump(SESSIONSGLOBAL[clientIdentifier][patientName][KEY_SLICE_IDXS_SCROLLED_OBJ], f, indent=2)

            # Step 2.99 - Return
            print ('|----------------------------------------------')
            return {"status": "{} Scrolled slice indices uploaded successfully".format(returnMessagePrefix)}

        else:
            print ('|----------------------------------------------')
            raise fastapi.HTTPException(status_code=500, detail="{} Error in /uploadScrolledSliceIdx. No data present in python server.".format(returnMessagePrefix))

    except Exception as e:
        traceback.print_exc()
        raise fastapi.HTTPException(status_code=500, detail=" Error in /uploadScrolledSliceIdxs => {}".format(str(e)))

@app.post("/closeSession")
async def closeSession(payload: PayloadCloseSession, request: starlette.requests.Request):

    try:
        print ('|----------------------------------------------')
        clientIdentifier   = getClientIdentifier(payload.identifier, payload.user)
        returnMessagePrefix = MSG_RETURN_PREFIX.format(clientIdentifier, 'All', LOAD_ONNX)
        if clientIdentifier in SESSIONSGLOBAL:
            memBeforeObj = getMemoryUsage(printFlag=False, returnFlag=True)
            del SESSIONSGLOBAL[clientIdentifier]
            memAfterObj  = getMemoryUsage(printFlag=False, returnFlag=True)
            if memBeforeObj[KEY_RAM_USAGE_IN_GB] is not None and memAfterObj[KEY_RAM_USAGE_IN_GB] is not None:
                savedRAMInGB = float(memBeforeObj[KEY_RAM_USAGE_IN_GB]) - float(memAfterObj[KEY_RAM_USAGE_IN_GB])
            else:
                savedRAMInGB = -1
            if memBeforeObj[KEY_GPU_USAGE_IN_GB] is not None and memAfterObj[KEY_GPU_USAGE_IN_GB] is not None:
                savedGPUInGB = float(memBeforeObj[KEY_GPU_USAGE_IN_GB]) - float(memAfterObj[KEY_GPU_USAGE_IN_GB])
            else:
                savedGPUInGB = -1
            print (' - [closeSession()][{}] Session closed. Saved RAM: {:.2f} GB, GPU: {:.2f} GB'.format(returnMessagePrefix, savedRAMInGB, savedGPUInGB))

            if 0:
                def print_largest_memory_objects(limit=10):
                    # Take a snapshot of the current memory allocations
                    snapshot = tracemalloc.take_snapshot()

                    # Filter and sort the memory allocations to find the largest objects
                    top_stats = snapshot.statistics('lineno')

                    print(f"Top {limit} largest memory objects:")
                    for index, stat in enumerate(top_stats[:limit], 1):
                        print(f"{index}. {stat}")
                
                print_largest_memory_objects(limit=10)


            print ('|----------------------------------------------')
            return {"status": "{} Session closed. Saved RAM: {:.2f} GB, GPU: {:.2f} GB".format(returnMessagePrefix, savedRAMInGB, savedGPUInGB)}
        else:
            print (' - [closeSession()][{}] No session found'.format(returnMessagePrefix))
            print ('|----------------------------------------------')
            return {"status": "{} No session found".format(returnMessagePrefix)}


    except:
        traceback.print_exc()
        print ('|----------------------------------------------')
        raise fastapi.HTTPException(status_code=500, detail=" Error in /closeSession")

#################################################################
#                           MAIN
#################################################################

if __name__ == "__main__":

    try:
        
        if checkAssetPaths(verbose=False):
            logConfig = yaml.safe_load(open(PATH_LOGCONFIG, 'r'))
            if USE_HTTPS:
                print (f'\n - [main()] Starting server with HTTPS on {HOST} ...\n')
                uvicorn.run(f"{Path(__file__).stem}:app", host=HOST, port=PORT_PYTHON, ssl_keyfile=PATH_HOSTKEY, ssl_certfile=PATH_HOSTCERT, log_config=logConfig
                            , reload=True
                            , reload_excludes=PATHS_EXCLUDE_UTILS_RELATIVE, reload_includes=PATHS_INCLUDE_UTILS_RELATIVE
                            )
            else:
                print (f'\n - [main()] Starting server with HTTP on {HOST} ...\n')
                uvicorn.run(f"{Path(__file__).stem}:app", host=HOST, port=PORT_PYTHON, log_config=logConfig
                        , reload=True
                        , reload_excludes=PATHS_EXCLUDE_UTILS_RELATIVE, reload_includes=PATHS_INCLUDE_UTILS_RELATIVE
                        )
        else:
            print (' - [main()] Required files (e.g. SSL keys) not found. Exiting...')
               
    except KeyboardInterrupt:
        import sys; sys.exit(1)
        

"""
To-run
DO_HTTPS=True python src/backend/interactive-server.py
"""

"""
To-Do
1. Model training
 - [P] train the model with random sequence of 90 deg rotations along random axes
 - [P] train the model to do nothing when the scribble is made in a random region in the background.
 - [P] Change the Z-value of the distance map (randomly)
 - [P] include obedience loss in the model

2. Other stuff
 - difference between time.time() and time.process_time() 
 - do /prepare in parallel, otherwise its too time-consuming

3. Update Orthanc questions here
 - https://groups.google.com/g/orthanc-users/c/oUgOW8lctUw?pli=1
"""

"""
Data-Transformation Pipeline
1. FROM base-model pipelines TO .dcms (where were validated in 3D Slicer)
    - for scans 
        --> 3 x anti-clockwise rotations --> Flip LR
    - for SEG
        --> np.moveaxis(maskArray, [0,1,2], [2,1,0])

2. From .dcms (in python) TO numpy/torch arrays
"""

"""
TO RUN
0. conda activate interactive-refinement
1. python src/backend/interactive-server.py
  - nohup python -u src/backend/interactive-server.py > _logs/interactive-server-$(date +"%Y-%m-%d-%H-%M-%S").log 2>&1 &
2. Open your web browser and test for http://localhost:55000/
"""

"""
Unix commands
 - sudo netstat -tuln | grep -E '(:50000|:55000|:8042)'
"""