# Standard Library
import pdb
import json
import traceback
from pathlib import Path

# 3rd party libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Constants
DIR_FILE        = Path(__file__).resolve().parent # <projectRoot>/src/backend/utils/
DIR_ROOT        = DIR_FILE.parent.parent.parent # <projectRoot>/src/backend/
DIR_EXPERIMENTS         = DIR_ROOT / '_experiments'
DIR_EXPERIMENTS_OUTPUTS = DIR_EXPERIMENTS / 'experiment-outputs'

SUFFIX_SLICE_SCROLL_JSON = '__slice-scroll.json'
KEY_AXIAL   = 'Axial'
KEY_CORONAL = 'Coronal'
KEY_SAGITTAL= 'Sagittal'

COLOR_MANUAL_DICE = 'blue'
COLOR_SEMIAUTO_DICE = 'orange'

def openJSON(pathJSON):

    res = {}

    try:
        with open(pathJSON, 'r') as f:
            res = json.load(f)

    except:
        traceback.print_exc()
    
    return res

def convertObjToInt(scrollsObj):

    newScrollsObj = {}

    try:
        for view in scrollsObj.keys():
            newScrollsObj[view] = {}
            for epoch in scrollsObj[view].keys():
                newScrollsObj[view][int(epoch)] = int(scrollsObj[view][epoch])

    except:
        traceback.print_exc()
    
    return newScrollsObj

def getMinEpochInMS(scrollsObj):
    
    minEpoch = -1

    try:
        axialEpochInMS = list(scrollsObj[KEY_AXIAL].keys())
        saggitalEpochInMS = list(scrollsObj[KEY_SAGITTAL].keys())
        coronalEpochInMS = list(scrollsObj[KEY_CORONAL].keys())
        
        # minEpoch = min(min(axialEpochInMS), min(saggitalEpochInMS), min(coronalEpochInMS))
        minEpoch = min(axialEpochInMS + saggitalEpochInMS + coronalEpochInMS)

    except:
        traceback.print_exc()
    
    return minEpoch

def plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments):

    try:

        # Step 0 - Init (get paths)
        pathManualScrolls = pathManualExperiments / f'{patientId}{SUFFIX_SLICE_SCROLL_JSON}'
        pathSemiAutoScrolls = pathSemiAutoExperiments / f'{patientId}{SUFFIX_SLICE_SCROLL_JSON}'
        
        # Step 1 - Get data
        manualScrollsObj = openJSON(pathManualScrolls)
        if not manualScrollsObj:
            print (f' - [ERROR]: Could not load data from {pathManualScrolls}')
        semiAutoScrollsObj = openJSON(pathSemiAutoScrolls)
        if not semiAutoScrollsObj:
            print (f' - [ERROR]: Could not load data from {pathSemiAutoScrolls}')
        
        manualScrollsObj = convertObjToInt(manualScrollsObj)
        semiAutoScrollsObj = convertObjToInt(semiAutoScrollsObj)

        if len(manualScrollsObj):
            manualMinEpochInMs = getMinEpochInMS(manualScrollsObj)
            if manualMinEpochInMs == -1:
                print (f' - [ERROR]: Could not get manualMinEpochInMs')
        if len(semiAutoScrollsObj):
            semiAutoMinEpochInMs = getMinEpochInMS(semiAutoScrollsObj)
            if semiAutoMinEpochInMs == -1:
                print (f' - [ERROR]: Could not get semiAutoMinEpochInMs')
            
        # Step 2 - Split data (as per view)
        if len(semiAutoScrollsObj):
            semiAutoAxial = semiAutoScrollsObj[KEY_AXIAL] # {epoch:sliceId}
            semiAutoCoronal = semiAutoScrollsObj[KEY_CORONAL]
            semiAutoSagittal = semiAutoScrollsObj[KEY_SAGITTAL]
        
        if len(manualScrollsObj):
            manualAxial = manualScrollsObj[KEY_AXIAL]
            manualCoronal = manualScrollsObj[KEY_CORONAL]
            manualSagittal = manualScrollsObj[KEY_SAGITTAL]

        # Step 3 - Plot 
        # Idea 3.1 (6 plots with x-axis as timestamp and y-axis as slice ID in that particular view)
        if 0:
            f,axarr = plt.subplots(6,1,figsize=(10,10), sharex=True)
                        
            sns.lineplot(x=np.array(list(semiAutoAxial.keys()))-semiAutoMinEpochInMs, y=list(semiAutoAxial.values()), ax=axarr[0], label='Semi-Auto', color=COLOR_SEMIAUTO_DICE)
            sns.scatterplot(x=np.array(list(semiAutoAxial.keys()))-semiAutoMinEpochInMs, y=list(semiAutoAxial.values()), ax=axarr[0], label='Semi-Auto', color=COLOR_SEMIAUTO_DICE)
            sns.lineplot(x=np.array(list(manualAxial.keys()))-manualMinEpochInMs, y=list(manualAxial.values()), ax=axarr[3], label='Manual', color=COLOR_MANUAL_DICE)
            sns.scatterplot(x=np.array(list(manualAxial.keys()))-manualMinEpochInMs, y=list(manualAxial.values()), ax=axarr[3], label='Manual', color=COLOR_MANUAL_DICE)

            sns.lineplot(x=np.array(list(semiAutoCoronal.keys()))-semiAutoMinEpochInMs, y=list(semiAutoCoronal.values()), ax=axarr[1], label='Semi-Auto', color=COLOR_SEMIAUTO_DICE)
            sns.scatterplot(x=np.array(list(semiAutoCoronal.keys()))-semiAutoMinEpochInMs, y=list(semiAutoCoronal.values()), ax=axarr[1], label='Semi-Auto', color=COLOR_SEMIAUTO_DICE)
            sns.lineplot(x=np.array(list(manualCoronal.keys()))-manualMinEpochInMs, y=list(manualCoronal.values()), ax=axarr[4], label='Manual', color=COLOR_MANUAL_DICE)
            sns.scatterplot(x=np.array(list(manualCoronal.keys()))-manualMinEpochInMs, y=list(manualCoronal.values()), ax=axarr[4], label='Manual', color=COLOR_MANUAL_DICE)
            
            sns.lineplot(x=np.array(list(semiAutoSagittal.keys()))-semiAutoMinEpochInMs, y=list(semiAutoSagittal.values()), ax=axarr[2], label='Semi-Auto', color=COLOR_SEMIAUTO_DICE)
            sns.scatterplot(x=np.array(list(semiAutoSagittal.keys()))-semiAutoMinEpochInMs, y=list(semiAutoSagittal.values()), ax=axarr[2], label='Semi-Auto', color=COLOR_SEMIAUTO_DICE)
            sns.lineplot(x=np.array(list(manualSagittal.keys()))-manualMinEpochInMs, y=list(manualSagittal.values()), ax=axarr[5], label='Manual', color=COLOR_MANUAL_DICE)
            sns.scatterplot(x=np.array(list(manualSagittal.keys()))-manualMinEpochInMs, y=list(manualSagittal.values()), ax=axarr[5], label='Manual', color=COLOR_MANUAL_DICE)
            
            axarr[5].set_xlabel('Time (ms)')
        
        elif 0:
            f,axarr = plt.subplots(2,1,figsize=(10,10), sharex=True); s=15; alpha=0.8
            if len(semiAutoScrollsObj):
                sns.scatterplot(x=np.array(list(semiAutoAxial.keys()))-semiAutoMinEpochInMs, y=list(semiAutoAxial.values()), ax=axarr[0], label='Axial', color=COLOR_SEMIAUTO_DICE, marker='o', s=s, alpha=alpha)
                sns.scatterplot(x=np.array(list(semiAutoCoronal.keys()))-semiAutoMinEpochInMs, y=list(semiAutoCoronal.values()), ax=axarr[0], label='Sagittal', color=COLOR_SEMIAUTO_DICE, marker='x',s=s+5, alpha=alpha)
                sns.scatterplot(x=np.array(list(semiAutoSagittal.keys()))-semiAutoMinEpochInMs, y=list(semiAutoSagittal.values()), ax=axarr[0], label='Coronal', color=COLOR_SEMIAUTO_DICE, marker='v',s=s+5, alpha=alpha)

            if len(manualScrollsObj):
                sns.scatterplot(x=np.array(list(manualAxial.keys()))-manualMinEpochInMs, y=list(manualAxial.values()), ax=axarr[1], label='Axial', color=COLOR_MANUAL_DICE, marker='o',s=s, alpha=alpha)
                sns.scatterplot(x=np.array(list(manualCoronal.keys()))-manualMinEpochInMs, y=list(manualCoronal.values()), ax=axarr[1], label='Sagittal', color=COLOR_MANUAL_DICE, marker='x',s=s+5, alpha=alpha)
                sns.scatterplot(x=np.array(list(manualSagittal.keys()))-manualMinEpochInMs, y=list(manualSagittal.values()), ax=axarr[1], label='Coronal', color=COLOR_MANUAL_DICE, marker='v',s=s+5, alpha=alpha)
        
        elif 1:
            pass

        f.suptitle(f'Patient: {patientId} - User: {userName}')
        plt.show()

    except:
        traceback.print_exc()
        pdb.set_trace()

if __name__ == "__main__":
    
    
    ############ P1 (CHUP-033)
    if 0:
        patientId = 'CHUP-033-gt-filtered-gausssig2'
        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-03 14-40-44 -- pedantic_lehmann__Martin-De Jong-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-12 14-50-57 -- brave_diffie__Martin-De Jong-Expert-AI-based'
            userName = 'P1(CHUP-033)-C1(Martin)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)
        
        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-06 14-31-12 -- reverent_taussig__Mischa-de Ridder-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-03 08-12-22 -- jolly_hopper__Mischa-de Ridder-Expert-AI-based'
            userName = 'P1(CHUP-033)-C2(Mishca)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)
        
        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-12 13-54-50 -- exciting_poincare__Niels-den Haan-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-09 12-15-16 -- modest_lamarr__Niels-den Haan-Expert-AI-based' 
            userName = 'P1(CHUP-033)-C3(Niels)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)
        
        if 1:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-06 08-47-00 -- jovial_vaughan__Jos-Elbers-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-05 08-28-22 -- fervent_lederberg__Jos-Elbers-Expert-AI-based' 
            userName = 'P1(CHUP-033)-C4(Jos)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)
    
    ############ P3(CHUP-005)
    if 0:
        patientId = 'CHUP-005-gt-filtered-gausssig2'
        if 1:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-05 13-00-58 -- tender_hugle__Martin-De Jong-Expert-Manual'
            pathsSemiAutoExperiments = DIR_EXPERIMENTS / '2024-11-29 09-43-36 -- distracted_feistel__Martin-De Jong-Expert-AI-based' 
            userName = 'P3(CHUP-005)-C1(Martin)'
            plotScrolling(patientId, userName, pathManualExperiments, pathsSemiAutoExperiments)

        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-09 15-03-46 -- dreamy_ritchie__Mischa-de ridder-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-18 15-38-23 -- silly_goodall__Mischa-de Ridder-Expert-AI-based'
            userName = 'P3(CHUP-005)-C2(Mischa)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)
        
        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-16 12-00-19 -- great_jennings__Niels -den Haan-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-09 12-26-54 -- crazy_cohen__Niels-den Haan-Expert-AI-based' 
            userName = 'P3(CHUP-005)-C3(Niels)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)
        
        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-13 09-05-17 -- gracious_ride__Jos-Elbers-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-05 08-36-55 -- epic_leavitt__Jos-Elbers-Expert-AI-based' 
            userName = 'P3(CHUP-005)-C4(Jos)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)

    ############ P5(CHUP-028)
    if 1:
        patientId = 'CHUP-028-gt-filtered-gausssig2'
        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-05 13-12-15 -- zealous_rhodes__Martin-De Jong-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-03 14-10-13 -- objective_austin__Martin-De Jong-Expert-AI-based' 
            userName = 'P5(CHUP-028)-C1(Martin)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)

        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-09 15-16-47 -- elastic_thompson__Mischa-de Ridder-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-18 15-43-51 -- exciting_grothendieck__Mischa-de Ridder-Expert-AI-based'
            userName = 'P5(CHUP-028)-C2(Mischa)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)
        
        if 0:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-16 12-08-22 -- great_noether__Niels-den Haan-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-12 13-43-11 -- nostalgic_payne__Niels-den Haan-Expert-AI-based' 
            userName = 'P5(CHUP-028)-C3(Niels)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)

        if 1:
            pathManualExperiments = DIR_EXPERIMENTS / '2024-12-13 09-12-00 -- mystifying_shamir__Jos-Elbers-Expert-Manual'
            pathSemiAutoExperiments = DIR_EXPERIMENTS / '2024-12-06 08-35-46 -- focused_austin__Jos-Elbers-Expert-AI-based' 
            userName = 'P5(CHUP-028)-C4(Jos)'
            plotScrolling(patientId, userName, pathManualExperiments, pathSemiAutoExperiments)