import pdb
import tqdm
import shutil
import datetime
import traceback
from pathlib import Path

# Constants
DIR_FILE                = Path(__file__).resolve().parent # <projectRoot>/src/backend/utils/
DIR_ROOT                = DIR_FILE.parent.parent.parent # <projectRoot>
DIR_TMP                 = DIR_ROOT / '_tmp'
DIR_EXPERIMENTS         = DIR_ROOT / '_experiments'

FOLDERNAME_JOHNDOE = 'experiments-john-doe'

DIR_EXPERIMENTS_JOHNDOE = DIR_EXPERIMENTS / FOLDERNAME_JOHNDOE
DIR_TMP_SCRIBBLES       = DIR_TMP / 'scribble-interactions-v4'
Path(DIR_TMP_SCRIBBLES).mkdir(parents=True, exist_ok=True)
print (f'\n - Created directory: {DIR_TMP_SCRIBBLES}')

FOLDERNAME_IGNORE_LIST = ['experiment-outputs', FOLDERNAME_JOHNDOE]

SEARCH_SUFFIX = '-interaction.png'

dateMinimum = datetime.datetime(2024, 11, 30)

def copyFileToTmpScribbles(pathExpFile):
    try:

        # Step 0.1 - Condition 1
        if pathExpFolder.name in FOLDERNAME_IGNORE_LIST:
            return
        
        # Step 0.2 - Condition 2
        dateStr = pathExpFolder.name.split('--')[0].strip()
        date = datetime.datetime.strptime(dateStr, '%Y-%m-%d %H-%M-%S')
        if date < dateMinimum:
            return

        for pathExpFile in Path(pathExpFolder).glob(f'**/*{SEARCH_SUFFIX}'):
            try:
                newPathExpFileName = '____'.join(Path(pathExpFile).parts[-2:])
                shutil.copy(pathExpFile, Path(DIR_TMP_SCRIBBLES).joinpath(newPathExpFileName))
            except:
                print (f' - Error in copying file: {pathExpFile}')
                traceback.print_exc()
                pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()

try:

    # Step 1 - Copying non-johndoe files
    print (f'\n - Looking for files with suffix: {SEARCH_SUFFIX} in {DIR_EXPERIMENTS} ({DIR_EXPERIMENTS.exists()})')
    pathExpFolders = [pathExpFolder for pathExpFolder in Path(DIR_EXPERIMENTS).iterdir() if pathExpFolder.is_dir()]
    with tqdm.tqdm(total=len(pathExpFolders)) as pbar:
        for pathExpFolder in pathExpFolders:
            
            copyFileToTmpScribbles(pathExpFolder)
            pbar.update(1)

    # Step 2 - Copying johndoe files
    print (f'\n - Looking for files with suffix: {SEARCH_SUFFIX} in {DIR_EXPERIMENTS_JOHNDOE} ({DIR_EXPERIMENTS_JOHNDOE.exists()})')
    pathExpFoldersJohnDoe = [pathExpFolder for pathExpFolder in Path(DIR_EXPERIMENTS_JOHNDOE).iterdir() if pathExpFolder.is_dir()]
    with tqdm.tqdm(total=len(pathExpFoldersJohnDoe)) as pbar: 
        for pathExpFolder in pathExpFoldersJohnDoe:
            copyFileToTmpScribbles(pathExpFolder)
            pbar.update(1)
    
    # Step 3 - Check how many files are copied
    totalFiles = len(list(Path(DIR_TMP_SCRIBBLES).glob(f"*{SEARCH_SUFFIX}")))
    print (f'\n - Total files copied: {totalFiles}')

except:
    traceback.print_exc()
    pdb.set_trace()