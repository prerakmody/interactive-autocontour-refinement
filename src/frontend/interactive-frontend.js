import dicomParser from 'dicom-parser';

import * as cornerstone3D from '@cornerstonejs/core';
import * as cornerstoneAdapters from "@cornerstonejs/adapters"; // dont remove this, gives cirucular dependency error: ReferenceError: Cannot access '__WEBPACK_DEFAULT_EXPORT__' before initialization
import * as cornerstoneDICOMImageLoader from '@cornerstonejs/dicom-image-loader';
import * as cornerstoneStreamingImageLoader from '@cornerstonejs/streaming-image-volume-loader';

import * as cornerstone3DTools from '@cornerstonejs/tools';

import createImageIdsAndCacheMetaData from './helpers/createImageIdsAndCacheMetaData'; // https://github.com/cornerstonejs/cornerstone3D/blob/a4ca4dde651d17e658a4aec5a4e3ec1b274dc580/utils/demo/helpers/createImageIdsAndCacheMetaData.js
// // import setPetColorMapTransferFunctionForVolumeActor from './helpers/setPetColorMapTransferFunctionForVolumeActor'; //https://github.com/cornerstonejs/cornerstone3D/blob/v1.77.13/utils/demo/helpers/setPetColorMapTransferFunctionForVolumeActor.js
// // import setCtTransferFunctionForVolumeActor from './helpers/setCtTransferFunctionForVolumeActor'; // https://github.com/cornerstonejs/cornerstone3D/blob/v1.77.13/utils/demo/helpers/setCtTransferFunctionForVolumeActor.js

import * as config from './helpers/config'
import * as makeGUIElementsHelper from './helpers/makeGUIElementsHelper'
import * as updateGUIElementsHelper from './helpers/updateGUIElementsHelper'
import * as keyboardEventsHelper from './helpers/keyboardEventsHelper'
import * as apiEndpointHelpers from './helpers/apiEndpointHelpers'
import * as segmentationHelpers from './helpers/segmentationHelpers'

import { vec3 } from 'gl-matrix';



// **************************** VARIABLES ************************************

// HTML ids
const interactionButtonsDivId = 'interactionButtonsDiv'
const axialID                 = 'ViewPortId-Axial';
const sagittalID              = 'ViewPortId-Sagittal';
const coronalID               = 'ViewPortId-Coronal';
const viewportIds             = [axialID, sagittalID, coronalID];
const otherButtonsDivId       = 'otherButtonsDiv';


// Rendering + Volume + Segmentation ids
const renderingEngineId        = 'myRenderingEngine';
// const toolGroupIdContours      = 'MY_TOOL_GROUP_ID_CONTOURS';
const toolGroupIdScribble      = 'MY_TOOL_GROUP_ID_SCRIBBLE'; // not in use, failed experiment: Multiple tool groups found for renderingEngineId: myRenderingEngine and viewportId: ViewPortId-Axial. You should only have one tool group per viewport in a renderingEngine.
const volumeLoaderScheme       = 'cornerstoneStreamingImageVolume';
const volumeIdPETBase      = `${volumeLoaderScheme}:myVolumePET`; //+ cornerstone3D.utilities.uuidv4()
const volumeIdCTBase       = `${volumeLoaderScheme}:myVolumeCT`;

const MASK_TYPE_GT   = 'GT';
const MASK_TYPE_PRED = 'PRED';

const INIT_BRUSH_SIZE = 5

const scribbleSegmentationIdBase = `SCRIBBLE_SEGMENTATION_ID`; // this should not change for different scribbles

let scribbleSegmentationId;

const SEG_TYPE_LABELMAP = 'LABELMAP'

// General
let fusedPETCT   = false;
let petBool      = false;
let totalImagesIdsCT = undefined;
let totalImagesIdsPET = undefined;
let totalROIsRTSTRUCT = undefined;

/****************************************************************
*                         HTML ELEMENTS  
*****************************************************************/

// Step 1 - Create Viewports
await makeGUIElementsHelper.createViewPortsHTML();
let axialDiv=config.axialDiv, sagittalDiv=config.sagittalDiv, coronalDiv=config.coronalDiv;
let axialDivPT=config.axialDivPT, sagittalDivPT=config.sagittalDivPT, coronalDivPT=config.coronalDivPT;

// Step 2 - Create buttons
await makeGUIElementsHelper.createContouringHTML();

// Step 3 - Create case selection dropdown
const {caseSelectionHTML} = await makeGUIElementsHelper.otherHTMLElements();
caseSelectionHTML.addEventListener('change', async function() {
    config.setPatientIdx(parseInt(this.value));
    await fetchAndLoadData(config.patientIdx);
});

function printHeaderInConsole(strToPrint){
    console.log(`\n\n | ================================================================ ${strToPrint} ================================================================ | \n\n`)
}

/****************************************************************
*                             UTILS  
*****************************************************************/

let orthancHeaders = new Headers();
orthancHeaders.set('Authorization', 'Basic ' + btoa('orthanc'+ ":" + 'orthanc'));

async function setupDropDownMenu(orthanDataURLS, patientIdx) {

    const cases = Array.from({length: orthanDataURLS.length}, (_, i) => orthanDataURLS[i].caseName).filter(caseName => caseName.length > 0);
    
    cases.forEach((caseName, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.text = index + ' - ' + caseName;
        caseSelectionHTML.appendChild(option);

        if (index == patientIdx) 
            option.selected = true;
    });

    caseSelectionHTML.selectedIndex = patientIdx;
}

function calculatePlaneNormal(imageOrientation) {
    const rowCosineVec = vec3.fromValues(
      imageOrientation[0],
      imageOrientation[1],
      imageOrientation[2]
    );
    const colCosineVec = vec3.fromValues(
      imageOrientation[3],
      imageOrientation[4],
      imageOrientation[5]
    );
    return vec3.cross(vec3.create(), rowCosineVec, colCosineVec);
}
  
function sortImageIds(imageIds) {
    const { imageOrientationPatient } = cornerstone3D.metaData.get(
      'imagePlaneModule',
      imageIds[0]
    );
    const scanAxisNormal = calculatePlaneNormal(imageOrientationPatient);
    const { sortedImageIds } = cornerstone3D.utilities.sortImageIdsAndGetSpacing(
      imageIds,
      scanAxisNormal
    );
    return sortedImageIds;
}

// ******************************* CORNERSTONE FUNCS *********************************

async function cornerstoneInit() {

    cornerstoneDICOMImageLoader.external.cornerstone = cornerstone3D;
    cornerstoneDICOMImageLoader.external.dicomParser = dicomParser;
    cornerstone3D.volumeLoader.registerVolumeLoader('cornerstoneStreamingImageVolume',cornerstoneStreamingImageLoader.cornerstoneStreamingImageVolumeLoader);

    // Step 3.1.2 - Init cornerstone3D and cornerstone3DTools
    await cornerstone3D.init();
    const c3DConfig = cornerstone3D.getConfiguration();
    const detectedFPS = c3DConfig.gpuTier.fps;
    config.setDetectedFPS(detectedFPS);
    console.log(' - cornerstone3D.getConfiguration(): ', cornerstone3D.getConfiguration())
}

async function getToolsAndToolGroup() {

    // Step 1 - Init cornerstone3DTools
    await cornerstone3DTools.init();    

    // // Step 2 - Get tools
    const windowLevelTool           = cornerstone3DTools.WindowLevelTool;
    const panTool                   = cornerstone3DTools.PanTool;
    const zoomTool                  = cornerstone3DTools.ZoomTool;
    const stackScrollMouseWheelTool = cornerstone3DTools.StackScrollMouseWheelTool;
    const probeTool                 = cornerstone3DTools.ProbeTool;
    const referenceLinesTool        = cornerstone3DTools.ReferenceLines;
    const segmentationDisplayTool   = cornerstone3DTools.SegmentationDisplayTool;
    const brushTool                 = cornerstone3DTools.BrushTool;
    const planarFreeHandRoiTool     = cornerstone3DTools.PlanarFreehandROITool;
    const planarFreeHandContourTool = cornerstone3DTools.PlanarFreehandContourSegmentationTool; 
    const sculptorTool              = cornerstone3DTools.SculptorTool;
    // const toolState      = cornerstone3DTools.state;
    // const {segmentation} = cornerstone3DTools;

    // Step 3 - init tools
    cornerstone3DTools.addTool(windowLevelTool);
    cornerstone3DTools.addTool(panTool);
    cornerstone3DTools.addTool(zoomTool);
    cornerstone3DTools.addTool(stackScrollMouseWheelTool);
    cornerstone3DTools.addTool(probeTool);
    cornerstone3DTools.addTool(referenceLinesTool);
    cornerstone3DTools.addTool(segmentationDisplayTool);
    cornerstone3DTools.addTool(planarFreeHandRoiTool);
    if (config.MODALITY_CONTOURS == config.MODALITY_SEG)
        cornerstone3DTools.addTool(brushTool);
    else if (config.MODALITY_CONTOURS == MODALITY_RTSTRUCT){
        cornerstone3DTools.addTool(planarFreeHandContourTool);
        cornerstone3DTools.addTool(sculptorTool);
    }
     
    // Step 4.1 - Make toolGroupContours
    const toolGroupContours = cornerstone3DTools.ToolGroupManager.createToolGroup(config.toolGroupIdContours);
    toolGroupContours.addTool(windowLevelTool.toolName);
    toolGroupContours.addTool(panTool.toolName);
    toolGroupContours.addTool(zoomTool.toolName);
    toolGroupContours.addTool(stackScrollMouseWheelTool.toolName);
    toolGroupContours.addTool(probeTool.toolName);
    toolGroupContours.addTool(referenceLinesTool.toolName);
    toolGroupContours.addTool(segmentationDisplayTool.toolName);

    if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
        toolGroupContours.addTool(brushTool.toolName);
        toolGroupContours.addToolInstance(config.strBrushCircle, brushTool.toolName, { activeStrategy: 'FILL_INSIDE_CIRCLE', brushSize:INIT_BRUSH_SIZE}) ;
        toolGroupContours.addToolInstance(config.strEraserCircle, brushTool.toolName, { activeStrategy: 'ERASE_INSIDE_CIRCLE', brushSize:INIT_BRUSH_SIZE});
    }
    else if (config.MODALITY_CONTOURS == config.MODALITY_RTSTRUCT){
        toolGroupContours.addTool(planarFreeHandContourTool.toolName);
        toolGroupContours.addTool(sculptorTool.toolName);
    }

    // Step 4.2 - Make toolGroupScribble
    // const toolGroupScribble = cornerstone3DTools.ToolGroupManager.createToolGroup(toolGroupIdScribble);
    toolGroupContours.addTool(planarFreeHandRoiTool.toolName);

    // Step 5 - Set toolGroup(s) elements as active/passive
    toolGroupContours.setToolPassive(windowLevelTool.toolName);// Left Click
    toolGroupContours.setToolActive(panTool.toolName, {bindings: [{mouseButton: cornerstone3DTools.Enums.MouseBindings.Auxiliary, },],}); // Middle Click
    toolGroupContours.setToolActive(zoomTool.toolName, {bindings: [{mouseButton: cornerstone3DTools.Enums.MouseBindings.Secondary, },],}); // Right Click    
    toolGroupContours.setToolActive(stackScrollMouseWheelTool.toolName);
    toolGroupContours.setToolEnabled(probeTool.toolName);
    toolGroupContours.setToolEnabled(referenceLinesTool.toolName);
    // toolGroupContours.setToolConfiguration(referenceLinesTool.toolName, {sourceViewportId: axialID,});
    config.viewPortDivsAll.forEach((viewportDiv, index) => {
        viewportDiv.addEventListener('mouseenter', function() {
            // console.log(' - [cornerstoneInit()] Mouse entered viewportIds[index]: ', config.viewPortIdsAll[index]);
            toolGroupContours.setToolConfiguration(referenceLinesTool.toolName, {sourceViewportId: config.viewPortIdsAll[index], lineWidth: 1,});
        });
    });

    // Step 5.2 - Set all contouring tools as passive
    toolGroupContours.setToolEnabled(segmentationDisplayTool.toolName);
    if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
        // toolGroupContours.setToolPassive(brushTool.toolName);
        toolGroupContours.setToolPassive(config.strBrushCircle); // , { bindings: [ { mouseButton: cornerstone3DTools.Enums.MouseBindings.Primary, }, ], });
        toolGroupContours.setToolPassive(config.strEraserCircle); // , { bindings: [ { mouseButton: cornerstone3DTools.Enums.MouseBindings.Primary, }, ], });
    } else if (config.MODALITY_CONTOURS == config.MODALITY_RTSTRUCT){
        toolGroupContours.setToolPassive(planarFreeHandContourTool.toolName);
        toolGroupContours.setToolPassive(sculptorTool.toolName);
    }

    // Step 5.3 - Set config.toolGroupIdContours elements as passive
    toolGroupContours.setToolPassive(planarFreeHandRoiTool.toolName);
    toolGroupContours.setToolConfiguration(planarFreeHandRoiTool.toolName, {calculateStats: false, lineWidth:50}); // TODO: linewidth does not work

    // Step 6 - Make toolGroup3D
    const trackballRotateTool = cornerstone3DTools.TrackballRotateTool;
    cornerstone3DTools.addTool(trackballRotateTool);

    const toolGroup3D = cornerstone3DTools.ToolGroupManager.createToolGroup(config.toolGroupId3D);
    toolGroup3D.addTool(panTool.toolName);
    toolGroup3D.addTool(zoomTool.toolName);
    toolGroup3D.addTool(trackballRotateTool.toolName);
    toolGroup3D.addTool(segmentationDisplayTool.toolName);
    toolGroup3D.setToolActive(trackballRotateTool.toolName);
    toolGroup3D.setToolActive(panTool.toolName, {bindings: [{mouseButton: cornerstone3DTools.Enums.MouseBindings.Auxiliary, },],}); // Middle Click
    toolGroup3D.setToolActive(zoomTool.toolName, {bindings: [{mouseButton: cornerstone3DTools.Enums.MouseBindings.Secondary, modifierKey: cornerstone3DTools.Enums.KeyboardBindings.Shift,},],}); // Right Click    
    toolGroup3D.setToolActive(trackballRotateTool.toolName, {bindings: [{mouseButton: cornerstone3DTools.Enums.MouseBindings.Primary, },],}); // Left Click
    toolGroup3D.setToolEnabled(segmentationDisplayTool.toolName);
}

async function setRenderingEngineAndViewports(){

    const renderingEngine = new cornerstone3D.RenderingEngine(renderingEngineId);

    // Step 2.5.1 - Add image planes to rendering engine
    const viewportInputs = [
        {element: axialDiv     , viewportId: config.axialID            , type: cornerstone3D.Enums.ViewportType.ORTHOGRAPHIC, defaultOptions: { orientation: cornerstone3D.Enums.OrientationAxis.AXIAL},},
        {element: sagittalDiv  , viewportId: config.sagittalID         , type: cornerstone3D.Enums.ViewportType.ORTHOGRAPHIC, defaultOptions: { orientation: cornerstone3D.Enums.OrientationAxis.SAGITTAL},},
        {element: coronalDiv   , viewportId: config.coronalID          , type: cornerstone3D.Enums.ViewportType.ORTHOGRAPHIC, defaultOptions: { orientation: cornerstone3D.Enums.OrientationAxis.CORONAL},},
        {element: axialDivPT   , viewportId: config.axialPTID   , type: cornerstone3D.Enums.ViewportType.ORTHOGRAPHIC, defaultOptions: { orientation: cornerstone3D.Enums.OrientationAxis.AXIAL},},
        {element: sagittalDivPT, viewportId: config.sagittalPTID, type: cornerstone3D.Enums.ViewportType.ORTHOGRAPHIC, defaultOptions: { orientation: cornerstone3D.Enums.OrientationAxis.SAGITTAL},},
        {element: coronalDivPT , viewportId: config.coronalPTID , type: cornerstone3D.Enums.ViewportType.ORTHOGRAPHIC, defaultOptions: { orientation: cornerstone3D.Enums.OrientationAxis.CORONAL},},
        {element: config.viewport3DDiv, viewportId: config.viewport3DId, type: cornerstone3D.Enums.ViewportType.VOLUME_3D, defaultOptions: { background: cornerstone3D.CONSTANTS.BACKGROUND_COLORS.slicer3D},},
    ]
    renderingEngine.setViewports(viewportInputs);
    
    // Step 2.5.2 - Add config.toolGroupIdContours to rendering engine
    const toolGroup = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupIdContours);
    config.viewPortIdsAll.forEach((viewportId) =>
        toolGroup.addViewport(viewportId, renderingEngineId)
    );

    // Step 2.5.3 - Add config.toolGroupId3D to rendering engine
    const toolGroup3D = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupId3D);
    toolGroup3D.addViewport(config.viewport3DId, renderingEngineId);

    // return {renderingEngine};
}

// (Sub) MAIN FUNCTION
async function restart() {
    
    try{
        printHeaderInConsole('Step 0 - restart()');

        // Step 1 - Block GUI
        [config.windowLevelButton , config.contourSegmentationToolButton, config.sculptorToolButton, config.editBaseContourViaScribbleButton].forEach((buttonHTML) => {
            if (buttonHTML === null) return;
            makeGUIElementsHelper.setButtonBoundaryColor(buttonHTML, false);
            buttonHTML.disabled = true;
        });

        // Other GUI
        apiEndpointHelpers.setServerStatus(0);
        
        // Step 2 - Remove all segmentationIds
        try{

            const allSegObjs     = cornerstone3DTools.segmentation.state.getSegmentations();
            const allSegRepsObjs = cornerstone3DTools.segmentation.state.getAllSegmentationRepresentations()[config.toolGroupIdContours];
            allSegObjs.forEach(segObj => {
                const thisSegRepsObj = allSegRepsObjs.filter(obj => obj.segmentationId === segObj.segmentationId)[0]
                cornerstone3DTools.segmentation.removeSegmentationsFromToolGroup(config.toolGroupIdContours, [thisSegRepsObj.segmentationRepresentationUID,], false);
                cornerstone3DTools.segmentation.state.removeSegmentation(segObj.segmentationId);
                if (segObj.type == SEG_TYPE_LABELMAP)
                    cornerstone3D.cache.removeVolumeLoadObject(segObj.segmentationId);
            });
            console.log(' - [restart()] new allSegObjs      : ', cornerstone3DTools.segmentation.state.getSegmentations());
            console.log(' - [restart()] new allSegIdsAndUIDs: ', cornerstone3DTools.segmentation.state.getAllSegmentationRepresentations());

        } catch (error){
            console.error(' - [restart()] Error in removing segmentations: ', error);
            updateGUIElementsHelper.showToast('Error in removing segmentations', 3000);
        }

        // Step 3 - Clear cache (images and volumes)
        // await cornerstone3D.cache.purgeCache(); // does cache.removeVolumeLoadObject() and cache.removeImageLoadObject() inside // cornerstone3D.cache.getVolumes(), cornerstone3D.cache.getCacheSize()
        if (config.volumeIdCT != undefined) cornerstone3D.cache.removeVolumeLoadObject(config.volumeIdCT);
        if (config.volumeIdPET != undefined) cornerstone3D.cache.removeVolumeLoadObject(config.volumeIdPET);

        // Step 4 - Clear other patientSpecific data
        config.resetScrolledSliceIdxsObj();

    } catch (error){
        console.error(' - [restart()] Error: ', error);
        updateGUIElementsHelper.showToast('Error in restart()', 3000);
    }
    
    // Step 4 - Reset global variables
    fusedPETCT   = false;
    petBool      = false;
    config.setVolumeIdCT(undefined);
    config.setVolumeIdPET(undefined);
    totalImagesIdsCT  = undefined;
    totalImagesIdsPET = undefined;
    totalROIsRTSTRUCT = undefined;


    // Step 5 - Other stuff
    // setToolsAsActivePassive(true);
    // const stackScrollMouseWheelTool = cornerstone3DTools.StackScrollMouseWheelTool;
    // const toolGroup = cornerstone3DTools.ToolGroupManager.getToolGroup(toolGroupId);
    // toolGroup.setToolPassive(stackScrollMouseWheelTool.toolName);

}

// MAIN FUNCTION
async function fetchAndLoadData(patientIdx){

    config.setAllDivsLoaded(false);
    await updateGUIElementsHelper.showLoaderAnimation();
    await restart();
    printHeaderInConsole('Step 1 - Getting .dcm data')
    
    //////////////////////////////////////////////////////////// Step 1 - Get search parameters
    if (config.orthanDataURLS.length >= patientIdx+1){
        
        const {caseName, reverseImageIds, searchObjCT, searchObjPET, searchObjRTSGT, searchObjRTSPred} = config.orthanDataURLS[patientIdx];
        
        ////////////////////////////////////////////////////////////// Step 2.1 - Create volume for CT
        if (searchObjCT.wadoRsRoot.length > 0){

            // Step 2.1.0 - Init for CT load
            const renderingEngine = cornerstone3D.getRenderingEngine(renderingEngineId);

            // Step 2.1.1 - Load CT data (in python server)
            const timeNow = new Date().getTime();
            apiEndpointHelpers.makeRequestToPrepare(patientIdx)
            console.log(' - [loadData()] Time taken to prepare data on python server: ', new Date().getTime() - timeNow);

            // Step 2.1.2 - Fetch CT data
            config.setVolumeIdCT([volumeIdCTBase, cornerstone3D.utilities.uuidv4()].join(':'));
            config.setCTFetchBool(false);
            // global.ctFetchBool = false;
            try{
                let imageIdsCTTmp = await createImageIdsAndCacheMetaData(searchObjCT);
                imageIdsCTTmp = sortImageIds(imageIdsCTTmp);
                config.setImageIdsCT(imageIdsCTTmp);
                // global.ctFetchBool = true;
                config.setCTFetchBool(true);
            } catch (error){
                console.error(' - [loadData()] Error in createImageIdsAndCacheMetaData(searchObjCT): ', error);
                updateGUIElementsHelper.showToast('Error in loading CT data', 3000);
            }
            
            // Step 2.1.3 - Load CT data
            // if (global.ctFetchBool){
            if (config.ctFetchBool){
                try{
                    if (reverseImageIds){
                        config.setImageIdsCT(config.imageIdsCT.reverse());
                        console.log(' - [loadData()] Reversed imageIdsCT');
                    }
                    totalImagesIdsCT = config.imageIdsCT.length;
                    const volumeCT   = await cornerstone3D.volumeLoader.createAndCacheVolume(config.volumeIdCT, { imageIds:config.imageIdsCT});
                    await volumeCT.load();
                    
                } catch (error){
                    console.error(' - [loadData()] Error in createAndCacheVolume(volumeIdCT, { imageIds:imageIdsCT }): ', error);
                    updateGUIElementsHelper.showToast('Error in creating volume for CT data', 3000);
                }

                ////////////////////////////////////////////////////////////// Step 2.2 - Create volume for PET
                if (searchObjPET.wadoRsRoot.length > 0){
                    
                    // Step 2.2.1 - Fetch PET data
                    config.setVolumeIdPET([volumeIdPETBase, cornerstone3D.utilities.uuidv4()].join(':'));
                    // let petFetchBool  = false;
                    config.setPTFetchBool(false);
                    let imageIdsPET  = [];
                    try{
                        imageIdsPET = await createImageIdsAndCacheMetaData(searchObjPET);
                        // petFetchBool = true;
                        config.setPTFetchBool(true);
                    } catch(error){
                        console.error(' - [loadData()] Error in createImageIdsAndCacheMetaData(searchObjPET): ', error);
                        updateGUIElementsHelper.showToast('Error in loading PET data', 3000);
                    }

                    // Step 2.2.2 - Load PET data
                    // if (petFetchBool){
                    if (config.ptFetchBool){

                        try{
                            if (reverseImageIds){
                                imageIdsPET = imageIdsPET.reverse();
                            }
                            totalImagesIdsPET = imageIdsPET.length;
                            if (totalImagesIdsPET != totalImagesIdsCT)
                                updateGUIElementsHelper.showToast(`CT (${totalImagesIdsCT}) and PET (${totalImagesIdsPET}) have different number of imageIds`, 5000);
                            const volumePT    = await cornerstone3D.volumeLoader.createAndCacheVolume(config.volumeIdPET, { imageIds: imageIdsPET });
                            volumePT.load();
                            petBool = true;
                        } catch (error){
                            console.error(' - [loadData()] Error in createAndCacheVolume(volumeIdPET, { imageIds:imageIdsPET }): ', error);
                            updateGUIElementsHelper.showToast('Error in creating volume for PET data', 3000);
                        }
                    }
                }

                ////////////////////////////////////////////////////////////// Step 3 - Set volumes for viewports
                await cornerstone3D.setVolumesForViewports(renderingEngine, [{ volumeId:config.volumeIdCT}, ], viewportIds, true);
                await cornerstone3D.setVolumesForViewports(renderingEngine, [{ volumeId:config.volumeIdPET}, ], config.viewPortPTIds, true);
                
                ////////////////////////////////////////////////////////////// Step 4 - Render viewports
                await renderingEngine.renderViewports(viewportIds);
                await renderingEngine.renderViewports(config.viewPortPTIds);
                await renderingEngine.renderViewports([config.viewport3DId]);

                ////////////////////////////////////////////////////////////// Step 5 - setup segmentation
                printHeaderInConsole(`Step 3 - Segmentation stuff (${caseName} - CT slices:${totalImagesIdsCT})`)
                console.log(' - orthanDataURLS[caseNumber]: ', config.orthanDataURLS[patientIdx])
                if (searchObjRTSGT.wadoRsRoot.length > 0){
                    try{
                        await segmentationHelpers.fetchAndLoadDCMSeg(searchObjRTSGT, config.imageIdsCT, MASK_TYPE_GT)
                    } catch (error){
                        console.error(' - [loadData()] Error in fetchAndLoadDCMSeg(searchObjRTSGT, imageIdsCT): ', error);
                        updateGUIElementsHelper.showToast('Error in loading GT segmentation data', 3000);
                    }
                }
                if (searchObjRTSPred.wadoRsRoot.length > 0){
                    try{
                        await segmentationHelpers.fetchAndLoadDCMSeg(searchObjRTSPred, config.imageIdsCT, MASK_TYPE_PRED)
                        console.log(' - [loadData()] Done with fetchAndLoadDCMSeg(searchObjRTSPred, imageIdsCT)');
                    } catch (error){
                        console.error(' - [loadData()] Error in fetchAndLoadDCMSeg(searchObjRTSPred, imageIdsCT): ', error);
                        updateGUIElementsHelper.showToast('Error in loading predicted segmentation data', 3000);
                    }
                }
                try{
                    // NOTE: What is the difference between "segReprUIDs" and annotationUIDs?
                    scribbleSegmentationId = scribbleSegmentationIdBase + '::' + cornerstone3D.utilities.uuidv4();
                    let { segReprUIDs} = await segmentationHelpers.addSegmentationToState(scribbleSegmentationId, cornerstone3DTools.Enums.SegmentationRepresentations.Contour);
                    config.setScribbleSegmentationUIDs(segReprUIDs);
                } catch (error){
                    console.error(' - [loadData()] Error in addSegmentationToState(scribbleSegmentationId, cornerstone3DTools.Enums.SegmentationRepresentations.Contour): ', error);
                }

                ////////////////////////////////////////////////////////////// Step 6 - Set tools as active/passive
                // const stackScrollMouseWheelTool = cornerstone3DTools.StackScrollMouseWheelTool;
                // const toolGroup = cornerstone3DTools.ToolGroupManager.getToolGroup(toolGroupId);
                // toolGroup.setToolActive(stackScrollMouseWheelTool.toolName);
                
                ////////////////////////////////////////////////////////////// Step 99 - Done
                caseSelectionHTML.selectedIndex = patientIdx;
                updateGUIElementsHelper.unshowLoaderAnimation();
                updateGUIElementsHelper.setSliceIdxHTMLForAllHTML();
                updateGUIElementsHelper.showToast(`Data loaded successfully (CT=${totalImagesIdsCT} slices, ROIs=${totalROIsRTSTRUCT})`, 3000, true);
                [config.windowLevelButton , config.contourSegmentationToolButton, config.sculptorToolButton, config.editBaseContourViaScribbleButton].forEach((buttonHTML) => {
                    if (buttonHTML === null) return;
                    buttonHTML.disabled = false;
                });
                // await updateGUIElementsHelper.takeSnapshots(config.viewPortIdsAll)
                await updateGUIElementsHelper.takeSnapshots([config.viewportDivId])
                config.viewportIds.forEach((viewportId) => {
                    const viewportTmp = renderingEngine.getViewport(viewportId);
                    viewportTmp.setProperties({ voiRange: { upper: config.MODALITY_CT_HU_MAX, lower: config.MODALITY_CT_HU_MIN, } }, config.volumeIdCT); // TODO: does not work
                });

            }

        }else{
            updateGUIElementsHelper.showToast('No CT data available')
            await updateGUIElementsHelper.unshowLoaderAnimation()
        }
    }else{
        updateGUIElementsHelper.showToast('Default case not available. Select another case.')
        await updateGUIElementsHelper.unshowLoaderAnimation()
    }

    await updateGUIElementsHelper.unshowLoaderAnimation()
    config.setAllDivsLoaded(true, false);
}

// ******************************** MAIN ********************************

async function setup(patientIdx){

    // Step 0 - Load orthanc data
    await apiEndpointHelpers.getDataURLs();
    await setupDropDownMenu(config.orthanDataURLS, patientIdx);
    await updateGUIElementsHelper.showLoaderAnimation()
    await updateGUIElementsHelper.unshowLoaderAnimation()
    
    // -------------------------------------------------> Step 1 - Cornestone3D Init
    await cornerstoneInit();
    
    // // -------------------------------------------------> Step 2 - Do tooling stuff
    await getToolsAndToolGroup();    

    // // -------------------------------------------------> Step 3 - Make rendering engine
    setRenderingEngineAndViewports();
    
    // // -------------------------------------------------> Step 4 - Get .dcm data
    await makeGUIElementsHelper.waitForCredentials(); // For the first time
    await fetchAndLoadData(patientIdx);
    makeGUIElementsHelper.setContouringButtonsLogic();
    keyboardEventsHelper.setMouseAndKeyboardEvents();

}

// Some debug params
if (1){
    // config.setPatientIdx(8); // CHMR005 (wont follow instruction)
    // config.setPatientIdx(13); // CHMR016
    // config.setPatientIdx(5); // CHMR020 // (GTVn irritation predictions)
    // config.setPatientIdx(4);  // CHMR023 (problematic. No more)
    // config.setPatientIdx(19); // CHMR028 (many edits to make!)
    // config.setPatientIdx(21); // CHMR030 (many edits to make!)
    // config.setPatientIdx(22); // CHMR034 (no major issues)
    // config.setPatientIdx(23); // CHMR040 (1 edit to make in coronal)
    // config.setPatientIdx(0); // CHMR001-gt-filtered-gausssig2
    // config.setPatientIdx(1); // CHMR005-gt-filtered-gausssig2
    // config.setPatientIdx(3); // CHMR016-gt-filtered-gausssig2
    // config.setPatientIdx(44); // CHUP-060-gt-filtered-gausssig2-gt-deformed-warp4
    // config.setPatientIdx(2); // CHMR-004

    // ------- CHUP patients
    // config.setPatientIdx(13); // S1:CHUP-033 (dice=0.724)
    // config.setPatientIdx(7); // S2:CHUP-059 (dice=0.718)
    // config.setPatientIdx(10); // S3:CHUP-005 (dice=0.714)
    // config.setPatientIdx(11); // S4:CHUP-064 (dice=0.696)
    config.setPatientIdx(12); // S5:CHUP-028 (dice=0.690)
    // config.setPatientIdx(14); // S6:CHUP-044 (dice=0.677)

    config.setModalityContours(config.MODALITY_SEG);
}


if (process.env.NETLIFY === "true")
    config.setPatientIdx(0);
setup(config.patientIdx)



/**
TO-DO
1. [P] Make trsnsfer function for PET
2. [P] ORTHANC__DICOM_WEB__METADATA_WORKER_THREADS_COUNT = 1
3. [P] https://www.cornerstonejs.org/docs/examples/#polymorph-segmentation
    - https://github.com/cornerstonejs/cornerstone3D/issues/1351
4. Sometimes mouseUp event does not capture events
 */

/**
1. Function that resets viewports to default
 - resetView() --> self-defined
 - ??

2. showSliceIds()
 - need to connect this to an event that might be released when the scroll is done
*/

/**
 * To compile (from webpack.config.js) npx webpack --mode development --watch
 * To run: npm start
 */