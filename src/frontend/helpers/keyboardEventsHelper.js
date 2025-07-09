import * as config from './config.js';
import * as makeGUIElementsHelper from './makeGUIElementsHelper.js';
import * as updateGUIElementsHelper from './updateGUIElementsHelper.js';
import * as cornerstoneHelpers from './cornerstoneHelpers.js';
import * as annotationHelpers from './annotationHelpers.js';
import * as apiEndpointHelpers from './apiEndpointHelpers.js';
import * as segmentationHelpers from './segmentationHelpers.js';

import * as cornerstone3D from '@cornerstonejs/core';
import * as cornerstone3DTools from '@cornerstonejs/tools';
import * as cornerstoneAdapters from "@cornerstonejs/adapters";

// ******************************* Screen handling ********************************************
function getZoomLevel() {
    return Math.round((window.outerWidth / window.innerWidth) * 100);
  }

// ******************************* Contour handling ********************************************
function showUnshowAllSegmentations() {
    const toolGroupContours = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupIdContours);
    const segmentationDisplayTool = cornerstone3DTools.SegmentationDisplayTool;

    if (toolGroupContours.toolOptions[segmentationDisplayTool.toolName].mode === config.MODE_ENABLED){
        toolGroupContours.setToolDisabled(segmentationDisplayTool.toolName);
    } else {
        toolGroupContours.setToolEnabled(segmentationDisplayTool.toolName);
        if (config.userCredRole === config.USERROLE_EXPERT){
            segmentationHelpers.setSegmentationIndexOpacity(config.toolGroupIdContours, config.gtSegmentationUIDs[0], 1, 0);
        }
    }
}

// ******************************* 3D World Position Handling ********************************************
function getIndex(volume, worldPos) {

    try{
        const {imageData} = volume;
        const index = imageData.worldToIndex(worldPos);
        return index
    } catch (error){
        console.error('   -- [getIndex()] Error: ', error);
        return undefined;
    }
}

function getValue(volume, worldPos) {

    try{
        if (volume === undefined || volume === null || volume.scalarData === undefined || volume.scalarData === null || volume.dimensions === undefined || volume.dimensions === null || volume.dimensions.length !== 3 || volume.imageData === undefined || volume.imageData === null) {
            return;
        }
        const { dimensions, scalarData } = volume;

        const index = getIndex(volume, worldPos);

        index[0] = Math.floor(index[0]);
        index[1] = Math.floor(index[1]);
        index[2] = Math.floor(index[2]);

        if (!cornerstone3D.utilities.indexWithinDimensions(index, dimensions)) {
        return;
        }

        const yMultiple = dimensions[0];
        const zMultiple = dimensions[0] * dimensions[1];

        const value = scalarData[index[2] * zMultiple + index[1] * yMultiple + index[0]];

        return value;
    }catch (error){
        console.error('   -- [getValue()] Error: ', error);
        return undefined;
    }
}

// ******************************* CONTOURING TOOLS ********************************************
function getAllContouringToolsPassiveStatus(){
    
    // Step 0 - Init
    const toolGroupContours = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupIdContours);
    let allToolsStatus = false;

    // Step 1 - Brush tool
    let brushToolMode = false;
    if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
        brushToolMode = toolGroupContours.toolOptions[config.strBrushCircle].mode
    } else if (config.MODALITY_CONTOURS == config.MODALITY_RTSTRUCT){
        const planarFreeHandContourToolMode = toolGroupContours.toolOptions[cornerstone3DTools.PlanarFreehandROITool.toolName].mode;
        brushToolMode = planarFreeHandContourToolMode;
    }

    // Step 2 - Eraser tool
    let eraserToolMode = false;
    if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
        eraserToolMode = toolGroupContours.toolOptions[config.strEraserCircle].mode
    } else if (config.MODALITY_CONTOURS == config.MODALITY_RTSTRUCT){
        eraserToolMode = toolGroupContours.toolOptions[cornerstone3DTools.SculptorTool.toolName].mode;
    }

    // Step 3 - Window level tool
    let windowLevelToolMode = toolGroupContours.toolOptions[cornerstone3DTools.WindowLevelTool.toolName].mode;

    // Step 4 - AI Interactive tool
    let aiInteractiveToolMode = toolGroupContours.toolOptions[cornerstone3DTools.PlanarFreehandROITool.toolName].mode;

    allToolsStatus = brushToolMode === config.MODE_PASSIVE && eraserToolMode === config.MODE_PASSIVE && windowLevelToolMode === config.MODE_PASSIVE && aiInteractiveToolMode === config.MODE_PASSIVE;
    return allToolsStatus;
}

function getBrushOrEraserToolMode(){
    
    const toolGroupContours = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupIdContours);

    let brushToolMode = false;
    if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
        brushToolMode = toolGroupContours.toolOptions[config.strBrushCircle].mode
    } else if (config.MODALITY_CONTOURS == config.MODALITY_RTSTRUCT){
        const planarFreeHandContourToolMode = toolGroupContours.toolOptions[cornerstone3DTools.PlanarFreehandROITool.toolName].mode;
        brushToolMode = planarFreeHandContourToolMode;
    }

    // Step 2 - Eraser tool
    let eraserToolMode = false;
    if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
        eraserToolMode = toolGroupContours.toolOptions[config.strEraserCircle].mode
    } else if (config.MODALITY_CONTOURS == config.MODALITY_RTSTRUCT){
        eraserToolMode = toolGroupContours.toolOptions[cornerstone3DTools.SculptorTool.toolName].mode;
    }

    // console.log('   -- [getBrushOrEraserToolMode()] brushToolMode: ', brushToolMode, ' || eraserToolMode: ', eraserToolMode);

    if (brushToolMode === config.MODE_ACTIVE || eraserToolMode === config.MODE_ACTIVE)
        return config.MODE_ACTIVE;
    else
        return config.MODE_PASSIVE;
}


// ******************************* DIV TOOLS ********************************************
function getOtherDivs(htmlElement){

    // Step 0 - Init
    let otherHTMLElements = [];
    
    // Step 1.1 - For CT divs
    if (htmlElement.id == config.axialID){
        otherHTMLElements = [config.sagittalDiv, config.coronalDiv];
    } else if (htmlElement.id == config.sagittalID){
        otherHTMLElements = [config.axialDiv, config.coronalDiv];
    } else if (htmlElement.id == config.coronalID){
        otherHTMLElements = [config.axialDiv, config.sagittalDiv];
    }

    // Step 1.2 - For PET divs
    else if (htmlElement.id == config.axialPTID){
        otherHTMLElements = [config.sagittalDivPT, config.coronalDivPT];
    } else if (htmlElement.id == config.sagittalPTID){
        otherHTMLElements = [config.axialDivPT, config.coronalDivPT];
    } else if (htmlElement.id == config.coronalPTID){
        otherHTMLElements = [config.axialDivPT, config.sagittalDivPT];
    }

    return otherHTMLElements
}

function getCorrespondingCTPTDiv(htmlElement){
    
        // Step 0 - Init
        let correspondingDiv = undefined;
    
        // Step 1 - For CT divs
        if (htmlElement.id == config.axialID){
            correspondingDiv = config.axialDivPT;
        } else if (htmlElement.id == config.sagittalID){
            correspondingDiv = config.sagittalDivPT;
        } else if (htmlElement.id == config.coronalID){
            correspondingDiv = config.coronalDivPT;
        }

        // Step 2 - For PET divs
        else if (htmlElement.id == config.axialPTID){
            correspondingDiv = config.axialDiv;
        } else if (htmlElement.id == config.sagittalPTID){
            correspondingDiv = config.sagittalDiv;
        } else if (htmlElement.id == config.coronalPTID){
            correspondingDiv = config.coronalDiv;
        }
    
        return correspondingDiv
}

async function updateCorrespondingDivs(referenceDiv, referenceViewport){
    
    // Step 1 - Get correspondingDiv (and its viewport and viewportId)
    const correspondingDiv = getCorrespondingCTPTDiv(referenceDiv);
    const {viewport: correspondingViewport, viewportId: correspondingViewportId} = cornerstone3D.getEnabledElement(correspondingDiv);

    // Step 2 - Update correspondingDiv (its viewReference and its sliceIdxHTML)
    let correspondingViewportViewReference = correspondingViewport.getViewReference();
    correspondingViewportViewReference.sliceIndex = referenceViewport.getViewReference().sliceIndex;
    await correspondingViewport.setViewReference(correspondingViewportViewReference);
    await cornerstoneHelpers.renderNow();

    // Step 3 - Update sliceIdxHTMLForCorrespondingViewport
    const imageIdxHTMLForCorrespondingViewport = correspondingViewport.getCurrentImageIdIndex()
    const totalImagesForCorrespondingViewPort  = correspondingViewport.getNumberOfSlices()
    updateGUIElementsHelper.setSliceIdxHTMLForViewPort(correspondingViewportId, imageIdxHTMLForCorrespondingViewport, totalImagesForCorrespondingViewPort)
    updateGUIElementsHelper.setGlobalSliceIdxViewPortReferenceVars()

}

function getHoveredPointIn3D(volumeId, htmlOfElement, evt) {
    
    // Step 0 - Init
    let index3D = undefined;
    const volume = cornerstone3D.cache.getVolume(volumeId);
    if (!volume) {
        return index3D;
    }

    // Step 1 - Get viewport
    const viewportOfElement = cornerstone3D.getEnabledElement(htmlOfElement).viewport; 

    // Step 1 - Get the canvas/world and final index3D position
    const rect      = htmlOfElement.getBoundingClientRect();
    const canvasPos = [Math.floor(evt.clientX - rect.left), Math.floor(evt.clientY - rect.top)];
    const worldPos  = viewportOfElement.canvasToWorld(canvasPos);
    index3D         = getIndex(volume, worldPos);
    
    // Step 2 - Round the index3D values
    if (index3D[0] != NaN || index3D[0] != 'NaN')
        index3D = index3D.map((val) => Math.round(val));
    else
        console.log('   -- [getHoveredPointIn3D()] index3D[0] is NaN: '. rect, canvasPos, worldPos);

    // Step 3 - Set global vars
    config.setLatestMousePos({'x':evt.clientX, 'y':evt.clientY});
    config.setLatestMouseDiv(htmlOfElement);

    return index3D;
}

function getViewTypeFromDiv(htmlElement){
    let viewType = undefined;

    if (htmlElement.id == config.axialID || htmlElement.id == config.axialPTID)
        viewType = config.KEY_AXIAL;
    else if (htmlElement.id == config.sagittalID || htmlElement.id == config.sagittalPTID)
        viewType = config.KEY_SAGITTAL;
    else if (htmlElement.id == config.coronalID || htmlElement.id == config.coronalPTID)
        viewType = config.KEY_CORONAL;

    return viewType

}

let isUpdatingCamera = false;
async function syncCamera(targetElement, camera) {
    const { viewport: targetViewport } = cornerstone3D.getEnabledElement(targetElement);
    isUpdatingCamera = true;
    await targetViewport.setCamera(camera);
    await targetViewport.render();
    isUpdatingCamera = false;
}

// ******************************* MAIN FUNCTION ********************************************
function setMouseAndKeyboardEvents(){

    // Step 1 - Handle keydown events
    window.addEventListener('keydown', async function(event) {
        
        // Contour-related: For show/unshow contours
        if (event.key === config.SHORTCUT_KEY_C) {
            showUnshowAllSegmentations()
        }
        
        // Viewport-related: For reset view
        if (event.key === config.SHORTCUT_KEY_R){
            cornerstoneHelpers.resetView();
            updateGUIElementsHelper.setSliceIdxHTMLForAllHTML()
            updateGUIElementsHelper.setGlobalSliceIdxViewPortReferenceVars()
        }

        // Viewport-related: For slice traversal
        if (event.key == config.SHORTCUT_KEY_ARROW_LEFT || event.key == config.SHORTCUT_KEY_ARROW_RIGHT){

            try {

                if (config.viewPortIdsAll.includes(event.target.id)){
                    console.log('   -- [keydown] event.target.id: ', event.target.id);
                    
                    // Step 1 - Init
                    const {viewport: activeViewport, viewportId: activeViewportId} = cornerstone3D.getEnabledElement(event.target);
                    const sliceIdxHTMLForViewport = activeViewport.getCurrentImageIdIndex()
                    const totalImagesForViewPort  = activeViewport.getNumberOfSlices()
                    let viewportViewReference     = activeViewport.getViewReference()
                    
                    // Step 2 - Handle keydown event 
                    // Step 2.1 - Update sliceIdxHTMLForViewport
                    let newSliceIdxHTMLForViewport;
                    if (event.key == config.SHORTCUT_KEY_ARROW_LEFT){
                        newSliceIdxHTMLForViewport = sliceIdxHTMLForViewport - 1;
                    } else if (event.key == config.SHORTCUT_KEY_ARROW_RIGHT){
                        newSliceIdxHTMLForViewport = sliceIdxHTMLForViewport + 1;
                    }
                    if (newSliceIdxHTMLForViewport < 0) newSliceIdxHTMLForViewport = 0;
                    if (newSliceIdxHTMLForViewport > totalImagesForViewPort-1) newSliceIdxHTMLForViewport = totalImagesForViewPort - 1;
                    updateGUIElementsHelper.setSliceIdxHTMLForViewPort(activeViewportId, newSliceIdxHTMLForViewport, totalImagesForViewPort)

                    // Step 2.2 - Update the viewport itself
                    const newSliceIdxViewPortReference = updateGUIElementsHelper.convertSliceIdxHTMLToSliceIdxViewportReference(newSliceIdxHTMLForViewport, activeViewportId, totalImagesForViewPort)
                    viewportViewReference.sliceIndex = newSliceIdxViewPortReference;
                    await activeViewport.setViewReference(viewportViewReference);
                    cornerstoneHelpers.renderNow();

                    // Update sliceIdx vars
                    updateGUIElementsHelper.setGlobalSliceIdxViewPortReferenceVars()

                    // Step 2.3 - Update correspondingDiv (its viewReference and its sliceIdxHTML)
                    await updateCorrespondingDivs(event.target, activeViewport)
                }

            } catch (error){
                console.error('   -- [keydown] Error: ', error);
            }
        }

        // Tool-related: For AI interactive tool
        if (event.key === config.SHORTCUT_KEY_F){
            makeGUIElementsHelper.eventTriggerForFgdBgdCheckbox(config.fgdCheckboxId);
            config.editBaseContourViaScribbleButton.click();
        }
        
        // Tool-related: For AI interactive tool
        if (event.key === config.SHORTCUT_KEY_B){
            makeGUIElementsHelper.eventTriggerForFgdBgdCheckbox(config.bgdCheckboxId);
            config.editBaseContourViaScribbleButton.click();
        }

        // Tool-related: For disabling all tools
        if (event.key === config.SHORTCUT_KEY_ESC){

            // Step 0 - Init
            makeGUIElementsHelper.changeCursorToDefault();

            // Set all buttons to false
            [config.windowLevelButton , config.contourSegmentationToolButton, config.sculptorToolButton, config.editBaseContourViaScribbleButton].forEach((buttonHTML, buttonId) => {
                
                // Step 1 - Change HTML
                buttonHTML.checked = false;
                makeGUIElementsHelper.setButtonBoundaryColor(buttonHTML, false);

                // Step 2 - Disable cornerstone3D tools
                const toolGroupContours         = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupIdContours);
                // const toolGroupScribble         = cornerstone3DTools.ToolGroupManager.getToolGroup(toolGroupIdScribble);
                const windowLevelTool           = cornerstone3DTools.WindowLevelTool;
                const planarFreeHandContourTool = cornerstone3DTools.PlanarFreehandContourSegmentationTool;
                const sculptorTool              = cornerstone3DTools.SculptorTool;
                const planarFreehandROITool     = cornerstone3DTools.PlanarFreehandROITool;
                toolGroupContours.setToolPassive(windowLevelTool.toolName);          
                if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
                    toolGroupContours.setToolPassive(config.strBrushCircle);
                    toolGroupContours.setToolPassive(config.strEraserCircle);
                } else if (config.MODALITY_CONTOURS == config.MODALITY_RTSTRUCT){
                    toolGroupContours.setToolPassive(planarFreeHandContourTool.toolName);
                    toolGroupContours.setToolPassive(sculptorTool.toolName);
                }
                toolGroupContours.setToolPassive(planarFreehandROITool.toolName);  

                // Remove any scribbles
                const scribbleAnnotations = annotationHelpers.getAllPlanFreeHandRoiAnnotations()
                if (scribbleAnnotations.length > 0){
                    const scribbleAnnotationUID = scribbleAnnotations[scribbleAnnotations.length - 1].annotationUID;
                    annotationHelpers.handleStuffAfterProcessEndpoint(scribbleAnnotationUID);
                }
            });
        }
        
        // Tool-related: For changing brush size
        if (event.key === config.SHORTCUT_KEY_PLUS || event.key === config.SHORTCUT_KEY_EQUAL || event.key === config.SHORTCUT_KEY_MINUS){
            if (config.MODALITY_CONTOURS == config.MODALITY_SEG){
                const toolGroupContours = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupIdContours);
                if (toolGroupContours.toolOptions[config.strBrushCircle].mode === config.MODE_ACTIVE || toolGroupContours.toolOptions[config.strEraserCircle].mode === config.MODE_ACTIVE){
                    
                    // Step 0 - Init
                    makeGUIElementsHelper.changeCursorForBrushAndEraser()
                    
                    // Step 1 - Get initial brush size
                    const segUtils       = cornerstone3DTools.utilities.segmentation;
                    const toolName       = toolGroupContours.toolOptions[config.strBrushCircle].mode === config.MODE_ACTIVE ? config.strBrushCircle : config.strEraserCircle;
                    let initialBrushSize = segUtils.getBrushSizeForToolGroup(config.toolGroupIdContours, toolName);

                    // Step 2 - Update brush size
                    if (event.key === config.SHORTCUT_KEY_PLUS || event.key === config.SHORTCUT_KEY_EQUAL)
                        segUtils.setBrushSizeForToolGroup(config.toolGroupIdContours, initialBrushSize + 1);
                    else if (event.key === config.SHORTCUT_KEY_MINUS){
                        if (initialBrushSize > 1)
                            segUtils.setBrushSizeForToolGroup(config.toolGroupIdContours, initialBrushSize - 1);
                    }

                    // Step 3 - Show toast
                    let newBrushSize = segUtils.getBrushSizeForToolGroup(config.toolGroupIdContours);
                    updateGUIElementsHelper.showToast(`Brush size: ${newBrushSize}`);
                }
            }    
        }

        // Tool-related: For activate brush/eraser tool
        if (event.key === config.SHORTCUT_KEY_Q || event.key === config.SHORTCUT_KEY_W){
            if (event.key === config.SHORTCUT_KEY_Q){
                makeGUIElementsHelper.setContourSegmentationToolLogic();
            } else if (event.key === config.SHORTCUT_KEY_W){
                makeGUIElementsHelper.setSculptorToolLogic();
            }
        }

    });

    // Step 2 - Handle mouse events
    const toolGroupContours = cornerstone3DTools.ToolGroupManager.getToolGroup(config.toolGroupIdContours);
    config.viewPortDivsAll.forEach((viewportDiv, index) => {
        
        // Step 2.1 - Wheel event
        viewportDiv.addEventListener(config.MOUSE_EVENT_WHEEL, async function(evt) {

            // Step 2.1.1 - Update viewportDiv where scroll tool place
            const {viewport: activeViewport, viewportId: activeViewportId} = cornerstone3D.getEnabledElement(viewportDiv);
            const imageIdxHTMLForViewport = activeViewport.getCurrentImageIdIndex()
            const totalImagesForViewPort  = activeViewport.getNumberOfSlices()
            updateGUIElementsHelper.setSliceIdxHTMLForViewPort(activeViewportId, imageIdxHTMLForViewport, totalImagesForViewPort)
            updateGUIElementsHelper.setGlobalSliceIdxViewPortReferenceVars()
            
            // Step 2.1.2 - Update correspondingDiv (its viewReference and its sliceIdxHTML)
            await updateCorrespondingDivs(viewportDiv, activeViewport)

            // Step 2.1.3 - Append to list of scrolled slices
            const viewType = getViewTypeFromDiv(viewportDiv)
            config.updateScrolledSliceIdxsObj(viewType, Date.now(), imageIdxHTMLForViewport)
        });
        
        // Step 2.2 - Mousedown event (for AI-annotation)
        viewportDiv.addEventListener(cornerstone3DTools.Enums.Events.MOUSE_DOWN, async function(evt) {
            
            // Step 2.2.1 - Set mouseDownEpochInMs
            config.setMouseDownEpochInMs(Date.now());

        });

        // Step 2.3 - Mouseup event (for AI-annotation)
        viewportDiv.addEventListener(cornerstone3DTools.Enums.Events.MOUSE_UP, async function(evt) {
            const timebetweenMouseDownAndMouseUpInSec = (Date.now() - config.mouseDownEpochInMs)/1000.0;
            
            // Step 2.3.1 - Handle AI-annotation
            setTimeout(async () => {

                // const freehandRoiToolMode = toolGroupContours.toolOptions[planarFreehandROITool.toolName].mode;
                const freehandRoiToolMode = toolGroupContours.toolOptions[cornerstone3DTools.PlanarFreehandROITool.toolName].mode;
                if (freehandRoiToolMode === config.MODE_ACTIVE){
                    const scribbleAnnotations = annotationHelpers.getAllPlanFreeHandRoiAnnotations()
                    if (scribbleAnnotations.length > 0){
                        const scribbleAnnotationUID = scribbleAnnotations[scribbleAnnotations.length - 1].annotationUID;
                        if (scribbleAnnotations.length > 0){
                            const polyline           = scribbleAnnotations[0].data.contour.polyline;
                            const points3D = polyline.map(function(point) {
                                return getIndex(cornerstone3D.cache.getVolume(config.volumeIdCT), point);
                            });
                            // console.log(' - [setContouringButtonsLogic()] points3D: ', points3D);
                            const points3DInt = points3D.map(x => x.map(y => Math.abs(Math.round(y))));
                            await updateGUIElementsHelper.takeSnapshots([config.viewportDivId]);
                            await apiEndpointHelpers.makeRequestToProcess(points3DInt, getViewTypeFromDiv(viewportDiv), scribbleAnnotationUID, timebetweenMouseDownAndMouseUpInSec, config.mouseDownEpochInMs);
                        }
                    } else {
                        console.log(' - [setMouseAndKeyboardEvents()] scribbleAnnotations: ', scribbleAnnotations);
                        cornerstoneHelpers.renderNow();
                    }
                } else if (getAllContouringToolsPassiveStatus()) {
                    // console.log('   -- [setContouringButtonsLogic()] freehandRoiToolMode: ', freehandRoiToolMode);
                    updateGUIElementsHelper.showToast('Please enable the AI-scribble button to draw contours');
                }
            }, 100);
            
            // Step 2.3.2 - Handle brush/eraser tool
            if (getBrushOrEraserToolMode() === config.MODE_ACTIVE){
                
                // Step 1 - Get data
                const volumeCT           = cornerstone3D.cache.getVolume(config.volumeIdCT);
                const predSegUID         = config.predSegmentationUIDs[0];
                const images             = volumeCT.getCornerstoneImages();
                const segmentationVolume = cornerstone3D.cache.getVolume(config.predSegmentationId);
                const labelMapObj = cornerstoneAdapters.adaptersSEG.Cornerstone3D.Segmentation.generateLabelMaps2DFrom3D(segmentationVolume) 
                // labelMapObj.dimensions = [144,144,144], but labelMapObj.labelmaps2D only contains mask data
                // labelMapObj.numFrames = 0 [Why?]

                // Step 2 - Generate fake metadata as an example
                labelMapObj.metadata = [];
                // console.log('   -- [setMouseAndKeyboardEvents()] labelMapObj: ', labelMapObj);
                labelMapObj.segmentsOnLabelmap.forEach(segmentIndex => { // only 1 segment in this project
                    const color = cornerstone3DTools.segmentation.config.color.getColorForSegmentIndex(
                        config.toolGroupIdContours,
                        predSegUID,
                        segmentIndex
                    );
                    const segmentMetadata = segmentationHelpers.generateMockMetadata(segmentIndex, color);
                    labelMapObj.metadata[segmentIndex] = segmentMetadata;
                });

                const generatedSegmentation =
                    cornerstoneAdapters.adaptersSEG.Cornerstone3D.Segmentation.generateSegmentation(
                        images, // for the UIDs
                        labelMapObj,
                        cornerstone3D.metaData
                    );
                
                // Step 3 - Upload the segmentation
                // console.log('   -- [setMouseAndKeyboardEvents()] generatedSegmentation: ', generatedSegmentation);
                apiEndpointHelpers.uploadDICOMData(generatedSegmentation.dataset
                    , config.orthanDataURLS[config.patientIdx][config.KEY_CASE_NAME] + config.FILENAME_SUFFIX_MANUAL_REFINE
                    , timebetweenMouseDownAndMouseUpInSec, config.mouseDownEpochInMs
                );
            }
        });

        // Step 2.4 - Mousemove event
        viewportDiv.addEventListener(config.MOUSE_EVENT_MOUSEMOVE, function(evt) {
            if (config.volumeIdCT != undefined){
                const points3DInt = getHoveredPointIn3D(config.volumeIdCT, viewportDiv, evt);
                config.canvasPosHTML.innerText = `Canvas position: (${config.viewPortIdsAll[index]}) \n ==> (${points3DInt[0]}, ${points3DInt[1]}, ${points3DInt[2]})`;
                config.ctValueHTML.innerText   = `CT value: ${getValue(cornerstone3D.cache.getVolume(config.volumeIdCT), points3DInt)}`;
                if (config.volumeIdPET != undefined){
                    const volumePTThis            = cornerstone3D.cache.getVolume(config.volumeIdPET);
                    config.ptValueHTML.innerText = `PT value: ${getValue(volumePTThis, points3DInt)}`;
                }
            }

        });

        // Step 2.5 - Mousedrag event
        viewportDiv.addEventListener(cornerstone3DTools.Enums.Events.MOUSE_DRAG, function(evt) {
            
            // Step 2.5.1 - For window level tool
            const windowLevelToolMode = toolGroupContours.toolOptions[cornerstone3DTools.WindowLevelTool.toolName].mode;
            if (windowLevelToolMode === config.MODE_ACTIVE){
                const htmlElement = evt.detail.element;
                const newVoiRange = cornerstone3D.getEnabledElement(htmlElement).viewport.getProperties().voiRange;

                getOtherDivs(htmlElement).forEach((element) => {
                    cornerstone3D.getEnabledElement(element).viewport.setProperties({ voiRange: newVoiRange });
                });
            }

        });

        // Step 2.6 - Mouseclick event (for AI-annotation)
        viewportDiv.addEventListener(cornerstone3DTools.Enums.Events.MOUSE_CLICK, async function(evt) {
            const scribbleStartEpoch = Date.now();
            const freehandRoiToolMode = toolGroupContours.toolOptions[cornerstone3DTools.PlanarFreehandROITool.toolName].mode;
            if (freehandRoiToolMode === config.MODE_ACTIVE){
                const scribbleAnnotations = annotationHelpers.getAllPlanFreeHandRoiAnnotations()
                if (scribbleAnnotations.length == 0){
                    const points3DInt = getHoveredPointIn3D(config.volumeIdCT, viewportDiv, evt);
                    console.log('   -- [setMouseAndKeyboardEvents(evt=click)] points3DInt: ', points3DInt);
                    (function() {
                        (async () => {
                            const timeToScribbleInSec = 0.0;
                            await updateGUIElementsHelper.takeSnapshots([config.viewportDivId]);
                            await apiEndpointHelpers.makeRequestToProcess([points3DInt], getViewTypeFromDiv(viewportDiv), [], timeToScribbleInSec, scribbleStartEpoch);
                        })();
                    })();
                }
            }
        });

        // Step 2.7 - Camera event
        viewportDiv.addEventListener(cornerstone3D.Enums.Events.CAMERA_MODIFIED, async function(evt) {

            if (!config.allDivsLoaded || isUpdatingCamera) return;

            if (evt.detail.element === config.axialDiv) {
                await syncCamera(config.axialDivPT, evt.detail.camera);
            } else if (evt.detail.element === config.axialDivPT) {
                await syncCamera(config.axialDiv, evt.detail.camera);
            } else if (evt.detail.element === config.sagittalDiv) {
                await syncCamera(config.sagittalDivPT, evt.detail.camera);
            } else if (evt.detail.element === config.sagittalDivPT) {
                await syncCamera(config.sagittalDiv, evt.detail.camera);
            } else if (evt.detail.element === config.coronalDiv) {
                await syncCamera(config.coronalDivPT, evt.detail.camera);
            } else if (evt.detail.element === config.coronalDivPT) {
                await syncCamera(config.coronalDiv, evt.detail.camera);
            }

        });
        
    });

    // Step 3 - Handle other cornerstone events
    cornerstone3D.eventTarget.addEventListener(cornerstone3DTools.Enums.Events.ANNOTATION_ADDED, function(evt) {
        if (evt.detail.annotation.metadata.toolName == cornerstone3DTools.PlanarFreehandROITool.toolName){
            annotationHelpers.setAnnotationStyle(evt.detail.annotation.annotationUID, {lineWidth: config.SCRIBBLE_LINE_WIDTH});
        }
    });

    // Step 4 - Handle window events
    window.addEventListener('resize', function() {
        const currentZoomLevel = getZoomLevel();
        config.setCurrentZoomLevel(currentZoomLevel);
        
        if (currentZoomLevel !== config.ZOOM_LEVEL_REQUIRED) {
            // updateGUIElementsHelper.showToast(`Current Zoom: ${currentZoomLevel}% (set back to ${config.ZOOM_LEVEL_REQUIRED}%)`);
            
        }
    });

    window.addEventListener('visibilitychange', () => {
        apiEndpointHelpers.uploadScrolledSliceIdxs();
        // if (document.visibilityState === 'hidden') {
        //   console.log('tabChanged', 'Tab hidden');
        // } else if (document.visibilityState === 'visible') {
        //     console.log('tabChanged', 'Tab visible');
        // }
    });

}

export {setMouseAndKeyboardEvents};