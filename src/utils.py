import cv2
import numpy as np
import streamlit as st

def initSessionState() -> None:
    if "Processor" not in st.session_state:
        st.session_state["Processor"] = None
        
    if "currentApp" not in st.session_state:
        st.session_state["currentApp"] = None
        
    if "runButton" not in st.session_state:
        st.session_state["runButton"] = None
    
    # Face Detection States
    if "inputImageFaceDetection" not in st.session_state:
        st.session_state["inputImageFaceDetection"] = None
        
    if "faceDetectionOutputJson" not in st.session_state:
        st.session_state["faceDetectionOutputJson"] = None
        
    if "faceDetectionOutputImage" not in st.session_state:
        st.session_state["faceDetectionOutputImage"] = None
        
    # Face Recognition States
    if "inputImageFaceRecognition1" not in st.session_state:
        st.session_state["inputImageFaceRecognition1"] = None
        
    if "inputImageFaceRecognition2" not in st.session_state:
        st.session_state["inputImageFaceRecognition2"] = None
    
    if "FaceRecognitionOutputSimilarity" not in st.session_state:
        st.session_state["FaceRecognitionOutputSimilarity"] = None
        
    if "FaceRecognitionOutputImage1" not in st.session_state:
        st.session_state["FaceRecognitionOutputImage1"] = None
        
    if "FaceRecognitionOutputImage2" not in st.session_state:
        st.session_state["FaceRecognitionOutputImage2"] = None
        
    if "FaceRecognitionOutputID1" not in st.session_state:
        st.session_state["FaceRecognitionOutputID1"] = None
        
    if "FaceRecognitionOutputID2" not in st.session_state:
        st.session_state["FaceRecognitionOutputID2"] = None
        
    # Face Analysis States
    if "inputImageFacialAnalysis" not in st.session_state:
        st.session_state["inputImageFacialAnalysis"] = None
        
    if "facialAnalysisOutputImage" not in st.session_state:
        st.session_state["facialAnalysisOutputImage"] = None 
    
    if "facialAnalysisAgeOutput" not in st.session_state:
        st.session_state["facialAnalysisAgeOutput"] = None
        
    if "facialAnalysisGenderOutput" not in st.session_state:
        st.session_state["facialAnalysisGenderOutput"] = None
        
    if "facialAnalysisGenderScoreOutput" not in st.session_state:
        st.session_state["facialAnalysisGenderScoreOutput"] = None
        
    if "facialAnalysisEmotionOutput" not in st.session_state:
        st.session_state["facialAnalysisEmotionOutput"] = None 
        
    if "facialAnalysisEmotionOutputScore" not in st.session_state:
        st.session_state["facialAnalysisEmotionOutputScore"] = None
        
    # Face Parsing States
    if "inputImageFaceParsing" not in st.session_state:
        st.session_state["inputImageFaceParsing"] = None
        
    if "faceParsingOutputAlignedImage" not in st.session_state:
        st.session_state["faceParsingOutputAlignedImage"] = None
        
    if "faceParsingOutputMaskImage" not in st.session_state:
        st.session_state["faceParsingOutputMaskImage"] = None
        
    if "faceParsingOutputVisImage" not in st.session_state:
        st.session_state["faceParsingOutputVisImage"] = None
    
    # Landmark Extraction States
    if "inputImageLandmarkExtraction" not in st.session_state:
        st.session_state["inputImageLandmarkExtraction"] = None
        
    if "landmarkExtractionOutputImage" not in st.session_state:
        st.session_state["landmarkExtractionOutputImage"] = None
        
    if "landmarkExtractionOutputJson" not in st.session_state:
        st.session_state["landmarkExtractionOutputJson"] = None  
        
    # Head Pose Estimation States
    if "inputImageHeadPoseEstimation" not in st.session_state:
        st.session_state["inputImageHeadPoseEstimation"] = None
    
    if "headPoseEstimationOutputImage" not in st.session_state:
        st.session_state["headPoseEstimationOutputImage"] = None  
        
    if "headPoseEstimationYawOutput" not in st.session_state:
        st.session_state["headPoseEstimationYawOutput"] = None  
        
    if "headPoseEstimationPitchOutput" not in st.session_state:
        st.session_state["headPoseEstimationPitchOutput"] = None  
        
    if "headPoseEstimationRollOutput" not in st.session_state:
        st.session_state["headPoseEstimationRollOutput"] = None 
        
    # ControlNet Operations States
    if "inputImageControlNetOperations" not in st.session_state:
        st.session_state["inputImageControlNetOperations"] = None 
    
    if "controlNetMethod" not in st.session_state:
        st.session_state["controlNetMethod"] = None 
        
    if "controlNetOutputImage" not in st.session_state:
        st.session_state["controlNetOutputImage"] = None 
        
def readImageFromFiles(text="Choose an image") -> np.array:
    uploadedFile = st.file_uploader(text, type=["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"])
    if uploadedFile is not None:
        byteData = uploadedFile.getvalue()
        arrayImage = cv2.imdecode(np.frombuffer(byteData, np.uint8), cv2.IMREAD_COLOR)
        return arrayImage


        
def clickRunButton() -> None:
    st.session_state.runButton = True
    
def declickRunButton() -> None:
    st.session_state.runButton = False
    
def clearSession() -> None:
    st.session_state.runButton = False
    
    # Face Detection States
    st.session_state.inputImageFaceDetection = None
    st.session_state.faceDetectionOutputJson = None
    st.session_state.faceDetectionOutputImage = None

    # Face Recognition States
    st.session_state.inputImageFaceRecognition1 = None
    st.session_state.inputImageFaceRecognition2 = None
    st.session_state.FaceRecognitionOutputSimilarity = None
    st.session_state.FaceRecognitionOutputImage1 = None
    st.session_state.FaceRecognitionOutputImage2 = None
    st.session_state.FaceRecognitionOutputID1 = None
    st.session_state.FaceRecognitionOutputID2 = None
    
    # Face Analysis States
    st.session_state.inputImageFacialAnalysis = None
    st.session_state.facialAnalysisOutputImage = None
    st.session_state.facialAnalysisAgeOutput = None
    st.session_state.facialAnalysisGenderOutput = None
    st.session_state.facialAnalysisGenderScoreOutput = None
    st.session_state.facialAnalysisEmotionOutput = None
    st.session_state.facialAnalysisEmotionOutputScore = None
    
    # Face Parsing States
    st.session_state.inputImageFaceParsing = None
    st.session_state.faceParsingOutputAlignedImage = None
    st.session_state.faceParsingOutputMaskImage = None
    st.session_state.faceParsingOutputVisImage = None
    
    # Landmark Extraction States
    st.session_state.inputImageLandmarkExtraction = None
    st.session_state.landmarkExtractionOutputImage = None
    st.session_state.landmarkExtractionOutputJson = None
    
    # Head Pose Estimation States
    st.session_state.inputImageHeadPoseEstimation = None
    st.session_state.headPoseEstimationOutputImage = None
    st.session_state.headPoseEstimationYawOutput = None
    st.session_state.headPoseEstimationPitchOutput = None
    st.session_state.headPoseEstimationRollOutput = None
    
    # ControlNet Operations States
    st.session_state.inputImageControlNetOperations = None
    st.session_state.controlNetMethod = None
    st.session_state.controlNetOutputImage = None