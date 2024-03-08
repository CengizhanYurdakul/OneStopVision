import io
import json
import torch
from PIL import Image

from src.utils import *

class Processor:
    def __init__(self) -> None:
        self.app = st.session_state.currentApp

        self.initVariables()
    
    def initVariables(self) -> None:
        """
        Initialises the variables of the available algorithms
        """
        self.faceDetector = None
        self.faceRecognizer = None
        self.faceAnalyzer = None
        self.faceParser = None
        self.landmarkExtractor = None
        self.headPoseEstimator = None
        self.controlNetOperator = None
        
    def initNecessaryModels(self):
        self.app = st.session_state.currentApp
        if self.app == "🔎Face Detection":
            self.initFaceDetector()
        elif self.app == "🎭Face Recognition":
            self.initFaceDetector()
            self.initFaceRecognizer()
        elif self.app == "📊Facial Attribute Analysis":
            self.initFaceDetector()
            self.initFaceAttributeAnalyzer()
        elif self.app == "👃Face Parsing":
            self.initFaceDetector()
            self.initLandmarkExtractor()
            self.initFaceParser()
        elif self.app == "🌌Landmark Extraction":
            self.initFaceDetector()
            self.initLandmarkExtractor()
        elif self.app == "📐Head Pose Estimation":
            self.initFaceDetector()
            self.initHeadPoseEstimator()
        elif self.app == "🎛️ControlNet Operations":
            self.initControlNetOperator()
        
    def deleteUnnecessaryModels(self):
        self.app = st.session_state.currentApp
        if self.app == "🔎Face Detection":
            self.deleteFaceRecognizer()
            self.deleteFaceAttributeAnalyzer()
            self.deleteFaceParser()
            self.deleteLandmarkExtractor()
            self.deleteHeadPoseEstimator()
            self.deleteControlNetOperations()
        elif self.app == "🎭Face Recognition":
            self.deleteFaceAttributeAnalyzer()
            self.deleteFaceParser()
            self.deleteLandmarkExtractor()
            self.deleteHeadPoseEstimator()
            self.deleteControlNetOperations()
        elif self.app == "📊Facial Attribute Analysis":
            self.deleteFaceRecognizer()
            self.deleteFaceParser()
            self.deleteLandmarkExtractor()
            self.deleteHeadPoseEstimator()
            self.deleteControlNetOperations()
        elif self.app == "👃Face Parsing":
            self.deleteFaceRecognizer()
            self.deleteFaceAttributeAnalyzer()
            self.deleteHeadPoseEstimator()
            self.deleteControlNetOperations()
        elif self.app == "🌌Landmark Extraction":
            self.deleteFaceRecognizer()
            self.deleteFaceAttributeAnalyzer()
            self.deleteFaceParser()
            self.deleteHeadPoseEstimator()
            self.deleteControlNetOperations()
        elif self.app == "📐Head Pose Estimation":
            self.deleteFaceRecognizer()
            self.deleteFaceAttributeAnalyzer()
            self.deleteFaceParser()
            self.deleteLandmarkExtractor()
            self.deleteControlNetOperations()
        elif self.app == "🎛️ControlNet Operations":
            self.deleteFaceDetector()
            self.deleteFaceRecognizer()
            self.deleteFaceAttributeAnalyzer()
            self.deleteFaceParser()
            self.deleteLandmarkExtractor()
            self.deleteHeadPoseEstimator()
    
    def initFaceDetector(self):
        from src.FaceDetection.FaceDetector import FaceDetector
        if self.faceDetector is None:
            self.faceDetector = FaceDetector()
            
    def deleteFaceDetector(self):
        if self.faceDetector is not None:
            del self.faceDetector
            self.faceDetector = None
            torch.cuda.empty_cache()
        
    def initFaceRecognizer(self):
        from src.FaceRecognition.FaceRecognizer import FaceRecognizer
        if self.faceRecognizer is None:
            self.faceRecognizer = FaceRecognizer()
            
    def deleteFaceRecognizer(self):
        if self.faceRecognizer is not None:
            del self.faceRecognizer
            self.faceRecognizer = None
            torch.cuda.empty_cache()
            
    def initFaceAttributeAnalyzer(self):
        from src.FacialAnalysis.FaceAnalyzer import FaceAnalyzer
        if self.faceAnalyzer is None:
            self.faceAnalyzer = FaceAnalyzer()
            
    def deleteFaceAttributeAnalyzer(self):
        if self.faceAnalyzer is not None:
            del self.faceAnalyzer
            self.faceAnalyzer = None
            torch.cuda.empty_cache()
            
    def initFaceParser(self):
        from src.FaceParsing.FaceParser import FaceParser
        if self.faceParser is None:
            self.faceParser = FaceParser()
            
    def deleteFaceParser(self):
        if self.faceParser is not None:
            del self.faceParser
            self.faceParser = None
            torch.cuda.empty_cache()
            
    def initLandmarkExtractor(self):
        from src.LandmarkExtraction.LandmarkExtractor import LandmarkExtractor
        if self.landmarkExtractor is None:
            self.landmarkExtractor = LandmarkExtractor()
            
    def deleteLandmarkExtractor(self):
        if self.landmarkExtractor is not None:
            del self.landmarkExtractor
            self.landmarkExtractor = None
            torch.cuda.empty_cache()
            
    def initHeadPoseEstimator(self):
        from src.HeadPoseEstimation.HeadPoseEstimator import HeadPoseEstimator
        if self.headPoseEstimator is None:
            self.headPoseEstimator = HeadPoseEstimator()
            
    def deleteHeadPoseEstimator(self):
        if self.headPoseEstimator is not None:
            del self.headPoseEstimator
            self.headPoseEstimator = None
            torch.cuda.empty_cache()
            
    def initControlNetOperator(self):
        from src.ControlNet.ControlNetOperator import ControlNetOperator
        if self.controlNetOperator is None:
            self.controlNetOperator = ControlNetOperator()
            
    def deleteControlNetOperations(self):
        if self.controlNetOperator is not None:
            del self.controlNetOperator
            self.controlNetOperator = None
            torch.cuda.empty_cache()
            
    def pipelineFaceDetection(self):
        st.session_state.inputImageFaceDetection = readImageFromFiles("Choose an image for input!")
        st.button("Run", on_click=clickRunButton)
        
        if st.session_state.runButton:
            if (st.session_state.inputImageFaceDetection is None):
                st.error("Input image is not selected!")
                st.session_state.runButton = False
            else:
                boxes, landmarks = self.faceDetector.detect(inputImage=st.session_state.inputImageFaceDetection)
                if (len(landmarks[0]) == 0) or (len(boxes[0]) == 0):
                    st.error("No face found in the image!")
                else:
                    st.session_state.faceDetectionOutputImage, st.session_state.faceDetectionOutputJson = self.faceDetector.parseFaceDetectionOutputs(
                        image=st.session_state.inputImageFaceDetection,
                        boxes=boxes,
                        landmarks=landmarks
                    )

        if (st.session_state.faceDetectionOutputImage is not None) and (st.session_state.faceDetectionOutputJson is not None):
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(st.session_state.faceDetectionOutputImage, cv2.COLOR_BGR2RGB), caption="Face detected image")
            with col2:
                jsonFile = json.dumps(str(st.session_state.faceDetectionOutputJson))
                st.download_button(
                                    label="Download JSON",
                                    file_name="Detection.json",
                                    mime="application/json",
                                    data=jsonFile,
                                )
                st.json(st.session_state.faceDetectionOutputJson)
                
    def pipelineFaceRecognition(self):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.inputImageFaceRecognition1 = readImageFromFiles("Choose an image for input 1!")
        with col2:
            st.session_state.inputImageFaceRecognition2 = readImageFromFiles("Choose an image for input 2!")
        
        st.button("Run", on_click=clickRunButton)
        
        if st.session_state.runButton:
            if (st.session_state.inputImageFaceRecognition1 is None) or (st.session_state.inputImageFaceRecognition2 is None):
                st.error("Input image is not selected!")
                st.session_state.runButton = False
            else:
                boxes1, landmarks1 = self.faceDetector.detect(inputImage=st.session_state.inputImageFaceRecognition1)
                boxes2, landmarks2 = self.faceDetector.detect(inputImage=st.session_state.inputImageFaceRecognition2)
                if (len(landmarks1[0]) == 0) or (len(boxes1[0]) == 0):
                    st.error("No face found in the first image!")
                elif (len(landmarks2[0]) == 0) or (len(boxes2[0]) == 0):
                    st.error("No face found in the second image!")
                else:
                    st.session_state.FaceRecognitionOutputSimilarity, st.session_state.FaceRecognitionOutputImage1, st.session_state.FaceRecognitionOutputImage2, st.session_state.FaceRecognitionOutputID1, st.session_state.FaceRecognitionOutputID2 = self.faceRecognizer.compare(
                        image1=[st.session_state.inputImageFaceRecognition1, boxes1, landmarks1],
                        image2=[st.session_state.inputImageFaceRecognition2, boxes2, landmarks2]
                    )
                
        if (st.session_state.FaceRecognitionOutputSimilarity is not None) and (st.session_state.FaceRecognitionOutputImage1 is not None) and (st.session_state.FaceRecognitionOutputImage2 is not None):
            st.divider()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(cv2.cvtColor(st.session_state.FaceRecognitionOutputImage1, cv2.COLOR_BGR2RGB), caption="Aligned image 1")
                @st.cache_data
                def loadID():
                    return st.session_state.FaceRecognitionOutputID1
                
                identity1 = loadID()
                with io.BytesIO() as buffer:
                    np.save(buffer, identity1)
                    downloadIDButton1 = st.download_button(
                        label="Download ID (.npy)",
                        data = buffer,
                        file_name = 'identity1.npy'
                    ) 
            with col2:
                st.image(cv2.cvtColor(st.session_state.FaceRecognitionOutputImage2, cv2.COLOR_BGR2RGB), caption="Aligned image 2")
                @st.cache_data
                def loadID():
                    return st.session_state.FaceRecognitionOutputID2
                
                identity2 = loadID()
                with io.BytesIO() as buffer:
                    np.save(buffer, identity2)
                    downloadIDButton2 = st.download_button(
                        label="Download ID (.npy)",
                        data = buffer,
                        file_name = 'identity2.npy'
                    ) 
            with col3:
                st.text("Cosine Similarity: %s" % round(st.session_state.FaceRecognitionOutputSimilarity.item(), 3))
                
    def pipelineFacialAttributeAnalysis(self):
        st.session_state.inputImageFacialAnalysis = readImageFromFiles("Choose an image for input!")
        st.button("Run", on_click=clickRunButton)
        
        if st.session_state.runButton:
            if (st.session_state.inputImageFacialAnalysis is None):
                st.error("Input image is not selected!")
                st.session_state.runButton = False
            else:
                boxes, landmarks = self.faceDetector.detect(inputImage=st.session_state.inputImageFacialAnalysis)
                if (len(landmarks[0]) == 0) or (len(boxes[0]) == 0):
                    st.error("No face found in the image!")
                else:
                    st.session_state.facialAnalysisOutputImage, _ = self.faceDetector.parseFaceDetectionOutputs(
                        image=st.session_state.inputImageFacialAnalysis,
                        boxes=boxes,
                        landmarks=landmarks
                    )
                    st.session_state.facialAnalysisGenderOutput, st.session_state.facialAnalysisGenderScoreOutput, st.session_state.facialAnalysisAgeOutput, st.session_state.facialAnalysisEmotionOutput, st.session_state.facialAnalysisEmotionOutputScore = self.faceAnalyzer.main(inputImage=st.session_state.inputImageFacialAnalysis, boxes=boxes)
        
        if (st.session_state.facialAnalysisGenderOutput is not None) and (st.session_state.facialAnalysisGenderScoreOutput is not None) and (st.session_state.facialAnalysisAgeOutput is not None) and (st.session_state.facialAnalysisEmotionOutput is not None) and (st.session_state.facialAnalysisEmotionOutputScore is not None):
            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                st.image(cv2.cvtColor(st.session_state.facialAnalysisOutputImage, cv2.COLOR_BGR2RGB), caption="Facial analysis image")
            
            with col2:
                st.text("Gender: %s" % st.session_state.facialAnalysisGenderOutput)
                st.text("Gender Probability: %s" % st.session_state.facialAnalysisGenderScoreOutput)
                st.text("Age: %s" % st.session_state.facialAnalysisAgeOutput)
                st.text("Emotion: %s" % st.session_state.facialAnalysisEmotionOutput)
                st.text("Emotion Probability: %s" % st.session_state.facialAnalysisEmotionOutputScore)
    
    def pipelineFaceParsing(self):
        st.session_state.inputImageFaceParsing = readImageFromFiles("Choose an image for input!")
        st.button("Run", on_click=clickRunButton)
        
        if st.session_state.runButton:
            if (st.session_state.inputImageFaceParsing is None):
                st.error("Input image is not selected!")
                st.session_state.runButton = False
            else:
                boxes, landmarks = self.faceDetector.detect(inputImage=st.session_state.inputImageFaceParsing)
                if (len(landmarks[0]) == 0) or (len(boxes[0]) == 0):
                    st.error("No face found in the image!")
                else:
                    landmark68 = self.landmarkExtractor.main(st.session_state.inputImageFaceParsing, boxes)
                    st.session_state.faceParsingOutputAlignedImage, st.session_state.faceParsingOutputMaskImage, st.session_state.faceParsingOutputVisImage = self.faceParser.extractMask(st.session_state.inputImageFaceParsing, landmark68)

        if (st.session_state.faceParsingOutputAlignedImage is not None) and (st.session_state.faceParsingOutputMaskImage is not None) and (st.session_state.faceParsingOutputVisImage is not None):
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(st.session_state.faceParsingOutputAlignedImage, caption="Face parsing aligned image")

                im = Image.fromarray(st.session_state.faceParsingOutputAlignedImage)
                buffer = io.BytesIO()
                im.save(buffer, format="PNG")
                
                downloadImageButton1 = st.download_button(
                    label="Download image",
                    data = buffer,
                    file_name="alignedimage.png",
                    mime="image/png"
                ) 
            
            with col2:
                st.image(cv2.cvtColor(st.session_state.faceParsingOutputMaskImage, cv2.COLOR_BGR2RGB), caption="Face parsing mask image")

                im = Image.fromarray(cv2.cvtColor(st.session_state.faceParsingOutputMaskImage, cv2.COLOR_BGR2RGB))
                buffer = io.BytesIO()
                im.save(buffer, format="PNG")
                
                downloadImageButton2 = st.download_button(
                    label="Download image",
                    data = buffer,
                    file_name="maskimage.png",
                    mime="image/png"
                ) 
                
            with col3:
                st.image(cv2.cvtColor(st.session_state.faceParsingOutputVisImage, cv2.COLOR_BGR2RGB), caption="Face parsing visualize image")
                
                im = Image.fromarray(cv2.cvtColor(st.session_state.faceParsingOutputVisImage, cv2.COLOR_BGR2RGB))
                buffer = io.BytesIO()
                im.save(buffer, format="PNG")
                
                downloadImageButton3 = st.download_button(
                    label="Download image",
                    data = buffer,
                    file_name="visimage.png",
                    mime="image/png"
                ) 
        
    def pipelineLandmarkExtraction(self):
        st.session_state.inputImageLandmarkExtraction = readImageFromFiles("Choose an image for input!")
        st.button("Run", on_click=clickRunButton)
        
        if st.session_state.runButton:
            if (st.session_state.inputImageLandmarkExtraction is None):
                st.error("Input image is not selected!")
                st.session_state.runButton = False
            else:
                boxes, landmarks = self.faceDetector.detect(inputImage=st.session_state.inputImageLandmarkExtraction)
                if (len(landmarks[0]) == 0) or (len(boxes[0]) == 0):
                    st.error("No face found in the image!")
                else:
                    landmark68 = self.landmarkExtractor.main(st.session_state.inputImageLandmarkExtraction, boxes)
                    st.session_state.landmarkExtractionOutputImage, st.session_state.landmarkExtractionOutputJson = self.landmarkExtractor.parseLandmarkExtractionOutputs(st.session_state.inputImageLandmarkExtraction, landmark68)
       
        if (st.session_state.landmarkExtractionOutputImage is not None) and (st.session_state.landmarkExtractionOutputJson is not None):
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(st.session_state.landmarkExtractionOutputImage, cv2.COLOR_BGR2RGB), caption="Facial landmarks image")
            
            with col2:
                jsonFile = json.dumps(str(st.session_state.landmarkExtractionOutputJson))
                st.download_button(
                                    label="Download JSON",
                                    file_name="Landmark68.json",
                                    mime="application/json",
                                    data=jsonFile,
                                )
                st.json(st.session_state.landmarkExtractionOutputJson)
                
    def pipelineHeadPoseEstimation(self):
        st.session_state.inputImageHeadPoseEstimation = readImageFromFiles("Choose an image for input!")
        st.button("Run", on_click=clickRunButton)
        
        if st.session_state.runButton:
            if (st.session_state.inputImageHeadPoseEstimation is None):
                st.error("Input image is not selected!")
                st.session_state.runButton = False
            else:
                boxes, landmarks = self.faceDetector.detect(inputImage=st.session_state.inputImageHeadPoseEstimation)
                if (len(landmarks[0]) == 0) or (len(boxes[0]) == 0):
                    st.error("No face found in the image!")
                else:
                    st.session_state.headPoseEstimationOutputImage, st.session_state.headPoseEstimationYawOutput, st.session_state.headPoseEstimationPitchOutput, st.session_state.headPoseEstimationRollOutput = self.headPoseEstimator.main(st.session_state.inputImageHeadPoseEstimation, boxes)
        
        if (st.session_state.headPoseEstimationOutputImage is not None) and (st.session_state.headPoseEstimationYawOutput is not None) and (st.session_state.headPoseEstimationPitchOutput is not None) and (st.session_state.headPoseEstimationRollOutput is not None):        
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(cv2.cvtColor(st.session_state.headPoseEstimationOutputImage, cv2.COLOR_BGR2RGB), caption="Head pose estimation image")
            
            with col2:
                st.text("Yaw: %s" % st.session_state.headPoseEstimationYawOutput)
                st.text("Pitch: %s" % st.session_state.headPoseEstimationPitchOutput)
                st.text("Roll: %s" % st.session_state.headPoseEstimationRollOutput)
                
    def pipelineControlNetOperations(self):
        st.session_state.inputImageControlNetOperations = readImageFromFiles("Choose an image for input!")
        
        st.session_state.controlNetMethod = st.radio(
            "Choose ControlNet method",
            ["Canny", "M-LSD", "HED", "OpenPose", "Depth", "Semantic Segmentation"]
        )
        
        st.button("Run", on_click=clickRunButton)
        
        if st.session_state.runButton:
            if (st.session_state.inputImageControlNetOperations is None):
                st.error("Input image is not selected!")
                st.session_state.runButton = False
            else:
                if st.session_state.controlNetMethod == "Canny":
                    st.session_state.controlNetOutputImage = self.controlNetOperator.mainCanny(st.session_state.inputImageControlNetOperations)
                elif st.session_state.controlNetMethod == "M-LSD":
                    st.session_state.controlNetOutputImage = self.controlNetOperator.mainMLSD(st.session_state.inputImageControlNetOperations)
                elif st.session_state.controlNetMethod == "HED":
                    st.session_state.controlNetOutputImage = self.controlNetOperator.mainHED(st.session_state.inputImageControlNetOperations)
                elif st.session_state.controlNetMethod == "OpenPose":
                    st.session_state.controlNetOutputImage = self.controlNetOperator.mainOpenPose(st.session_state.inputImageControlNetOperations)
                elif st.session_state.controlNetMethod == "Depth":
                    st.session_state.controlNetOutputImage = self.controlNetOperator.mainDepth(st.session_state.inputImageControlNetOperations)
                elif st.session_state.controlNetMethod == "Semantic Segmentation":
                    st.session_state.controlNetOutputImage = self.controlNetOperator.mainSemanticSegmentation(st.session_state.inputImageControlNetOperations)

            if (st.session_state.controlNetOutputImage is not None) and (st.session_state.inputImageControlNetOperations is not None):
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(cv2.cvtColor(st.session_state.inputImageControlNetOperations, cv2.COLOR_BGR2RGB), caption="ControlNet input image")
                
                with col2:
                    st.image(cv2.cvtColor(st.session_state.controlNetOutputImage, cv2.COLOR_BGR2RGB), caption="ControlNet output image")
            
            declickRunButton()