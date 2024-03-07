from src.utils import *
from src.Processor import Processor

#TODO add logging for each application

def main():
    
    if len(st.session_state) == 0:
        initSessionState()
    
    app = st.sidebar.selectbox(
        "Which application would you like to use?",
        (
            "🏠Home Page",
            "🔎Face Detection",
            "🎭Face Recognition",
            "📊Facial Attribute Analysis",
            "👃Face Parsing",
            "🌌Landmark Extraction",
            "📐Head Pose Estimation"
        )
    )

    if app == "🏠Home Page":
        
        st.title("OneStopVision")
        st.write("The following list are the applications you can use here. You can access the applications here from the sidebar on the left.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔎Face Detection")
            st.write("Effortlessly pinpoint human faces in any image.")
            st.subheader("🎭Face Recognition")
            st.write("Unmask the identity behind the face with lightning-fast recognition.")
            st.subheader("📊Facial Attribute Analysis")
            st.write("See beyond the surface - analyze facial features to understand gender, age, and emotional state.")
            st.subheader("👃Face Parsing")
            st.write("Dissect the face with precision, segmenting features like eyes, nose, and mouth.")
        with col2:
            st.subheader("🌌Landmark Extraction")
            st.write("Capture the intricate details of the face with 68 key landmarks.")
            st.subheader("📐Head Pose Estimation")
            st.write("Determine the precise orientation and tilt of a person's head.")
            st.subheader("🪄ControlNet Preprocessing")
            st.write("Determine the precise orientation and tilt of a person's head.")
        
    elif app == "🔎Face Detection":
        if (st.session_state.Processor is None):
            st.session_state.currentApp = app
            st.session_state.Processor = Processor()
            st.session_state.Processor.initNecessaryModels()
        elif (st.session_state.currentApp is not app):
            clearSession()
            st.session_state.currentApp = app
            st.session_state.Processor.deleteUnnecessaryModels()
            st.session_state.Processor.initNecessaryModels()
            
        st.session_state.Processor.pipelineFaceDetection()
        
    elif app == "🎭Face Recognition":
        if (st.session_state.Processor is None) and (st.session_state.currentApp is not app):
            st.session_state.currentApp = app
            st.session_state.Processor = Processor()
            st.session_state.Processor.initNecessaryModels()
        elif (st.session_state.currentApp is not app):
            clearSession()
            st.session_state.currentApp = app
            st.session_state.Processor.deleteUnnecessaryModels()
            st.session_state.Processor.initNecessaryModels()
        
        st.session_state.Processor.pipelineFaceRecognition()
            
    elif app == "📊Facial Attribute Analysis":
        if (st.session_state.Processor is None) and (st.session_state.currentApp is not app):
            st.session_state.currentApp = app
            st.session_state.Processor = Processor()
            st.session_state.Processor.initNecessaryModels()
        elif (st.session_state.currentApp is not app):
            clearSession()
            st.session_state.currentApp = app
            st.session_state.Processor.deleteUnnecessaryModels()
            st.session_state.Processor.initNecessaryModels()
        
        st.session_state.Processor.pipelineFacialAttributeAnalysis()
        
    elif app == "👃Face Parsing":
        if (st.session_state.Processor is None) and (st.session_state.currentApp is not app):
            st.session_state.currentApp = app
            st.session_state.Processor = Processor()
            st.session_state.Processor.initNecessaryModels()
        elif (st.session_state.currentApp is not app):
            clearSession()
            st.session_state.currentApp = app
            st.session_state.Processor.deleteUnnecessaryModels()
            st.session_state.Processor.initNecessaryModels()
            
        st.session_state.Processor.pipelineFaceParsing()
            
    elif app == "🌌Landmark Extraction":
        if (st.session_state.Processor is None) and (st.session_state.currentApp is not app):
            st.session_state.currentApp = app
            st.session_state.Processor = Processor()
            st.session_state.Processor.initNecessaryModels()
        elif (st.session_state.currentApp is not app):
            clearSession()
            st.session_state.currentApp = app
            st.session_state.Processor.deleteUnnecessaryModels()
            st.session_state.Processor.initNecessaryModels()
            
        st.session_state.Processor.pipelineLandmarkExtraction()
            
    elif app == "📐Head Pose Estimation":
        if (st.session_state.Processor is None) and (st.session_state.currentApp is not app):
            st.session_state.currentApp = app
            st.session_state.Processor = Processor()
            st.session_state.Processor.initNecessaryModels()
        elif (st.session_state.currentApp is not app):
            clearSession()
            st.session_state.currentApp = app
            st.session_state.Processor.deleteUnnecessaryModels()
            st.session_state.Processor.initNecessaryModels()
            
        st.session_state.Processor.pipelineHeadPoseEstimation()

    else:
        pass

main()