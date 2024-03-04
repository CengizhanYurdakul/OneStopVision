from src.utils import *
from src.Processor import Processor

#TODO add downloader for each algorithm
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
        st.header("**Applications**")
        st.write("The following list are the applications you can use here. You can access the applications here from the sidebar on the left.")
        st.subheader("🔎Face Detection")
        st.write("Describe application, input-output informations")
        st.subheader("🎭Face Recognition")
        st.write("Describe application, input-output informations")
        st.subheader("📊Facial Attribute Analysis")
        st.write("Describe application, input-output informations")
        st.subheader("👃Face Parsing")
        st.write("Describe application, input-output informations")
        st.subheader("🌌Landmark Extraction")
        st.write("Describe application, input-output informations")
        st.subheader("📐Head Pose Estimation")
        st.write("Describe application, input-output informations")
        
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