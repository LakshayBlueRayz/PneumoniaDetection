import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

def preprocess_image(img):
    """Preprocess image for ResNet50"""
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize((224, 224))
        
        # Convert to array
        img_array = np.array(img)
        
        # Preprocess for ResNet50
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def load_model():
    """Load the transfer learning model"""
    try:
        model = tf.keras.models.load_model('transfer_pneumonia_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Pneumonia Detection System", layout="wide")
    
    st.title("Pneumonia Detection from Chest X-rays")
    st.write("Upload a chest X-ray image and the AI model will predict if pneumonia is present.")
    
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded X-ray")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            st.write("Image Information:")
            st.write(f"Format: {image.format}")
            st.write(f"Size: {image.size}")
            st.write(f"Mode: {image.mode}")
        
        try:
            model = load_model()
            if model is not None:
                processed_image = preprocess_image(image)
                if processed_image is not None:
                    # Make prediction
                    prediction = model.predict(processed_image, verbose=0)
                    
                    # Calculate probabilities
                    normal_prob = float(prediction[0][0]) * 100
                    pneumonia_prob = float(prediction[0][1]) * 100
                    
                    with col2:
                        st.subheader("Analysis Results")
                        
                        # Progress bar
                        st.progress(pneumonia_prob/100)
                        
                        # Show prediction
                        if pneumonia_prob > 60:
                            st.error(f"Pneumonia Detected ({pneumonia_prob:.1f}% confidence)")
                        elif pneumonia_prob < 40:
                            st.success(f"Normal ({normal_prob:.1f}% confidence)")
                        else:
                            st.warning("⚠️ Uncertain prediction - Please consult a medical professional")
                        
                        # Show probabilities
                        st.write("Detailed Probabilities:")
                        st.write(f"Normal: {normal_prob:.1f}%")
                        st.write(f"Pneumonia: {pneumonia_prob:.1f}%")
                        
                        # Confidence indicator
                        if abs(normal_prob - pneumonia_prob) < 20:
                            st.info("Low confidence prediction - Additional review recommended")
                        elif max(normal_prob, pneumonia_prob) > 90:
                            st.info("High confidence prediction")
                        else:
                            st.info("Moderate confidence prediction")
                            
        except Exception as e:
            st.error("Error processing image. Please try another image.")
            st.write(f"Error details: {str(e)}")
    
    with st.expander("About this System"):
        st.write("""
        This system uses a Convolutional Neural Network (CNN) based on ResNet50 for chest X-ray analysis.
        
        Important Notes:
        - This is a screening tool only and should not replace professional medical diagnosis
        - Always consult with a healthcare provider for medical advice
        - The system provides confidence levels to indicate prediction reliability
        """)

if __name__ == "__main__":
    main()