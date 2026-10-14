import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
import time
import pandas as pd
import os
import traceback
from ultralytics import YOLO

st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="wide"
)

@st.cache_resource
def load_yolo_models():
    model_paths = {
        "YOLOv8": "Model/YoloV8.pt",
        "YOLOv9": "Model/YoloV9.pt",
        "YOLOv10": "Model/YoloV10.pt",
        "YOLOv11": "Model/YoloV11.pt",
    }
    
    models = {}
    for name, path in model_paths.items():
        try:
            models[name] = YOLO(path)
        except Exception as e:
            pass
    
    return models

class_names = [
    'First Aid Station', 'Many Children', 'Many Curves First Right', 'Many Curves First Left',
    'Stop', 'No Right Turn', 'No Left Turn', 'No Stopping', 'No Entry',
    'No Overtaking', 'No Parking', 'No U-Turn', 'Church', 'Caution', 'Follow Roundabout',
    'Bicycle Lane', 'Max Speed 30 km', 'Max Speed 40 km', 'Left Lane', 'Traffic Light',
    'Load Prohibition - 10 tons', 'Mosque', 'Bus Stop', 'Pedestrian Crossing',
    'Railway Crossing Warning', 'Crossing Lane Command', '3-Way Intersection Priority',
    '3-Way Intersection Right Priority', '3-Way Intersection Left Priority', '3-Way Left Side Intersection',
    '4-Way Intersection', 'Choose One Lane', 'Speed Bump', 'Gas Station', 'U-Turn',
    'Hospital', 'Parking Area', 'Double Curve First Right', 'Double Curve First Left',
    'Right Turn'
]

@st.cache_resource
def load_ssdlite_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 41  # 40 classes + background

    model_file = "Model/SSDLite.pth"

    try:
        model = torch.load(model_file, map_location=device)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e1:
        pass

    try:
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=False, 
            width_mult=0.5  
        )
        model.num_classes = num_classes

        state_dict = torch.load(model_file, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e2:
        pass

    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)
    model.num_classes = num_classes
    model.to(device)
    model.eval()
    return model, device

def predict_yolo(model, img):
    start = time.time()
    results = model(img)
    end = time.time()
    return results, end - start

def predict_ssdlite(model, img, device):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor(),
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    start = time.time()
    try:
        with torch.no_grad():
            results = model(img_tensor)
        end = time.time()

        if isinstance(results, list):
            result = results[0]
        else:
            result = results
            
        if 'boxes' not in result or 'labels' not in result or 'scores' not in result:
            dummy_result = {
                'boxes': torch.tensor([], device=device),
                'labels': torch.tensor([], device=device),
                'scores': torch.tensor([], device=device)
            }
            return dummy_result, end - start
            
        return result, end - start
    except Exception as e:
        end = time.time()
        dummy_result = {
            'boxes': torch.tensor([], device=device),
            'labels': torch.tensor([], device=device),
            'scores': torch.tensor([], device=device)
        }
        return dummy_result, end - start
def draw_yolo_boxes(img, results, model):
    img = np.array(img)
    detected_classes = set()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y_position = y1 - 10 if y1 - 10 >= 10 else y2 + 20
            cv2.putText(img, label, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected_classes.add(model.names[cls])

    return img, detected_classes

def draw_ssdlite_boxes(img, results):
    img = np.array(img.copy())
    h, w = img.shape[:2]

    if results['boxes'].numel() == 0:
        return img, set()
    
    try:
        boxes = results['boxes'].cpu().numpy()
        labels = results['labels'].cpu().numpy()
        scores = results['scores'].cpu().numpy()

        if len(scores) > 0:
            top_idx = np.argmax(scores)
            
            box = boxes[top_idx]
            label = labels[top_idx]
            score = scores[top_idx]
            
            x1, y1, x2, y2 = box
            
            x1 = max(0, int(x1 * w / 300))
            y1 = max(0, int(y1 * h / 300))
            x2 = min(w, int(x2 * w / 300))
            y2 = min(h, int(y2 * h / 300))
            
            idx = (int(label) - 1) % len(class_names)
            class_name = class_names[idx]
            
            label_text = f"{class_name} ({int(label)}) {score:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y_position = y1 - 10 if y1 - 10 >= 10 else y2 + 20
            cv2.putText(img, label_text, (x1, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            return img, {class_name}

        return img, set()
    except Exception as e:
        return img, set()

def main():
    page = st.sidebar.selectbox("🧭 Navigation", ["Object Detection", "About"])

    if page == "About":
        st.title("About")
        st.markdown("""
    This application is an output from the **Forum Group Discussion (FGD) Assistant for Advanced Practicum Course (DGX Practicum) Gunadarma University Even Semester AY 2024/2025**.

    ---

    ### 👨‍💻 Created by:

    **Research Division:**
    - Muhamad Alif Ramadhan  
    - Nabilah Ismah Wiananda  
    - Tegar Sakti Ramadhan

    **Paper Division:**
    - Muhammad Daffa Alghifari  
    - Zahwa Annisa Hendajani

    **Competition Division:**
    - Danu Tirta  
    - Muhammad Faqih Hakim  
    - Nazwa Akilla Zahra

    **Supervisors:**
    - Prof. Dr. Detty Purnamasari, S.Kom., MMSI., M.I.Kom.  
    - Ulfa Hidayati, S.T., MMSI.  
    - Milda Safrila Oktiana, S.Kom., MMSI.  
    - Fanka Arie Reza, S.Kom.  
    - Mario Mora Siregar

    ---

    ### Application Description:

    This application is a **deep learning**-based object detection website for recognizing and classifying various types of **traffic signs** from images uploaded by users.  

    Built using the **Streamlit** framework, this application provides an interactive and user-friendly interface. Users can:
    - Upload images or use camera
    - Select from available object detection models
    - View detection results in real-time

    The detection models used apply a **one-stage detection** approach, which directly predicts object positions and classes from input images.

    ---

    ### 🔍 Models Used:

    1. **YOLOv8n**  
    2. **YOLOv9t**  
    3. **YOLOv10n**  
    4. **YOLOv11n**  
    5. **SSDLite (Single Shot MultiBox Detector)**

    Each model has different strengths in terms of **speed and accuracy**, and users are free to choose the model that suits their needs.

    > ⚠️ **Note:** Models have limitations in detecting objects at distances greater than 50 meters due to resolution limitations and small object scale.

    ---

    ### 🛑 List of Detected Classes (40 classes):

    - First Aid Station  
    - Many Children  
    - Many Curves First Right  
    - Many Curves First Left  
    - Stop  
    - No Right Turn  
    - No Left Turn  
    - No Stopping  
    - No Entry  
    - No Overtaking  
    - No Parking  
    - No U-Turn  
    - Church  
    - Caution  
    - Follow Roundabout  
    - Bicycle Lane  
    - Max Speed 30 km  
    - Max Speed 40 km  
    - Left Lane  
    - Traffic Light  
    - Load Prohibition > 10 Tons  
    - Mosque  
    - Bus Stop  
    - Pedestrian Crossing  
    - Railway Crossing Warning  
    - Crossing Lane Command  
    - 3-Way Intersection Priority  
    - 3-Way Intersection Right Priority  
    - 3-Way Intersection Left Priority  
    - 3-Way Left Side Intersection  
    - 4-Way Intersection  
    - Choose One Lane  
    - Speed Bump  
    - Gas Station  
    - U-Turn  
    - Hospital  
    - Parking Area  
    - Double Curve First Right  
    - Double Curve First Left  
    - Right Turn

    ---
    """)


    else:
        st.title("Traffic Sign Object Detection 🚦")
        
        # Model loading
        yolo_models = load_yolo_models()
        
        # Add SSDLite to model choices only if YOLO models were successfully loaded
        if yolo_models:
            try:
                ssdlite_model, device = load_ssdlite_model()
                has_ssdlite = True
            except Exception as e:
                has_ssdlite = False
        else:
            return
        
        # Model selection
        model_options = list(yolo_models.keys())
        if has_ssdlite:
            model_options.append("SSDLite")
        model_options.append("Compare All Models")
        
        model_choice = st.selectbox("Select Model 🎭", model_options)

        # Image input
        input_method = st.selectbox("Select Input Method 📸", ["Upload Image", "Use Camera"])
        image = None

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Upload Image... 📥", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption='Uploaded Image 🖼️', use_column_width=True)

        elif input_method == "Use Camera":
            camera_input = st.camera_input("Take Picture 📸")
            if camera_input:
                image = Image.open(camera_input).convert("RGB")

        # Process detection if image is available
        if image:
            if model_choice == "Compare All Models":
                result_images = {}
                times = {}
                detected_sets = {}

                # Detection with YOLO models
                for name, model in yolo_models.items():
                    try:
                        result, duration = predict_yolo(model, image)
                        img_boxed, detected = draw_yolo_boxes(image.copy(), result, model)
                        result_images[name] = img_boxed
                        times[name] = duration
                        detected_sets[name] = detected
                    except Exception as e:
                        result_images[name] = np.array(image.copy())
                        times[name] = 0.0
                        detected_sets[name] = set()
                
                # Detection with SSDLite model if available
                if has_ssdlite:
                    try:
                        result, duration = predict_ssdlite(ssdlite_model, image, device)
                        img_boxed, detected = draw_ssdlite_boxes(image, result)
                        result_images["SSDLite"] = img_boxed
                        times["SSDLite"] = duration
                        detected_sets["SSDLite"] = detected
                    except Exception as e:
                        result_images["SSDLite"] = np.array(image.copy())
                        times["SSDLite"] = 0.0
                        detected_sets["SSDLite"] = set()
                
                # Display results from all models
                st.subheader("Detection Results 🔍")
                
                # Create column layout
                cols = st.columns(len(result_images))
                
                # Display each result
                for i, (name, img) in enumerate(result_images.items()):
                    with cols[i]:
                        st.subheader(name)
                        st.image(img, caption=f"{name} Results", use_column_width=True)
                        st.write(f"⏱️ Inference Time: {times[name]:.4f} seconds")
                        st.write("🏷️ Detected Objects:")
                        if detected_sets[name]:
                            for cls in detected_sets[name]:
                                st.write(f"- {cls}")
                        else:
                            st.write("- No objects detected")
                
                # Display comparison table
                st.subheader("Model Performance Comparison 📊")
                
                model_names = list(times.keys())
                inference_times = list(times.values())
                num_objects = [len(detected) for detected in detected_sets.values()]
                
                comparison_data = {
                    "Model": model_names,
                    "Inference Time (seconds)": inference_times,
                    "Number of Objects Detected": num_objects
                }
                
                df = pd.DataFrame(comparison_data)
                st.dataframe(df, use_container_width=True)
                
                # Inference time comparison visualization
                st.subheader("Inference Time Chart 📈")
                chart_data = pd.DataFrame({
                    "Model": model_names,
                    "Time (seconds)": inference_times
                })
                st.bar_chart(chart_data.set_index("Model"))
                
            else:  # If only one model is selected
                if model_choice in yolo_models:
                    try:
                        result, duration = predict_yolo(yolo_models[model_choice], image)
                        img_boxed, detected = draw_yolo_boxes(image.copy(), result, yolo_models[model_choice])
                        
                        st.subheader("Detection Results 🔍")
                        st.image(img_boxed, caption=f"{model_choice} Detection Results", use_column_width=True)
                        
                        # Display inference time and detected objects
                        st.write(f"⏱️ Inference Time: {duration:.4f} seconds")
                        st.write("🏷️ Detected Objects:")
                        if detected:
                            for cls in detected:
                                st.write(f"- {cls}")
                        else:
                            st.write("- No objects detected")
                    except Exception as e:
                        pass
                
                elif model_choice == "SSDLite" and has_ssdlite:
                    try:
                        result, duration = predict_ssdlite(ssdlite_model, image, device)
                        img_boxed, detected = draw_ssdlite_boxes(image, result)
                        
                        st.subheader("Detection Results 🔍")
                        st.image(img_boxed, caption="SSDLite Detection Results", use_column_width=True)
                        
                        # Display inference time and detected objects
                        st.write(f"⏱️ Inference Time: {duration:.4f} seconds")
                        st.write("🏷️ Detected Objects:")
                        if detected:
                            for cls in detected:
                                st.write(f"- {cls}")
                        else:
                            st.write("- No objects detected")
                    except Exception as e:
                        pass


        # Add help and instructions
        with st.sidebar:
            st.subheader("Usage Instructions 📝")
            st.markdown("""
            1. Select the detection model you want to use
            2. Upload an image or take a photo with the camera
            3. Detection results will be displayed along with inference time information
            4. To compare models, select "Compare All Models"
            """)

        footer = """
        <style>
        .footer {
            background-color: var(--secondary-background-color);
            width: 100%;
            text-align: center;
            font-size: 14px;
            padding: 15px 0 10px 0;
            margin-top: 50px;
            border-top: 1px solid #eaeaea;
        }
        </style>
        <div class="footer">
            Gunadarma University<br>
            May 2025
        </div>
        """

        st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
