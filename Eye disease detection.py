import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, ttk
from PIL import Image, ImageTk

# Model Paths
MODEL_PATHS = {
    "Diabetic Retinopathy": r"C:\Users\Neha Reddy\Desktop\DR(final)\dr_detection_densenet_model1.h5",
    "Glaucoma": r"C:\Users\Neha Reddy\Desktop\DR(final)\cnn_resnet50.keras"
}

# Load default model (DR)
selected_model_name = "Diabetic Retinopathy"
dnetmodel = tf.keras.models.load_model(MODEL_PATHS[selected_model_name])

# Function to load selected model
def load_selected_model():
    global dnetmodel, selected_model_name
    selected_model_name = model_var.get()
    dnetmodel = tf.keras.models.load_model(MODEL_PATHS[selected_model_name])
    result_label.config(text=f"{selected_model_name} Model Loaded Successfully!")

# Function to preprocess and predict a single image
def predict_single_image(image_path, model):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    prediction = model.predict(img_array)[0][0]
    label = selected_model_name if prediction >= 0.5 else "Normal"
    return label, prediction

# Function to handle image upload and prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    # Display selected image
    img = Image.open(file_path)
    img = img.resize((250, 250))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Predict and display result
    label, confidence = predict_single_image(file_path, dnetmodel)
    result_label.config(text=f"Prediction: {label}")
    # \nConfidence: {confidence:.4f}")

# Initialize Tkinter window
root = tk.Tk()
root.title("Eye Disease Detection")
root.geometry("500x600")
root.configure(bg="#f0f0f0")

# Title Label
title_label = Label(root, text="Eye Disease Detection", font=("Arial", 16, "bold"), fg="black", bg="#f0f0f0")
title_label.pack(pady=15)

# Model Selection Dropdown
model_var = tk.StringVar()
model_var.set(selected_model_name)
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=list(MODEL_PATHS.keys()), state="readonly", font=("Arial", 12))
model_dropdown.pack(pady=10)

# Load Model Button
load_model_btn = Button(root, text="Load Model", command=load_selected_model, font=("Arial", 12), bg="deepskyblue", fg="white", padx=10, pady=5, bd=3)
load_model_btn.pack(pady=10)

# Frame for Image and Prediction
frame = Frame(root, bg="white", bd=2, relief="solid")
frame.pack(pady=10, padx=20)

# Image Display Label
img_label = Label(frame, bg="white")
img_label.pack(padx=20, pady=20)

# Prediction Result Label
result_label = Label(frame, text="Select a Model and Upload an Image", font=("Arial", 12), fg="blue", bg="white")
result_label.pack(pady=10)

# Upload Button
upload_btn = Button(root, text="Upload Image", command=upload_and_predict, font=("Arial", 14), bg="deepskyblue", fg="white", padx=15, pady=8, relief="raised", bd=3)
upload_btn.pack(pady=20)

# Footer
# footer_label = Label(root, text="Developed by Neha Reddy", font=("Arial", 10, "italic"), fg="gray", bg="#f0f0f0")
# footer_label.pack(side="bottom", pady=10)

# Run the Tkinter event loop
root.mainloop()
