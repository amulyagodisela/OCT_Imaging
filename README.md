# 🧠 OCT_Imaging

Detection of Glaucoma and Diabetic Retinopathy using Deep Learning on OCT images. This project integrates two deep learning models (developed in Google Colab) with a Python-based GUI for offline diagnosis.

## 📁 Project Structure
├── colab/
│ ├── glaucoma_model.ipynb
│ └── diabetic_retinopathy_model.ipynb
├── gui/
│ ├── Eye disease detection.py # GUI interface for model predictions
│ └── utils.py
├── models/ # Place trained model files here
├── requirements.txt
└── README.md



## 🚀 How to Run

1. **Train Models in Colab**
   - Open and run both notebooks in `colab/`.
   - Export the trained models (e.g., `.h5`, `.pt`).
   - Move them to the `models/` directory.

2. **Run the GUI Locally**
   - Install dependencies:  
     `pip install -r requirements.txt`
   - Launch the GUI:  
     `python gui/"Eye disease detection.py"`

## 📝 Notes

- Make sure model file names and paths match those expected in the GUI code.
- Keep preprocessing steps consistent between the notebooks and GUI.




