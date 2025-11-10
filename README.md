🩺 Face Mask Detection using CNN
📘 Project Overview

This project aims to build a deep learning model capable of detecting whether a person is wearing a mask or not using image classification techniques.
It uses the Face Mask Dataset from Kaggle and trains a Convolutional Neural Network (CNN) to classify images into two categories:

😷 With Mask

😐 Without Mask

🧾 Dataset

Source: Face Mask Dataset – by Omkar Gurav

Contents:

Train/ – Training images divided into WithMask and WithoutMask folders

Test/ – Testing images for evaluation

Around 4,000+ labeled images of people wearing and not wearing masks

To download the dataset manually:

Go to the dataset link above

Click “Download”

Extract the contents inside your project directory:

/face-mask-dataset/
    ├── Train/
    │   ├── WithMask/
    │   └── WithoutMask/
    └── Test/
        ├── WithMask/
        └── WithoutMask/


If you want to automate it (as in your notebook):

!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d omkargurav/face-mask-dataset
!unzip face-mask-dataset.zip

⚙️ Installation & Setup

Clone this repository:

git clone https://github.com/<your-username>/masked_project.git
cd masked_project


Install dependencies:

pip install -r requirements.txt


or manually install the main libraries:

pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python


Prepare your Kaggle API key:

Go to your Kaggle account settings

Click “Create New API Token” — this downloads kaggle.json

Place it in your working directory or copy it to:

~/.kaggle/kaggle.json

🧠 Model Training

The notebook (masked_project.ipynb) trains a CNN model using TensorFlow/Keras.

Typical steps:

Load and preprocess dataset (resize, normalize images)

Split data into training and validation sets

Define CNN architecture (e.g., Conv2D → MaxPooling → Dropout → Dense layers)

Compile model with:

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


Train the model:

model.fit(train_generator, epochs=20, validation_data=validation_generator)


Evaluate model performance using:

model.evaluate(test_generator)


Save model:

model.save('mask_detector_model.h5')

📊 Results

After training, you should achieve:

Accuracy: 95–98% on test images

Loss: <0.1 (depending on epochs & architecture)

You can visualize training results:

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

🚀 Running Predictions

To predict on new images:

from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model('mask_detector_model.h5')

img = cv2.imread('sample.jpg')
img = cv2.resize(img, (128,128))
img = np.expand_dims(img, axis=0) / 255.0

prediction = model.predict(img)
print("With Mask" if prediction[0][0] > 0.5 else "Without Mask")

📂 Project Structure
masked_project/
│
├── masked_project.ipynb       # Main training and testing notebook
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── face-mask-dataset/          # Dataset folder (after download)
│   ├── Train/
│   └── Test/
└── mask_detector_model.h5      # Saved model (after training)

💡 Future Improvements

Implement real-time mask detection using webcam (cv2.VideoCapture)

Deploy using Streamlit or Flask

Add more diverse datasets for better generalization

👨‍💻 Author

Vasanth Naik Vislavath
GitHub: Vasanthnaik11
