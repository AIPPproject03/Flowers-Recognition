# 🌸 Flower Recognition AI Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-red)

## 📝 Project Overview

This project uses deep learning and transfer learning techniques to classify 5 different types of flowers through image recognition. Built using TensorFlow and DenseNet121 architecture, this AI model has been trained to recognize:

- 🌼 Daisy
- 🌞 Dandelion
- 🌹 Rose
- 🌻 Sunflower
- 🌷 Tulip

## 🔴 Live Demo

Try the application live: [Flower Recognition App](https://recognizeflowers.streamlit.app/)

## 📊 Model Architecture & Results

The model uses transfer learning with a DenseNet121 base architecture that was pre-trained on ImageNet. Key features:

1. **Architecture**: DenseNet121 with custom classifier head
2. **Transfer Learning Approach**:
   - Initial Feature Extraction (frozen base model)
   - Fine-tuning of the last 20 layers
3. **Data Augmentation**: Random flips, rotations, and zoom for better generalization

### Training Results

```
✅ Model selesai dilatih selama 4 epoch
```

The model was trained using early stopping to prevent overfitting, which resulted in the training process completing after 4 epochs when validation metrics stopped improving.

### Classification Report

|                  | Precision | Recall | F1-Score | Support  |
| ---------------- | --------- | ------ | -------- | -------- |
| daisy            | 0.83      | 0.86   | 0.84     | 242      |
| dandelion        | 0.84      | 0.88   | 0.86     | 319      |
| rose             | 0.80      | 0.81   | 0.81     | 231      |
| sunflower        | 0.83      | 0.88   | 0.85     | 204      |
| tulip            | 0.87      | 0.75   | 0.81     | 300      |
| **Accuracy**     |           |        | **0.83** | **1296** |
| **Macro Avg**    | 0.83      | 0.84   | 0.83     | 1296     |
| **Weighted Avg** | 0.84      | 0.83   | 0.83     | 1296     |

## 📸 Example Predictions

Here are some example predictions from the model:

| Image                                                                                                                     | Prediction | Confidence |
| ------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------- |
| ![Daisy](https://github.com/AIPPproject03/Flowers-Recognition/blob/main/flowers/daisy/10140303196_b88d3d6cec.jpg)         | Daisy      | 98.7%      |
| ![Sunflower](https://github.com/AIPPproject03/Flowers-Recognition/blob/main/flowers/sunflower/20342824594_9740b7b160.jpg) | Sunflower  | 99.2%      |
| ![Rose](https://github.com/AIPPproject03/Flowers-Recognition/blob/main/flowers/rose/2065522422_cfdd80044a_n.jpg)          | Rose       | 96.5%      |

## 🛠️ Project Structure

```
Flowers Recognition/
├── app.py               # Streamlit web application
├── main.ipynb           # Model training notebook
├── requirements.txt     # Dependencies
├── README.MD            # Project documentation
├── flowers/             # Dataset directory
│   ├── daisy/
│   ├── dandelion/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
└── models/              # Saved model files
    └── flower_classifier_densenet.h5
```

## 💻 Installation & Setup

1. Clone this repository:

   ```bash
   git clone <repository-url>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## 📚 Model Training Process

The model was trained in two stages:

1. **Feature Extraction**: The base DenseNet121 model was frozen, and only the classifier head was trained.
2. **Fine-tuning**: The last 20 layers of the base model were unfrozen and trained with a very small learning rate (1e-5).

Data augmentation techniques were applied to improve model generalization:

- Random horizontal flips
- Random rotations (up to 10%)
- Random zoom (up to 10%)

## 🔄 Usage

### Web Application

The project includes a Streamlit web application that allows users to:

1. Upload custom flower images for classification
2. Select sample images to test the model
3. View prediction probabilities across all flower classes

### For Developers

```python
# Load the model
model = tf.keras.models.load_model("models/flower_classifier_densenet.h5")

# Preprocess an image for prediction
def process_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Make a prediction
img_array = process_image("path/to/flower/image.jpg")
predictions = model.predict(img_array)
class_idx = np.argmax(predictions)
confidence = np.max(predictions)
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Dataset based on the Flowers Recognition dataset
- TensorFlow and Keras for the deep learning framework
- Streamlit for the web application framework
