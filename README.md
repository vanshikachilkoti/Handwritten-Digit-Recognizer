# **Handwritten Digit Recognition**  

## **Overview**  
This project implements a **Handwritten Digit Recognition System** using a **CNN** model trained on the **MNIST dataset**. The model can classify handwritten digits (0-9) from user input via a drawing canvas or an uploaded image.  

## **Features**  
- **Draw a digit** on the web-based canvas for real-time recognition.  
- **Upload an image** file containing a digit for prediction.  
- Uses a **Convolutional Neural Network (CNN)** trained on the MNIST dataset.  
- **Simple and interactive UI** powered by HTML, JavaScript, and Flask backend.  

## **Project Structure**  
```
├── static/                  # Contains frontend assets (CSS, JavaScript)
├── templates/               # HTML files for UI
│   ├── index.html           # Main UI for digit recognition
├── images/
│   ├── Screenshot.jpg   
├── model_training.py        # Script to train and save the CNN model
├── app.py                   # Flask backend for processing user input
├── handwritten_digit_recognition_model.h5  # Trained model file
└── README.md                # Project documentation
```

## **Installation & Setup**  

### **1 Install Dependencies**  
Ensure you have Python installed. Then, install the required libraries:  
```bash
pip install tensorflow flask numpy pillow
```

### **2 Train the Model**  
If you want to train the model from scratch, run:  
```bash
python model_training.py
```

### **3 Run the Web App**  
```bash
python app.py
```
Now, open **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

## **Model Architecture**  
The CNN model consists of:  
- **3 Convolutional layers** (`Conv2D`)  
- **2 MaxPooling layers**  
- **1 Fully Connected layer**  
- **Dropout to prevent overfitting**  
- **Adam optimizer** with categorical crossentropy loss  

## **Sample Predictions**  
![Handwritten Digit Recognition](https://raw.githubusercontent.com/vanshikachilkoti/Handwritten-Digit-Recognizer/main/images/Screenshot.jpg)

## **Future Enhancements**  
🔹 **Improve model accuracy** using more advanced architectures like ResNet.  
🔹 **Support multi-digit recognition.**  
🔹 **Deploy the model as a cloud-based API.**  
