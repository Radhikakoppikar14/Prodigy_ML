import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2

# Function to load images and their labels
def load_images_and_labels(dataset_dir, categories, image_size=(64, 64)):
    images = []
    labels = []
    
    for category in categories:
        category_path = os.path.join(dataset_dir, category)
        
        print(f"[INFO] Loading images from {category_path}")
        
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            
            # Skip non-image files
            if not file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
                print(f"[INFO] Skipping non-image file {file_path}")
                continue
                
            try:
                # Load and resize image
                image = Image.open(file_path)
                image = image.resize(image_size)
                image = np.array(image)  # Convert image to array
                
                # Append image and corresponding label (0 for cats, 1 for dogs)
                images.append(image)
                labels.append(categories.index(category))
                
                # Display the first few images
                if len(images) <= 5:  # Limit to first 5 images
                    plt.imshow(image)
                    plt.title(f"Label: {category}")
                    plt.show()
                
            except Exception as e:
                print(f"[INFO] Skipping file {file_path} due to error: {e}")
                
    print(f"[INFO] Loaded {len(images)} images.")
    return np.array(images), np.array(labels)

# Function to extract features from images (flatten the images)
def extract_features(images):
    features = images.reshape(images.shape[0], -1)  # Flatten images
    return features

# Load dataset
dataset_dir = "C:/Users/koppi.RK/OneDrive/Desktop/College works/Intership/prodigy infotech/Machine learning/task3/test_set/test_set"
categories = ['cats', 'dogs']  # Folder names for categories

print("[INFO] Loading dataset...")
images, labels = load_images_and_labels(dataset_dir, categories)

# Extract features (flatten images)
features = extract_features(images)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train SVM model
print("[INFO] Training SVM model...")
svm_model = SVC(kernel='linear')  # You can experiment with other kernels like 'rbf'
svm_model.fit(X_train, y_train)

# Evaluate the model
print("[INFO] Evaluating model...")
y_pred = svm_model.predict(X_test)

# Show classification report
print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

# Save the trained SVM model
import pickle
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
    
print("[INFO] Model saved as svm_model.pkl")
