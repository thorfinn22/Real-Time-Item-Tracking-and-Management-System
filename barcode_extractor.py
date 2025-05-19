#!/usr/bin/env python3
import os
import cv2
import numpy as np
from imutils import contours
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib

def load_training_data(train_dir):
    """
    Expects files named like <digit>_*.jpg in train_dir,
    where <digit> is a single numeric character.
    """
    images, labels = [], []
    for fname in os.listdir(train_dir):
        if not fname.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        label = int(fname.split('_')[0])       # e.g. "5_barcode1.jpg" -> 5
        img = cv2.imread(os.path.join(train_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        images.append(img)
        labels.append(label)
    return images, labels

def train_model(train_dir, model_path='barcode_knn.pkl'):
    """
    Train a KNN on HOG descriptors of single-digit crops,
    then dump the trained model to disk.
    """
    imgs, labels = load_training_data(train_dir)
    # Compute HOG for each image
    hogs = [hog(im, pixels_per_cell=(16,16), cells_per_block=(2,2)) for im in imgs]
    X_train, X_test, y_train, y_test = train_test_split(hogs, labels, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    print("Training accuracy:", clf.score(X_test, y_test))
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

def extract_barcode_from_image(img, model_path='barcode_knn.pkl'):
    """
    Given a greyscale image containing a barcode region,
    splits it into 12 digit crops, computes HOG on each,
    and predicts each digit with the trained KNN.
    Returns the 12-digit integer (or raises on failure).
    """
    # 1) Load model
    clf = joblib.load(model_path)

    # 2) Pre-process: binarize & find largest connected component
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No contours found for barcode extraction")
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    barcode_roi = img[y:y+h, x:x+w]

    # 3) Assume fixed 12-digit layout: slice ROI into 12 vertical bins
    digit_width = w // 12
    digits = []
    for i in range(12):
        d = barcode_roi[:, i*digit_width:(i+1)*digit_width]
        # resize to a fixed size for HOG
        d = cv2.resize(d, (128,128))
        feat = hog(d, pixels_per_cell=(16,16), cells_per_block=(2,2))
        pred = clf.predict([feat])[0]
        digits.append(str(int(pred)))

    return int(''.join(digits))
