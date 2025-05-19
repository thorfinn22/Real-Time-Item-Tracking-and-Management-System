from barcode_extractor import train_model

# adjust this path to the folder containing your digit sample images
TRAIN_DIR = r"C:\Users\arnob\Documents\EasyOCR-master\digit_samples_dir"

# this will write barcode_knn.pkl in the current folder
train_model(TRAIN_DIR, model_path="barcode_knn.pkl")
