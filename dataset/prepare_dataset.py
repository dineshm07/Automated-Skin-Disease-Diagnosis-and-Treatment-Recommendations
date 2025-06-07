import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
metadata_path = "./HAM10000_metadata.csv"
image_dir = "./images"
output_base_dir = "./split_data"

# Create output folders
train_dir = os.path.join(output_base_dir, "train")
val_dir = os.path.join(output_base_dir, "validation")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Load metadata
df = pd.read_csv(metadata_path)

# Only keep images that exist in the folder
available_images = set(os.listdir(image_dir))
df = df[df['image_id'].apply(lambda x: x + ".jpg" in available_images)]

# Add .jpg extension
df['filename'] = df['image_id'] + ".jpg"

# Label is 'dx' column (diagnosis)
df['label'] = df['dx']

# Split the data
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Helper function to copy images into class-based folders
def copy_images(dataframe, target_dir):
    for _, row in dataframe.iterrows():
        label_folder = os.path.join(target_dir, row['label'])
        os.makedirs(label_folder, exist_ok=True)

        src_path = os.path.join(image_dir, row['filename'])
        dst_path = os.path.join(label_folder, row['filename'])

        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)

# Copy images
print("Copying training images...")
copy_images(train_df, train_dir)

print("Copying validation images...")
copy_images(val_df, val_dir)

print("✅ Dataset preparation complete.")
