#YOLO to Classification Dataset Converter
This repository contains a simple and robust Python script designed to process and prepare image datasets for machine learning. The script's primary function is to convert an object detection dataset into a classification-ready format by cropping and sorting the annotated objects.

#Why This Script?
When you work with object detection datasets (like those from Roboflow in YOLO format), the images often contain multiple objects. For training a classification model, you need each image file to contain only one object. This script automates that entire process, saving you from manual work.

#Key Features
*YOLO to Classification Conversion: Takes a YOLO-formatted dataset (images and .txt annotation files) and generates a new dataset of cropped images.
*Automated Cropping: Automatically reads bounding box coordinates from annotation files and crops each object from its original image.
*Smart Sorting: Organizes the newly cropped images into separate, class-specific folders (e.g., healthy_images, unhealthy_images) for easy use in classification tasks.
*Robust Error Handling: Includes built-in error handling to gracefully skip over corrupted or malformed annotation lines without crashing, ensuring the script completes its task.

#How It Works
The script iterates through your dataset's images and their corresponding annotation files. For each annotated object, it calculates the correct pixel coordinates, crops the object using the OpenCV library, and saves the result to a new folder named after its class.

#Requirements
Python 3.x
opencv-python
pyyaml

#You can install the required libraries by running:
pip install opencv-python pyyaml

#How to Use
Export Your Dataset: Export your object detection dataset from a platform like Roboflow in YOLOv5 PyTorch format.
Update the Script: Change the dataset_path variable in the script to point to the location of your downloaded dataset.

#Run the Script: Execute the script from your terminal.
python your_script_name.py

The script will automatically create healthy_images and unhealthy_images folders containing all the cropped objects, ready for your next machine learning task.
