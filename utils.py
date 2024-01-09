from fastai.vision.all import *
from torchvision import transforms
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


# Function to label the images (this might change based on your task)
def get_y(file_path): 
    return parent_label(file_path)

def create_dls(train_pct=0.8, path='data/PlantDoc'):
    # Define the DataBlock
    datablock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # Define types of input and output
        get_items=get_image_files,           # How to get the items (images)
        splitter=RandomSplitter(valid_pct=(1.0 - train_pct), seed=42),  # Split method
        get_y=get_y,                         # How to label items
        item_tfms=Resize(224),               # Item-level transformations
        # batch_tfms=aug_transforms()          # Batch-level transformations
    )
    # Create the DataLoaders
    dls = datablock.dataloaders(Path(path))
    return dls

def create_dls_stratified(train_pct=0.8, path='data/PlantDoc/'):
    # Gather all image file paths and their corresponding labels
    files = get_image_files(path)
    labels = [parent_label(f) for f in files]

    # Filter out classes with only one member
    label_counts = Counter(labels)
    valid_files = [f for f, label in zip(files, labels) if label_counts[label] > 1]
    valid_labels = [label for label in labels if label_counts[label] > 1]

    # Perform a stratified split
    train_idxs, valid_idxs = train_test_split(
        range(len(valid_files)),
        test_size=1 - train_pct,
        stratify=valid_labels
    )

    # Function to split the dataset based on the indices
    def splitter(items): 
        return train_idxs, valid_idxs

    # Create DataBlock
    plants = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=splitter,
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=aug_transforms(size=224, min_scale=0.75)
    )

    return plants.dataloaders(path)

def adapt_model_to_new_dls(learn, dls):
    # Update the learner's dataloaders
    learn.dls = dls
    # Get the number of classes in the new dataset
    num_classes_new = dls.c
    # Replace the final layer of the model to match the new number of classes
    # Assuming the final layer is named 'classifier' which is common in timm models
    # Note: The naming might vary, so ensure to adjust according to your model's architecture
    learn.model.classifier = nn.Linear(learn.model.classifier.in_features, num_classes_new)
    # Now you can evaluate the model on the new dataset
    return learn

def get_mean_table(results_table):
    # 1 & 2: Group by 'train_pct' and calculate the mean of 'acc'
    grouped = results_table.groupby('train_pct')['acc'].mean().reset_index()
    # 3: Create the new DataFrame
    mean_table = grouped.rename(columns={'acc': 'mean_acc'})
    return mean_table

def dataset_forecast(mean_table):
    # Assuming 'mean_table' is your DataFrame with 'train_pct' and 'mean_acc'
    x = mean_table['train_pct']
    y = mean_table['mean_acc']

    # Choose the degree of the polynomial (e.g., 2 for quadratic)
    degree = 1

    # Fit the polynomial regression model
    coefficients = np.polyfit(x, y, degree)

    # Use the fitted model to predict values
    polynomial = np.poly1d(coefficients)

    # Extend the range of x_line beyond the maximum x value
    x_min = min(x)
    x_max = max(x)  # Maximum value in your data
    x_extension = 2.0  # For example, extend by 20%
    extended_x_max = x_max * x_extension

    x_line = np.linspace(x_min, extended_x_max, 100)
    y_line = polynomial(x_line)

    # Plotting the original data points
    plt.scatter(x, y, color='blue', label='Data Points')

    # Plotting the fitted curve over the extended range
    plt.plot(x_line, y_line, color='red', label='Fitted Curve')

    plt.xlabel('Train Percentage')
    plt.ylabel('Mean Accuracy')
    plt.title('Polynomial Regression Fit')
    plt.legend()
    plt.show()
    return polynomial, plt.plot