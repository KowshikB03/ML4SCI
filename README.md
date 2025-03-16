# DeepLense Project - Gravitational Lens Finding
# Common Test : Multi-class classification
Model Weights : https://drive.google.com/file/d/1nT6aQJgbQ8Lw-V2OPZk2N33vEW7wN1-I/view?usp=sharing
1. Imports and Setup:
- Libraries like PyTorch, torchvision, PIL, and matplotlib are imported to handle deep learning, image processing, and evaluation.
- The availability of a GPU is checked and the device is set accordingly.
- Dataset paths are defined for three classes: 'no', 'vort', and 'sphere'.
- A label mapping is created for class-to-numerical-label conversion.
2. Custom Dataset Class (LensingDataset):
- A custom dataset class is defined to load .npy image files from directories.
- The dataset class handles loading the images, applying transformations, and returning images along with their corresponding labels.
- A warning is printed if no .npy files are found in any directory.
3. Image Transformations:
- The mean and standard deviation for image normalization are computed based on the dataset.
- Several image transformations are applied to augment the dataset, such as random affine transformations, flipping, resizing, and Gaussian blur.
4. Dataset Loading:
- The training and validation datasets are loaded using the LensingDataset class.
- A 10% subset of the training data is selected to speed up experimentation, and DataLoaders are created for the full validation set and the subset of the training data.
- Label counts for each dataset are printed to check for class balance.
5. Model Definition (LensingResNet):
- A custom model is defined based on ResNet-50, with modifications to accommodate a single-channel input (for grayscale images) and three output classes.
- The model is initialized and moved to the appropriate device (GPU/CPU).
- The loss function (CrossEntropyLoss) and optimizer (Adam) are defined for training.
6. Training Loop:
- The model is trained for 25 epochs using the subset of the training data.
- During each epoch, the model processes batches of images, computes loss, and updates the model parameters using backpropagation.
- The training loss is printed after each epoch.
7. Model Weights Saving:
- The model weights are saved in JSON format by converting the PyTorch state dictionary to a JSON-compatible structure.
8. Evaluation:
- The model is evaluated on the full validation dataset after training.
- Predictions are made using the trained model, and the ROC-AUC score is computed using the roc_auc_score function.
- The evaluation prints the validation AUC-ROC score.
9. ROC Curve Plotting:
- The ROC curve and AUC score for each class are computed and plotted.
- A macro-average ROC curve is also calculated, which averages the performance across all classes.
- The plot includes individual ROC curves for each class, a macro-average curve, and a random classifier line for reference.

**Note:** *Entire dataset training took more than 1 hour for 3 epochs and required more epochs for AUC score improvement. Which made difficult to analyze and optimize the model given the limited time to submit the GSoC test for evaluation. Hence 10% of dataset was used for training with 25 epochs, which took same 1 hour and achieved optimal evaluation metric score.*

# Specific Test 2 : Lens Finding
Model weights: https://drive.google.com/file/d/1pwXNavIymE41kXsK_edQD6hKMmPocERy/view?usp=sharing
1. Setup Dataset and Model:
- Implemented a custom LensingDataset class for loading lens and non-lens images, assigning labels, and applying transformations (e.g., normalization).
- Reloaded datasets (train_lenses, train_nonlenses, test_lenses, test_nonlenses) with appropriate transformations for training and testing.

2. Handling Class Imbalance:
- Created a combined dataset and calculated class weights for handling class imbalance.
- Used a WeightedRandomSampler to ensure balanced class sampling during training.

3. Model Architecture:
- Defined a simple CNN model with two convolutional layers followed by fully connected layers, using ReLU activations and Sigmoid for binary classification.
- Defined the model's forward pass to process the input images through the convolutional and fully connected layers.

4.Training Setup:
- Initialized the CNN model, loss function (BCELoss), and optimizer (Adam with a learning rate of 0.001).
- Created a training loop to train the model for 5 epochs, updating weights using backpropagation.

5. Evaluation:
- Developed an evaluation function to compute and display the ROC curve and AUC score using the roc_curve and auc functions from sklearn.
  
6. Model Weights Saving:

- Saved the trained model's weights (state_dict) in a JSON file, converting tensors to lists for serialization.
7. Final Evaluation:

- Evaluated the trained model on the test dataset, plotting the ROC curve and printing the AUC score.
