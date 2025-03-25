# DeepLense Project 
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

# Specific Test 4 : Diffusion Models 

1. **Custom SSIM Loss Function:** Implemented a custom Structural Similarity (SSIM) loss function to improve image quality in training. It applies a Gaussian kernel to compute mean, variance, and covariance, returning 1−SSIM as the loss.

2. **UNet with Timestep Embedding:** Designed a UNet-based encoder-decoder architecture with timestep embedding for time-dependent processing. The encoder extracts hierarchical features, and the decoder reconstructs images while incorporating timestep information.

3. **DDPM Implementation:** Developed a Denoising Diffusion Probabilistic Model (DDPM) using UNet as the noise prediction network. The model adds and removes noise progressively, using both MSE and SSIM losses to enhance generation quality.

4. **Custom Lensing Dataset Loader:** Created a PyTorch dataset class to load, normalize, and preprocess gravitational lensing images from .npy files. Images are resized, converted to tensors, and normalized for consistent model input.

5. **Training Pipeline:** Designed a training loop that leverages the Adam optimizer and ReduceLROnPlateau scheduler. The model trains on batches from the Lensing dataset, with dynamic timestep selection and loss computation for optimization.

6. **Efficient Dataset Sampling:** To accelerate training, selected only 0.5% of the training set and performed a 90-10 train-validation split. Data is loaded in batches with shuffling for efficient gradient updates.

7. **Progressive Training and Optimization:** Trained the DDPM for 25 epochs, monitoring loss reduction from 1.0042 to 0.6779. Used learning rate scheduling and gradient accumulation to refine training dynamics.

8. **Model Saving and Future Work:** Saved trained model weights in JSON format for easy reuse. Future improvements include increasing dataset size, extending training epochs, and monitoring validation loss for better generalization.

# Specific Test 5 : Physics-Guided ML

1. **Essential Libraries & Setup:** Imported deep learning libraries (PyTorch, torchvision), data handling tools (numpy, PIL, os, glob), and evaluation metrics (sklearn). Used torchvision.transforms for preprocessing and handled datasets with Dataset, DataLoader, and Subset. Checked for GPU availability and set the device accordingly.

2. **Dataset Management (LensingDataset):** Created a custom dataset class to load .npy images and their labels from three classes ("no", "vort", "sphere"). Ensured consistent image shape, applied transformations, and normalized data. Computed dataset statistics (mean, std) for standardization.

3. **Data Preprocessing & Augmentation:** Converted numpy arrays to PIL images and applied random augmentations: affine transformations, flipping, resizing (128×128), and Gaussian blur. Normalized images using computed mean and standard deviation.

4. **Dataset Splitting & DataLoaders:** Created training and validation datasets, selecting a 10% subset of the training set for efficiency. Initialized DataLoader with batch size 32, shuffling for training and non-shuffling for validation. Analyzed class distributions to ensure balance.

5. **Physics-Informed ResNet Model (LensingResNet):** Used ResNet-50 as a backbone with pretrained ImageNet weights. Modified the first convolutional layer to accept grayscale images and replaced the fully connected (FC) layer with two heads:

- **Classification Head:** Predicts class labels using a fully connected network (2048 → 512 → 3) with ReLU and Dropout.

- **Physics-Informed Head:** Predicts deflection angles using a fully connected layer (2048 → 512 → 2) with tanh activation.

6. **Custom Physics-Based Loss Function (lensing_loss):** Implemented a physics-inspired loss using the lens equation. Computed predicted source position (β) and minimized the error with Smooth L1 Loss.

7. **Training Strategy (25 Epochs):** Used CrossEntropyLoss for classification and Adam optimizer (lr = 0.0001). Dynamically adjusted the physics loss weight using a sigmoid-based schedule. Trained the model on mini-batches, progressively reducing loss from 1.1215 → 0.6228, showing convergence.

8. **Evaluation & Validation:** Set model to eval() mode and used torch.no_grad() for efficient inference. Computed classification and physics losses, aggregated total validation loss, and measured performance. Final validation loss: 0.8567, indicating generalization ability.









