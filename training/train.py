from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cv2

######### PARAMETERS ##########
valid_size = 0.3
test_size = 0.1
batch_size = 4
epochs = 10
cuda = False
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
###############################

# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpeg'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*.png'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = [image_path_list[i] for i in indices[:test_ind]]
test_label_path_list = [mask_path_list[i] for i in indices[:test_ind]]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = [image_path_list[i] for i in indices[test_ind:valid_ind]]
valid_label_path_list = [mask_path_list[i] for i in indices[test_ind:valid_ind]]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = [image_path_list[i] for i in indices[valid_ind:]]
train_label_path_list = [mask_path_list[i] for i in indices[valid_ind:]]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list) // batch_size

print(f"Training samples: {len(train_input_path_list)}")
print(f"Validation samples: {len(valid_input_path_list)}")
print(f"Test samples: {len(test_input_path_list)}")
print(f"Steps per epoch: {steps_per_epoch}")

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=n_classes)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()
# train.py içine bu test satırlarını koy
for path in image_path_list:
    if not os.path.exists(path):
        print("YOK:", path)
    else:
        img = cv2.imread(path)
        if img is None:
            print("OKUNAMADI:", path)

# TRAINING THE NEURAL NETWORK
print("Starting training...")
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for ind in range(steps_per_epoch):
        #########################################
        # CODE
        #########################################

        # Get batch indices
        start_ind = ind * batch_size
        end_ind = start_ind + batch_size

        # Get batch paths
        batch_input_path = train_input_path_list[start_ind:end_ind]
        batch_label_path = train_label_path_list[start_ind:end_ind]

        # Tensorize batch
        batch_input = tensorize_image(batch_input_path, input_shape, cuda=cuda)
        batch_label = tensorize_mask(batch_label_path, input_shape, n_classes, cuda=cuda)

        # Convert to proper format for loss calculation
        # batch_input: [batch_size, 3, H, W]
        # batch_label: [batch_size, 2, H, W]
        batch_label = batch_label.float()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_input)

        # Calculate loss
        loss = criterion(outputs, batch_label)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        if ind % 10 == 9:  # Print every 10 batches
            print(f'[Epoch {epoch + 1}/{epochs}, Batch {ind + 1}/{steps_per_epoch}] '
                  f'Loss: {loss.item():.4f}')

    # Calculate average training loss
    avg_train_loss = running_loss / steps_per_epoch
    train_losses.append(avg_train_loss)

    # Validation phase
    if len(valid_input_path_list) > 0:
        model.eval()
        val_loss = 0.0
        val_steps = len(valid_input_path_list) // batch_size

        with torch.no_grad():
            for val_ind in range(val_steps):
                start_ind = val_ind * batch_size
                end_ind = start_ind + batch_size

                val_batch_input_path = valid_input_path_list[start_ind:end_ind]
                val_batch_label_path = valid_label_path_list[start_ind:end_ind]

                val_batch_input = tensorize_image(val_batch_input_path, input_shape, cuda=cuda)
                val_batch_label = tensorize_mask(val_batch_label_path, input_shape, n_classes, cuda=cuda)
                val_batch_label = val_batch_label.float()

                val_outputs = model(val_batch_input)
                val_batch_loss = criterion(val_outputs, val_batch_label)
                val_loss += val_batch_loss.item()

        if val_steps > 0:
            avg_val_loss = val_loss / val_steps
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            print(f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        else:
            print(f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}')
    else:
        print(f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.4f}')

print("Training completed!")

# Save the trained model
model_save_path = os.path.join(ROOT_DIR, 'trained_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'epochs': epochs,
    'input_shape': input_shape,
    'n_classes': n_classes
}, model_save_path)

print(f"Model saved to: {model_save_path}")

# Test the model if test data exists
if len(test_input_path_list) > 0:
    print("Testing the model...")
    model.eval()
    test_loss = 0.0
    test_steps = len(test_input_path_list) // batch_size

    with torch.no_grad():
        for test_ind in range(test_steps):
            start_ind = test_ind * batch_size
            end_ind = start_ind + batch_size

            test_batch_input_path = test_input_path_list[start_ind:end_ind]
            test_batch_label_path = test_label_path_list[start_ind:end_ind]

            test_batch_input = tensorize_image(test_batch_input_path, input_shape, cuda=cuda)
            test_batch_label = tensorize_mask(test_batch_label_path, input_shape, n_classes, cuda=cuda)
            test_batch_label = test_batch_label.float()

            test_outputs = model(test_batch_input)
            test_batch_loss = criterion(test_outputs, test_batch_label)
            test_loss += test_batch_loss.item()

    if test_steps > 0:
        avg_test_loss = test_loss / test_steps
        print(f'Test Loss: {avg_test_loss:.4f}')

print("All done!")