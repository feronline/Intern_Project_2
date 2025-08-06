import torch
import torch.nn as nn
import torch.nn.functional as F


## Here, you should build the neural network. A simple model is given below as an example, you can modify the neural network architecture.

class FoInternNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # Encoder layers (feature extraction)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Decoder layers (upsampling)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upbn1 = nn.BatchNorm2d(128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.upbn2 = nn.BatchNorm2d(64)

        # Final classification layer
        self.final_conv = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)

        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """This function feeds input data into the model layers defined.
        Args:
            x : input data
        """
        #########################################
        # CODE
        #########################################

        # Encoder path
        # Block 1
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1_pooled = self.maxpool(x1)

        # Block 2
        x2 = self.conv2(x1_pooled)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2_pooled = self.maxpool(x2)

        # Block 3 (bottleneck)
        x3 = self.conv3(x2_pooled)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)

        # Decoder path
        # Upsampling block 1
        up1 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        up1 = self.upconv1(up1)
        up1 = self.upbn1(up1)
        up1 = F.relu(up1)

        # Upsampling block 2
        up2 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=True)
        up2 = self.upconv2(up2)
        up2 = self.upbn2(up2)
        up2 = F.relu(up2)

        # Final classification
        output = self.final_conv(up2)

        # Apply sigmoid for binary segmentation
        output = torch.sigmoid(output)

        return output


if __name__ == '__main__':
    # Test the model
    HEIGHT, WIDTH = 224, 224
    N_CLASS = 2

    model = FoInternNet(input_size=(HEIGHT, WIDTH), n_classes=N_CLASS)

    # Test with dummy input
    dummy_input = torch.randn(1, 3, HEIGHT, WIDTH)
    output = model(dummy_input)

    print(f"Model input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Print model architecture
    print("\nModel Architecture:")
    print(model)