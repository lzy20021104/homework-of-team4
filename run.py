import torch
from violence_class import ViolenceClass  # 确保导入了你的 ViolenceClass

# Initialize the model
model_path = 'best_model.pth'  # The path to your trained model weights
violence_classifier = ViolenceClass(model_path)

# Prepare a batch of images (using dummy data here for demonstration)
test_images = torch.rand(5, 3, 224, 224)  # Suppose we have 5 images to classify

# Use the classify method to predict the classes
predictions = violence_classifier.classify(test_images)

# Print out the predicted classes
print("Predicted classes:", predictions)
