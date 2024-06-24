import torch
import torch.nn as nn
from model import ViolenceClassifier  # 确保此处路径正确，可能需要调整为相对或绝对路径

class ViolenceClass:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = ViolenceClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def classify(self, imgs: torch.Tensor) -> list:
        """
        Classify a batch of images.

        :param imgs: A tensor of shape (n, 3, 224, 224)
        :return: A list of predicted categories (0 or 1)
        """
        imgs = imgs.to(self.device)  # Ensure the input tensor is on the correct device
        with torch.no_grad():
            outputs = self.model(imgs)
            _, preds = torch.max(outputs, 1)
        return preds.cpu().tolist()

