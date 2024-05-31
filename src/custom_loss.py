import torch

class CustomLoss(torch.nn.Module):
    def __init__(self, alpha=0.85, epsilon=1e-12):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return -torch.mean(self.alpha * y_true * torch.log(y_pred.clamp(min=self.epsilon)) + (1-self.alpha)*(1-y_true)*torch.log(1-y_pred.clamp(min=self.epsilon)))
    