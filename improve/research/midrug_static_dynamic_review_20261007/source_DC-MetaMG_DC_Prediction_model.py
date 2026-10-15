import torch
from cbam import CBAM
import torch.nn as nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    """Dual-channel MLP with CBAM attention for drug-ncRNA association prediction.

    Two parallel feature channels (coarse-grained and fine-grained) are fused
    via a Convolutional Block Attention Module (CBAM) before independent
    branch classification. Final output is the sigmoid of the summed logits.
    """

    def __init__(self, num_in: int, num_hid1: int, num_hid2: int, num_out: int):
        super(MLP, self).__init__()
        self.cbam = CBAM(2)

        # Branch 1 (coarse-grained features)


        self.branch1_fc1  = torch.nn.Linear(num_in, num_hid1)
        self.branch1_fc2  = torch.nn.Linear(num_in, num_hid2)
        self.branch1_cls  = torch.nn.Linear(num_hid1, num_out)


        # Branch 2 (fine-grained features)
        self.branch2_fc1  = torch.nn.Linear(num_in, num_hid1)
        self.branch2_fc2  = torch.nn.Linear(num_in, num_hid2)
        self.branch2_cls  = torch.nn.Linear(num_hid1, num_out)


        self.relu    = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop    = torch.nn.Dropout(0.3)

        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.channel_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  coarse-grained feature tensor  [B, num_in]
            x1: fine-grained feature tensor    [B, num_in]

        Returns:
            Predicted association probability  [B, 1]
        """
        # Apply channel attention across the two feature channels
        x_stack = torch.stack((x, x1), dim=0)   # [2, B, num_in]
        x_stack = self.cbam(x_stack)
        x_, x_1 = x_stack[0], x_stack[1]

        # Branch 1
        x  = self.relu(self.branch1_fc1(x_))
        x  = self.drop(x)
        x  = self.branch1_cls(x)


        # Branch 2
        x1 = self.relu(self.branch2_fc1(x_1))
        x1 = self.drop(x1)
        x1 = self.branch2_cls(x1)

        out = (self.sigmoid(x) + self.sigmoid(x1)) / 2

        '''w = torch.softmax(self.channel_weight, dim=0)
        out = w[0] * torch.sigmoid(x) + w[1] * torch.sigmoid(x1)'''

        '''w = torch.einsum('ij,ij->i', self.branch1_fc2(x_),self.branch2_fc2(x_1)).unsqueeze(1)
        w = self.sigmoid(w)
        out = w * torch.sigmoid(x) + (1-w) * torch.sigmoid(x1)'''
        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
