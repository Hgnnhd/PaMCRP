import torch


class OrthogonalityLosses:
    @staticmethod
    def flexible_orthogonal_loss(W, row_wise=False):
        """
        """
        if W.dim() > 2:
            W = W.view(W.size(0), -1)
        if row_wise:
            WWt = torch.matmul(W, W.t())
            identity = torch.eye(WWt.size(0), device=W.device)
            return torch.norm(WWt - identity)
        else:
            WtW = torch.matmul(W.t(), W)
            identity = torch.eye(WtW.size(0), device=W.device)
            return torch.norm(WtW - identity)

