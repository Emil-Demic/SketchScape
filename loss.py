import torch
import torch.nn.functional as F
from torch import nn


def _validate_inputs(image_feats: torch.Tensor, sketch_feats: torch.Tensor):
    # Check input dimensionality.
    if sketch_feats.dim() != 2:
        raise ValueError('<sketch_feats> must have 2 dimensions.')
    if image_feats.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    # Check if the number of samples matches.
    if len(sketch_feats) != len(image_feats):
        raise ValueError('<sketch_feats> and <positive_key> must must have the same number of samples.')
    # Embedding vectors should have the same number of components.
    if sketch_feats.shape[-1] != image_feats.shape[-1]:
        raise ValueError('Vectors of <sketch_feats> and <positive_key> should have the same number of components.')


class InfoNCE(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sketch_feats: torch.Tensor, image_feats: torch.Tensor):
        return info_nce(sketch_feats, image_feats, temperature=self.temperature)


def info_nce(sketch_feats: torch.Tensor, image_feats: torch.Tensor, temperature: float = 0.1):
    """
    author = Robin Elbers
    home-page = https://github.com/RElbers/info-nce-pytorch

    Computes InfoNCE loss between sketch and image embeddings.

    Inputs:
        sketch_feats: [B, D] tensor of sketch embeddings (from sketch encoder)
        image_feats: [B, D] tensor of image embeddings (from image encoder)
        temperature: float, temperature for softmax similarity

    Output:
        Scalar loss (torch.Tensor)
    """
    _validate_inputs(image_feats, sketch_feats)

    # Normalize to unit vectors
    sketch_feats = F.normalize(sketch_feats, dim=-1)
    image_feats = F.normalize(image_feats, dim=-1)

    # Negative keys are implicitly off-diagonal positive keys.
    # Cosine between all combinations
    logits = sketch_feats @ image_feats.transpose(-2, -1)

    # Positive keys are the entries on the diagonal
    labels = torch.arange(len(sketch_feats), device=sketch_feats.device)

    return F.cross_entropy(logits / temperature, labels)


class ICon(nn.Module):

    def __init__(self, temperature: float = 0.07, alpha: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, sketch_feats: torch.Tensor, image_feats: torch.Tensor):
        return icon_loss(sketch_feats, image_feats, temperature=self.temperature, alpha=self.alpha)


def icon_loss(sketch_feats: torch.Tensor, image_feats: torch.Tensor, temperature: float = 0.07, alpha: float = 0.2):
    """
    Computes I-Con loss between sketch and image embeddings.

    Inputs:
        sketch_feats: [B, D] tensor of sketch embeddings (from sketch encoder)
        image_feats: [B, D] tensor of image embeddings (from image encoder)
        temperature: float, temperature for softmax similarity
        alpha: float in [0, 1], uniform debiasing weight

    Output:
        Scalar loss (torch.Tensor)
    """
    _validate_inputs(image_feats, sketch_feats)

    B, D = sketch_feats.size()

    # Normalize embeddings
    sketch_feats = F.normalize(sketch_feats, dim=1)
    image_feats = F.normalize(image_feats, dim=1)

    # Similarity scores: [B, B] (each sketch compared to all images)
    sim_matrix = torch.matmul(sketch_feats, image_feats.T) / temperature

    # q(j|i): learned distribution (softmax over images for each sketch)
    q = F.softmax(sim_matrix, dim=1)  # [B, B]

    # p(j|i): supervisory distribution
    # Each sketch i should match image i → one-hot, plus uniform smoothing
    p = torch.full_like(q, fill_value=alpha / B)  # uniform component
    p[range(B), range(B)] = 1.0 - alpha + (alpha / B)  # peak at matching pair

    # KL divergence for each sample: sum p * log(p / q)
    # Use logq + eps to avoid log(0)
    eps = 1e-8
    kl = p * (p.clamp(min=eps).log() - q.clamp(min=eps).log())
    loss = kl.sum(dim=1).mean()  # mean over batch

    return loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)

    def forward(self, sketch_feats: torch.Tensor, image_feats: torch.Tensor):
        _validate_inputs(image_feats, sketch_feats)
        # Shift the positive keys to get the negative keys
        negative_keys = image_feats.clone()
        negative_keys = torch.roll(negative_keys, shifts=1, dims=0)

        return self.triplet_loss(sketch_feats, image_feats, negative_keys)
