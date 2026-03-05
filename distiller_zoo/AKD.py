from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


def akd_loss(target_net, output_net, anchor_target, anchor_net, opt, eps=1e-7):
    """
    Anchor-based Knowledge Distillation loss.

    Args:
        target_net: teacher pooled features for the batch
        output_net: student pooled features for the batch
        anchor_target: teacher pooled features for anchor images (precomputed, detached)
        anchor_net: student pooled features for anchor images (computed each step)
        opt: options containing l_1 and l_2 hyperparameters
        eps: small constant for numerical stability
    """
    # Normalize each vector
    anchor_target = F.normalize(anchor_target, p=2, dim=1)
    output_net = F.normalize(output_net, p=2, dim=1)
    target_net = F.normalize(target_net, p=2, dim=1)
    anchor_net = F.normalize(anchor_net, p=2, dim=1)

    # Cosine similarity scaled to [0, 1]
    a_student_sim = (torch.mm(output_net, torch.t(anchor_net)) + 1) / 2
    a_teacher_sim = (torch.mm(target_net, torch.t(anchor_target)) + 1) / 2
    a_teacher_sim_t, a_student_sim_t = torch.t(a_teacher_sim), torch.t(a_student_sim)

    b_student_sim = (torch.mm(output_net, torch.t(output_net)) + 1) / 2
    b_teacher_sim = (torch.mm(target_net, torch.t(target_net)) + 1) / 2

    # Normalize to probability distributions
    a_student_sim = a_student_sim / torch.sum(a_student_sim, dim=1, keepdim=True)
    a_teacher_sim = a_teacher_sim / torch.sum(a_teacher_sim, dim=1, keepdim=True)
    a_teacher_sim_t = a_teacher_sim_t / torch.sum(a_teacher_sim_t, dim=1, keepdim=True)
    a_student_sim_t = a_student_sim_t / torch.sum(a_student_sim_t, dim=1, keepdim=True)
    b_student_sim = b_student_sim / torch.sum(b_student_sim, dim=1, keepdim=True)
    b_teacher_sim = b_teacher_sim / torch.sum(b_teacher_sim, dim=1, keepdim=True)

    # KL divergence components
    L_1 = torch.sum(b_teacher_sim * torch.log((b_teacher_sim + eps) / (b_student_sim + eps)))
    L_2 = torch.sum(a_teacher_sim * torch.log((a_teacher_sim + eps) / (a_student_sim + eps)))
    L_3 = torch.sum(a_teacher_sim_t * torch.log((a_teacher_sim_t + eps) / (a_student_sim_t + eps)))

    AKD_loss = opt.l_1 * L_1 + L_2 * (1 - opt.l_2) + L_3 * opt.l_2
    return AKD_loss


class AnchorNet(nn.Module):
    """Learnable spatial attention network for anchor images."""

    def __init__(self, anchor_batch_size=100, DIM=32, weight_size=14):
        super(AnchorNet, self).__init__()
        self.weight_size = weight_size
        self.dim = DIM
        self.weights = nn.Parameter(
            torch.full((anchor_batch_size, 1, weight_size, weight_size), 1.0)
        )

    def forward(self, x, indices=None):
        batch_size = x.size(0)
        weights = self.weights

        if indices is not None:
            if isinstance(indices, int):
                indices = torch.tensor([indices])
            weights = weights[indices]

        self.stabilize_energy()
        weights = F.interpolate(
            weights, size=(self.dim, self.dim), mode='bilinear', align_corners=False
        )
        weights = weights.expand(batch_size, 3, self.dim, self.dim)

        output = x * weights
        return output

    def stabilize_energy(self):
        E = 1.0 * self.weight_size * self.weight_size
        with torch.no_grad():
            for i in range(self.weights.size(0)):
                current_energy = self.weights[i].sum()
                self.weights[i].data = (E / current_energy) * self.weights[i]


def calculate_anchor_set(net, train_loader, subset_length, num_classes=100, anchors_per_class=1):
    """
    Select the most representative sample per class as anchors, based on
    cosine-similarity centrality of the teacher's pooled features.

    Adapted for SimKD's data loader format and model interface.
    """
    anchor_set = []
    class_samples = [[] for _ in range(num_classes)]
    class_features = [[] for _ in range(num_classes)]

    net.eval()
    with torch.no_grad():
        for idx, batch in enumerate(train_loader):
            images, labels = batch[0], batch[1]

            feat_list, _ = net(images.cuda(), is_feat=True)
            features = feat_list[-1].cpu().detach()  # pooled features

            for i in range(labels.size(0)):
                class_samples[labels[i].item()].append(images[i])
                class_features[labels[i].item()].append(features[i])

    for class_id, samples_in_one_class in enumerate(class_samples):
        if subset_length == "max":
            subset_len = len(samples_in_one_class)
        else:
            subset_len = subset_length

        if samples_in_one_class:
            features_in_one_class = class_features[class_id]
            random_indices = torch.randperm(len(samples_in_one_class))[:subset_len]
            subset = [samples_in_one_class[i] for i in random_indices]
            features_in_one_class = [features_in_one_class[i] for i in random_indices]

            vectorized_samples = torch.stack(features_in_one_class).view(subset_len, -1)
            similarity_matrix = torch.tensor(
                cosine_similarity(vectorized_samples.numpy())
            )
            centrality_scores = torch.sum(similarity_matrix, dim=1)

            if subset_length != 0:
                subset = [subset[i] for i in torch.argsort(centrality_scores, descending=True)]
            selected_images = [subset[i] for i in range(anchors_per_class)]
            anchor_set.extend(selected_images)

    return torch.stack(anchor_set)
