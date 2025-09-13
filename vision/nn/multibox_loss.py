import torch.nn as nn
import torch.nn.functional as F
import torch


from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # confidence = confidence[mask, :]
        masked_labels = labels[mask]
        masked_confidence = confidence[mask, :]
        #//////////////////////////////////////////////////////
        print("labels", masked_labels )
        print("labels", masked_labels .shape)
        print(30*"----")
        print("confidence",masked_confidence.shape)
        pred_classes = torch.argmax(masked_confidence, dim=2)
        print("pred_classes",pred_classes)
        print("pred_classes",pred_classes.shape)
        correct = (pred_classes == masked_labels)
        print("\nCorrect mask:\n", correct.shape)
        accuracy = correct.sum().item() / correct.numel()
        print("\nAccuracy:", accuracy)
        confidence_flat = masked_confidence.view(-1, masked_confidence.size(-1))
        labels_flat = labels.view(-1)
        loss = F.cross_entropy(confidence_flat, labels_flat, reduction="mean")
        print("\nCross-Entropy Loss:", loss.item())
        print(30*"----")
        #/////////////////////////////////////////////////////
        classification_loss = F.cross_entropy(confidence[mask, :].reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos
