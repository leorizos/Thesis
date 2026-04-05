"""
Evaluate the HuggingFace ViT model (Ahmed9275/Vit-Cifar100) on the CIFAR-100 test set.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm


DATA_FOLDER = '../data/'
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    print(f"Using device: {DEVICE}")

    # Load processor and model
    print("Loading model from HuggingFace...")
    processor = AutoImageProcessor.from_pretrained("Ahmed9275/Vit-Cifar100")
    model = AutoModelForImageClassification.from_pretrained("Ahmed9275/Vit-Cifar100")
    model.eval()
    model.to(DEVICE)

    # Load raw CIFAR-100 test set (no transforms — processor handles preprocessing)
    test_set = datasets.CIFAR100(root=DATA_FOLDER, train=False, download=True)

    def collate_fn(batch):
        # batch is a list of (PIL Image, label)
        images, labels = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        return inputs, torch.tensor(labels)

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.to(DEVICE)

            outputs = model(**inputs)
            logits = outputs.logits  # (B, num_classes)

            # Top-1
            preds = logits.argmax(dim=-1)
            correct_top1 += (preds == labels).sum().item()

            # Top-5
            top5 = logits.topk(5, dim=-1).indices
            correct_top5 += sum(
                labels[i].item() in top5[i].tolist() for i in range(len(labels))
            )

            total += labels.size(0)

    acc_top1 = 100.0 * correct_top1 / total
    acc_top5 = 100.0 * correct_top5 / total
    print(f"\nTest set: {total} samples")
    print(f"Top-1 Accuracy: {acc_top1:.2f}%")
    print(f"Top-5 Accuracy: {acc_top5:.2f}%")


if __name__ == '__main__':
    main()
