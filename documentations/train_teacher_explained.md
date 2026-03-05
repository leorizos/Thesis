# train_teacher.py - Detailed Explanation

## Purpose

This script trains a large, accurate **teacher model** from scratch using standard supervised learning (no knowledge distillation). The teacher model will later be used to guide the training of smaller student models.

---

## Imports and Dependencies

### Core PyTorch:
- `torch`, `torch.nn`, `torch.optim`: Neural network components
- `torch.distributed`: Multi-GPU training support
- `torch.multiprocessing`: Parallel processing
- `torch.backends.cudnn`: CUDA optimizations

### Custom modules:
- `models.model_dict`: Registry of all available models
- `dataset.cifar100/imagenet`: Data loaders
- `helper.util`: Utility functions (learning rate, metrics, JSON saving)
- `helper.loops`: Training and validation loops
- `tensorboard_logger`: TensorBoard logging for visualization

---

## Function: `parse_option()`

### What It Does:
Parses command-line arguments and sets up training configuration

### Key Parameters:

#### Basic Settings:
```
--print_freq (default: 200)
  How often to print training statistics
  Every 200 batches, prints loss and accuracy

--batch_size (default: 64)
  Number of samples per training batch
  Higher = more memory, more stable gradients
  Lower = less memory, more noisy gradients

--num_workers (default: 8)
  Parallel data loading threads
  Higher = faster data loading (if CPU/IO allows)

--epochs (default: 240)
  Total training iterations over full dataset
  CIFAR-100: 240 epochs is standard

--gpu_id (default: '0')
  Which GPU to use
  '0' = first GPU, '0,1' = first two GPUs
```

#### Optimization Settings:
```
--learning_rate (default: 0.05)
  Initial learning rate for SGD
  Special cases: 0.01 for MobileNet/ShuffleNet

--lr_decay_epochs (default: '150,180,210')
  When to reduce learning rate
  At epochs 150, 180, 210: multiply LR by 0.1

--lr_decay_rate (default: 0.1)
  Learning rate multiplication factor
  LR schedule: 0.05 → 0.005 → 0.0005 → 0.00005

--weight_decay (default: 5e-4)
  L2 regularization strength
  Prevents overfitting by penalizing large weights

--momentum (default: 0.9)
  SGD momentum parameter
  Accelerates training in relevant directions
```

#### Model Settings:
```
--model (default: 'resnet32x4')
  Which teacher architecture to train
  Options: resnet38, resnet110, resnet32x4, vgg13, etc.

--dataset (default: 'cifar100')
  Which dataset to use
  Choices: 'cifar100' or 'imagenet'

--trial (default: '0')
  Experiment ID for tracking multiple runs
  Useful for repeated experiments
```

#### Distributed Training:
```
--multiprocessing-distributed
  Enable multi-GPU distributed training
  Fastest method for multi-GPU setups

--dist-url (default: 'tcp://127.0.0.1:23451')
  URL for process communication in distributed training
```

### Path Setup:
```python
opt.model_path = './save/teachers/models'
opt.tb_path = './save/teachers/tensorboard'
opt.model_name = '{model}_vanilla_{dataset}_trial_{trial}'
```

**Example:** `resnet32x4_vanilla_cifar100_trial_0`

### Special Handling:
- MobileNet/ShuffleNet get lower learning rate (0.01 instead of 0.05)
- Creates directories if they don't exist

---

## Function: `main()`

### What It Does:
Main entry point that sets up training environment

### Flow:
1. Parse command-line arguments
2. Set CUDA_VISIBLE_DEVICES environment variable
3. Determine number of available GPUs
4. Launch training:
   - If multiprocessing: spawn multiple processes (one per GPU)
   - Else: launch single main_worker

### Why Multiprocessing?
- DistributedDataParallel is faster than DataParallel
- Each GPU gets its own process
- Better GPU utilization and scaling

---

## Function: `main_worker(gpu, ngpus_per_node, opt)`

This is the core training function. Let's break it down step-by-step:

### STEP 1: Setup GPU and Distributed Training

```python
opt.gpu = int(gpu)
if opt.multiprocessing_distributed:
    opt.rank = int(gpu)
    dist.init_process_group(...)
```

**What happens:**
- Each process gets assigned one GPU
- Processes are initialized for communication
- Rank identifies each process (GPU 0 = rank 0, etc.)

---

### STEP 2: Model Initialization

```python
n_cls = 100  # for CIFAR-100
model = model_dict[opt.model](num_classes=n_cls)
```

**What happens:**
- Looks up model architecture in model_dict
- Creates model instance (e.g., ResNet32x4)
- Model has n_cls output neurons for classification

**Model Architecture (example: resnet32x4):**
```
Input: [B, 3, 32, 32]
  ↓
conv1 + bn1 + relu: [B, 32, 32, 32]
  ↓
layer1 (blocks): [B, 32, 32, 32]
  ↓
layer2 (blocks): [B, 64, 16, 16]
  ↓
layer3 (blocks): [B, 128, 8, 8]
  ↓
avgpool: [B, 128]
  ↓
fc (classifier): [B, 100]
```

---

### STEP 3: Optimizer and Loss

```python
optimizer = optim.SGD(model.parameters(),
                      lr=opt.learning_rate,
                      momentum=opt.momentum,
                      weight_decay=opt.weight_decay)
criterion = nn.CrossEntropyLoss()
```

#### SGD (Stochastic Gradient Descent):
- Updates weights using gradients and momentum
- Formula: `w_new = w_old - lr * gradient + momentum * prev_update`
- weight_decay adds L2 penalty: `gradient += weight_decay * weights`

#### CrossEntropyLoss:
- Combines softmax + negative log likelihood
- For predicted logits [B, 100] and labels [B]:
  ```
  loss = -log(softmax(logits)[label_index])
  ```

---

### STEP 4: Move to GPU

```python
if opt.multiprocessing_distributed:
    torch.cuda.set_device(opt.gpu)
    model = model.cuda(opt.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opt.gpu])
else:
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()
```

**Three scenarios:**
1. **Distributed multi-GPU**: DistributedDataParallel (fastest)
2. **Single-node multi-GPU**: DataParallel
3. **Single GPU**: Simple `.cuda()`

**DistributedDataParallel vs DataParallel:**
- **DDP**: Each GPU has own process, better scaling
- **DP**: One process controls all GPUs, more overhead

---

### STEP 5: GPU Verification Block

```python
print("GPU CONFIGURATION CHECK")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

**Why this matters:**
- Confirms model is actually on GPU (not CPU)
- Debugging multi-GPU setups
- Prevents silent failures (training on CPU is ~100x slower)

---

### STEP 6: Data Loading

```python
if opt.dataset == 'cifar100':
    train_loader, val_loader = get_cifar100_dataloaders(
        batch_size=opt.batch_size,
        num_workers=opt.num_workers)
```

**CIFAR-100 specifics:**
- 50,000 training images (500 per class × 100 classes)
- 10,000 test images (100 per class)
- Image size: 32×32×3 RGB

**DataLoader creates batches:**
- Shuffles training data each epoch
- Applies data augmentation (RandomCrop, RandomHorizontalFlip)
- Normalizes with mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
- Uses num_workers processes for parallel loading

---

### STEP 7: Training Loop (Main Routine)

```python
for epoch in range(1, opt.epochs + 1):
    adjust_learning_rate(epoch, opt, optimizer)
    train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
    test_acc, test_acc_top5, test_loss = validate_vanilla(val_loader, model, criterion, opt)
    if test_acc > best_acc:
        save_model(...)
```

#### Epoch-by-Epoch Flow:

**Epochs 1-149:** Learning rate = 0.05
```
For each batch:
  1. Load batch: images [64, 3, 32, 32], labels [64]
  2. Forward pass: logits = model(images)
  3. Calculate loss: loss = CrossEntropy(logits, labels)
  4. Backward pass: loss.backward() computes gradients
  5. Optimizer step: updates weights using gradients
  6. Track metrics: top-1, top-5 accuracy, average loss

After all batches:
  1. Validate on test set
  2. Log to TensorBoard
  3. Save model if best accuracy so far
```

**Epoch 150:** Learning rate = 0.05 × 0.1 = 0.005

**Epoch 180:** Learning rate = 0.005 × 0.1 = 0.0005

**Epoch 210:** Learning rate = 0.0005 × 0.1 = 0.00005

**Epoch 240:** Final epoch

#### Why Decay Learning Rate?
- Early training: large LR explores parameter space quickly
- Later training: small LR fine-tunes for optimal solution
- Without decay: model oscillates and doesn't converge well

---

### STEP 8: Validation and Model Saving

```python
test_acc, test_acc_top5, test_loss = validate_vanilla(val_loader, model, criterion, opt)

if test_acc > best_acc:
    best_acc = test_acc
    state = {
        'epoch': epoch,
        'best_acc': best_acc,
        'model': model.state_dict(),
    }
    torch.save(state, save_file)
```

**Validation process:**
- Set model to eval mode (disables dropout, batch norm uses running stats)
- Forward pass on test set (no gradient computation)
- Calculate top-1 and top-5 accuracy
- **Top-1**: prediction == true label
- **Top-5**: true label in top 5 predictions

**Saved checkpoint contains:**
- `model.state_dict()`: All weight matrices and biases
- `epoch`: Which epoch achieved this accuracy
- `best_acc`: The top-1 accuracy on test set

**Additional JSON files:**

1. `test_best_metrics.json`:
```json
{
  "test_loss": 0.85,
  "test_acc": 78.45,
  "test_acc_top5": 94.23,
  "epoch": 235
}
```

2. `parameters.json`:
```json
{
  "model": "resnet32x4",
  "dataset": "cifar100",
  "learning_rate": 0.05,
  "Total params": 2.76,
  "Total time": 4.5
}
```

---

## Typical Training Output

```
Epoch: [1][0/782]    GPU 0    Time: 0.523    Loss 4.6234    Acc@1 1.562    Acc@5 6.250
Epoch: [1][200/782]  GPU 0    Time: 0.142    Loss 4.1523    Acc@1 3.281    Acc@5 12.109
...
Epoch: [1][782/782]  GPU 0    Time: 0.138    Loss 3.8245    Acc@1 8.437    Acc@5 23.594

 * Epoch 1, Acc@1 8.437, Acc@5 23.594, Time 112.34

Test: [0/157]   GPU: 0   Time: 0.234   Loss 3.7123   Acc@1 9.375   Acc@5 25.000
...
Test: [157/157] GPU: 0   Time: 0.187   Loss 3.6542   Acc@1 11.234  Acc@5 28.765

 ** Acc@1 11.234, Acc@5 28.765
saving the best model!

[Training continues...]

Epoch: [150][0/782]  GPU 0    Time: 0.145    Loss 0.7234    Acc@1 76.562    Acc@5 96.875
[LR decayed to 0.005]

...

Epoch: [240][782/782] GPU 0    Time: 0.139    Loss 0.5123    Acc@1 82.187    Acc@5 98.437

best accuracy: 78.45

Model saved to: save/teachers/models/resnet32x4_vanilla_cifar100_trial_0/resnet32x4_best.pth
```

---

## Key Design Decisions

### 1. Why SGD with Momentum?
- Momentum helps escape local minima
- More stable than Adam for this task
- Standard choice for image classification

### 2. Why Step LR Decay?
- Simple and effective
- Works well for CIFAR-100
- Alternative: Cosine annealing, but step decay is proven

### 3. Why CrossEntropy Loss?
- Standard for classification
- Numerically stable (combines log-softmax + NLL)
- Provides probability calibration

### 4. Why Validate Every Epoch?
- Track progress
- Early stopping if needed
- Save best model (might not be last epoch)

### 5. Why Distributed Training Support?
- Faster training on multi-GPU systems
- Scales to ImageNet (needs large batches)
- Production-ready code

---

## Common Usage Examples

```bash
# Train ResNet32x4 on CIFAR-100 (standard teacher)
python train_teacher.py --model resnet32x4 --dataset cifar100 --epochs 240

# Train with multiple GPUs (distributed)
python train_teacher.py --model resnet32x4 --multiprocessing-distributed --gpu_id 0,1

# Train ResNet110 on CIFAR-100 (very deep)
python train_teacher.py --model resnet110 --epochs 240

# Train on ImageNet (requires more resources)
python train_teacher.py --model ResNet50 --dataset imagenet --batch_size 256 --epochs 100

# Multiple trials for statistical significance
python train_teacher.py --model resnet32x4 --trial 0
python train_teacher.py --model resnet32x4 --trial 1
python train_teacher.py --model resnet32x4 --trial 2
```

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Reduce `--batch_size` (try 32 instead of 64)

### Issue: Training is slow
**Solution:**
- Increase `--num_workers` (more parallel data loading)
- Enable `cudnn.benchmark` (done automatically)
- Use multiple GPUs with `--multiprocessing-distributed`

### Issue: Accuracy not improving
**Solution:**
- Check learning rate (might be too high or too low)
- Verify data augmentation is working
- Check GPU verification block (might be training on CPU!)

### Issue: Loss becomes NaN
**Solution:**
- Lower learning rate
- Check for batch norm issues
- Verify data normalization

---

## Relation to Knowledge Distillation

This script produces the **TEACHER** model that will later guide student training.

### Teacher characteristics:
✓ Large model (many parameters)
✓ High accuracy (78%+ on CIFAR-100)
✓ Trained from scratch with full supervision
✓ Slow inference (many FLOPs)

The trained teacher will be loaded in `train_student.py` to:
1. Extract feature representations
2. Provide soft probability targets
3. Guide student learning through PKT loss

The teacher's knowledge (learned representations and decision boundaries) will be compressed into a much smaller student model while maintaining most of the accuracy.

---

## End of train_teacher.py Explanation
