# SimKD Codebase Study Roadmap
## How to Study and Understand the Knowledge Distillation Framework

This roadmap guides you through studying the SimKD codebase systematically, from beginner to advanced understanding.

---

## 📚 Quick Start (30 minutes)

**Goal:** Get a high-level understanding of what the codebase does

### Read First:
1. **[CODEBASE_FLOW_EXPLANATION.md](CODEBASE_FLOW_EXPLANATION.md)** (15 min)
   - Section 1: Overview of Knowledge Distillation
   - Section 2: Complete Training Flow
   - Section 10: Extending the Codebase

2. **[README.md](README.md)** (5 min)
   - Installation instructions
   - Quick usage examples

3. **[train_teacher_explained.md](train_teacher_explained.md)** (10 min)
   - Skip to "Complete Training Example" section
   - Understand how teachers are trained

### What You'll Learn:
- ✓ What knowledge distillation is
- ✓ How to train a teacher model
- ✓ How to train a student model
- ✓ Basic command-line usage

---

## 🎯 Beginner Level (2-3 hours)

**Goal:** Understand the core workflow and be able to run experiments

### Study Order:

####1. **Teacher Training** (30 min)
Read: [train_teacher_explained.md](train_teacher_explained.md)

**Focus on:**
- parse_option() function - all hyperparameters
- main_worker() - complete training loop
- Learning rate scheduling
- Model saving

**Hands-on:**
```bash
# Train a small teacher model
python train_teacher.py --model resnet32x4 --dataset cifar100 --epochs 5 --trial test
```

#### 2. **Data Loading** (30 min)
Read: [dataset_cifar100_explained.md](dataset_cifar100_explained.md)

**Focus on:**
- CIFAR-100 dataset structure
- Data augmentation (RandomCrop, RandomHorizontalFlip)
- Normalization statistics
- Standard vs. contrastive loaders

**Hands-on:**
```python
# Test data loaders interactively
from dataset.cifar100 import get_cifar100_dataloaders
train_loader, val_loader = get_cifar100_dataloaders(batch_size=64)
for images, labels in train_loader:
    print(f"Batch shape: {images.shape}, Labels: {labels.shape}")
    break
```

#### 3. **Model Architectures** (45 min)
Read: [models_explained.md](models_explained.md)

**Focus on:**
- ResNet architecture basics
- BasicBlock vs. Bottleneck
- **Critical:** `forward(x, is_feat=True)` for distillation
- Model comparison table

**Hands-on:**
```python
# Test feature extraction
from models import model_dict
model = model_dict['resnet32x4'](num_classes=100)
import torch
x = torch.randn(2, 3, 32, 32)
features, logits = model(x, is_feat=True)
print(f"Features: {[f.shape for f in features]}")
print(f"Logits: {logits.shape}")
```

#### 4. **Helper Functions** (45 min)
Read: [helper_util_explained.md](helper_util_explained.md) and [helper_loops_explained.md](helper_loops_explained.md)

**Focus on:**
- AverageMeter class (metric tracking)
- adjust_learning_rate() (LR scheduling)
- accuracy() function
- train_vanilla() vs. train_distill()

**Key Insight:**
Understand the difference between standard training (teacher) and distillation training (student).

### Checkpoint: Can You Answer These?
- [ ] What are the three components of the distillation loss?
- [ ] Why do we use `is_feat=True` during distillation?
- [ ] What's the difference between `module_list` and `trainable_list`?
- [ ] How does the learning rate change during training?

---

## 🚀 Intermediate Level (4-6 hours)

**Goal:** Deep understanding of PKT algorithm and student training

### Study Order:

#### 1. **The PKT Algorithm** (2 hours) ⭐ MOST IMPORTANT
Read: [distiller_zoo/PKT_explained.md](distiller_zoo/PKT_explained.md)

**Study in this order:**
1. Mathematical Foundation section (understand the theory)
2. Line-by-Line Explanation (understand the code)
3. Full Algorithm Walkthrough (see a complete example)
4. Why PKT Works (intuition)

**Deep Dive:**
- Read the original paper sections referenced
- Work through the example calculation by hand
- Understand why similarity matrices are [B, B] regardless of feature dimension

**Hands-on:**
```python
# Implement PKT loss yourself and compare
import torch
from distiller_zoo import PKT

pkt = PKT()
f_s = torch.randn(4, 256)  # Student: 4 samples, 256-dim
f_t = torch.randn(4, 512)  # Teacher: 4 samples, 512-dim (different!)
loss = pkt(f_s, f_t)
print(f"PKT Loss: {loss.item()}")

# Try to implement it yourself step-by-step
# Compare your result with the official implementation
```

#### 2. **Student Training Deep Dive** (2 hours)
Read: [train_student_explained.md](train_student_explained.md)

**Focus on:**
- All 10 distillation methods (Section 5)
- **PKT setup (Section 5I)** - why no projection module needed
- Loss components and their weights
- Complete training example

**Compare Methods:**
Create a table comparing:
- PKT vs. FitNet vs. KD vs. SimKD
- Which need projection modules?
- Which have learnable parameters?
- Performance trade-offs

**Hands-on:**
```bash
# Train student with PKT (assuming you have a trained teacher)
python train_student.py \
  --model_s resnet8x4 \
  --path_t save/teachers/models/resnet32x4_vanilla_cifar100_trial_test/resnet32x4_best.pth \
  --distill pkt \
  --cls 1.0 --div 1.0 --beta 1.0 \
  --epochs 10
```

#### 3. **Training Loops Analysis** (1 hour)
Read: [helper_loops_explained.md](helper_loops_explained.md)

**Focus on:**
- train_distill() function - step-by-step
- How teacher features are extracted (with `no_grad()`)
- Loss computation for different methods
- validate_distill() - SimKD special case

**Debugging Exercise:**
Add print statements to track loss components:
```python
# In helper/loops.py, add:
print(f"loss_cls: {loss_cls.item():.4f}, "
      f"loss_div: {loss_div.item():.4f}, "
      f"loss_kd: {loss_kd.item():.4f}")
```

### Checkpoint: Can You Answer These?
- [ ] Why doesn't PKT need a projection module like FitNet?
- [ ] What is the role of cosine similarity in PKT?
- [ ] How does PKT preserve geometric information?
- [ ] What's the difference between `cls`, `div`, and `beta` loss weights?
- [ ] Why is the teacher set to `eval()` mode during student training?

---

## 🔬 Advanced Level (6-10 hours)

**Goal:** Master the codebase, ready to modify and extend

### Study Order:

#### 1. **Paper-to-Code Mapping** (2 hours)
Read the ECCV 2018 PKT paper alongside:
- [distiller_zoo/PKT_explained.md](distiller_zoo/PKT_explained.md)
- [CODEBASE_FLOW_EXPLANATION.md](CODEBASE_FLOW_EXPLANATION.md) - Section 5

**Map each equation in the paper to code:**
- Equation 3-4 (probability distributions) → Lines 330-331 in PKT.py
- Equation 6 (cosine kernel) → Lines 239-240 in PKT.py
- Equation 8 (KL divergence) → Line 372 in PKT.py

**Theory Questions:**
- Why does PKT preserve Quadratic Mutual Information?
- What's the connection to metric learning?
- How does PKT relate to contrastive learning?

#### 2. **Compare All Distillation Methods** (3 hours)

Study each method in `distiller_zoo/`:
1. KD.py - Classic Hinton distillation
2. FitNet.py - Intermediate layer matching
3. AT.py - Attention transfer
4. SP.py - Similarity preservation
5. VID.py - Variational information distillation
6. PKT.py - Probabilistic knowledge transfer
7. SemCKD.py - Semantic cross-layer distillation
8. (CRD in crd/ folder)

**Create a comparison matrix:**

| Method | Params? | Projection? | Loss Type | Best For | Limitations |
|--------|---------|-------------|-----------|----------|-------------|
| KD | No | No | KL div | Logits | Only final layer |
| FitNet | Yes | Yes | MSE | Intermediate | Needs alignment |
| PKT | No | No | KL div (similarities) | Any features | Needs large batch |
| ... | ... | ... | ... | ... | ... |

#### 3. **Distributed Training** (2 hours)
**Focus on:**
- DistributedDataParallel vs. DataParallel
- `reduce_tensor()` function for metric aggregation
- Multi-GPU synchronization

**Hands-on:**
```bash
# Try multi-GPU training (if you have 2+ GPUs)
python train_student.py \
  --multiprocessing-distributed \
  --gpu_id 0,1 \
  --model_s resnet8x4 \
  --path_t <teacher_path> \
  --distill pkt
```

#### 4. **Extending the Codebase** (3 hours)

**Exercise 1: Implement a new distillation method**
- Create `distiller_zoo/MyMethod.py`
- Add to `distiller_zoo/__init__.py`
- Modify `train_student.py` to include your method
- Test it!

**Exercise 2: Add a new model architecture**
- Implement a custom ResNet variant
- Ensure it has `forward(x, is_feat=True)`
- Add to `models/__init__.py`
- Train it as teacher and student

**Exercise 3: Modify PKT**
- Try different kernels (Gaussian instead of cosine)
- Try different divergences (JS instead of KL)
- Apply PKT to multiple layers
- Compare results

### Checkpoint: Can You Answer These?
- [ ] How would you implement multi-layer PKT?
- [ ] What changes are needed to use a different dataset?
- [ ] How does gradient flow work in distributed training?
- [ ] Can you explain each line of PKT.py without looking?
- [ ] What are the trade-offs between different distillation methods?

---

## 🎓 Expert Level (10+ hours)

**Goal:** Research-level understanding, ready to publish

### Advanced Topics:

#### 1. **Ablation Studies** (3 hours)
Design and run experiments:
- Effect of batch size on PKT performance
- Optimal beta values for different teacher-student pairs
- Multi-layer vs. single-layer PKT
- PKT with and without classification loss

#### 2. **Theoretical Deep Dive** (4 hours)
- Prove why PKT preserves mutual information
- Analyze PKT's connection to optimal transport
- Study PKT's relationship to t-SNE
- Understand the geometry of similarity spaces

#### 3. **Cross-Domain Knowledge Transfer** (3 hours)
From the paper: PKT can transfer across modalities
- Try transferring from ImageNet to CIFAR-100
- Try transferring from handcrafted features to neural nets
- Experiment with different teacher-student architecture pairs

#### 4. **Reproduce Paper Results** (4+ hours)
Goal: Match Table 1 from ECCV 2018 paper

Train all configurations:
```bash
# Teacher: ResNet32x4
python train_teacher.py --model resnet32x4 --epochs 240

# Student: ResNet8x4 (baseline)
python train_teacher.py --model resnet8x4 --epochs 240

# Student: ResNet8x4 + PKT
python train_student.py --model_s resnet8x4 --distill pkt --epochs 240

# Compare: KD, FitNet, Attention, etc.
```

Expected results (CIFAR-100):
- Teacher (ResNet32x4): ~78%
- Student baseline (ResNet8x4): ~69%
- Student + PKT: ~72%

#### 5. **Novel Research Directions** (Ongoing)
Ideas to explore:
- **Adaptive PKT:** Learn optimal beta per layer
- **Curriculum PKT:** Start with easy samples, progress to hard
- **Self-distillation:** Use PKT for self-supervised learning
- **PKT + Contrastive:** Combine with SimCLR/MoCo
- **Efficient PKT:** Approximate similarity matrices for large batches

---

## 📝 Study Tips

### Active Learning:
- ✅ **Don't just read** - type out examples and run them
- ✅ **Modify code** - change hyperparameters and observe effects
- ✅ **Draw diagrams** - visualize data flow and architecture
- ✅ **Teach someone** - explain concepts to reinforce understanding

### Code Reading Strategy:
1. **Top-down:** Start with high-level (train_student.py) → drill down to details (PKT.py)
2. **Bottom-up:** Understand components (models, datasets) → see how they combine
3. **Trace execution:** Pick a single input batch, follow it through entire pipeline
4. **Compare implementations:** See how different methods solve similar problems

### Debugging Workflow:
1. Add print statements to track tensor shapes
2. Use Python debugger: `import pdb; pdb.set_trace()`
3. Visualize similarity matrices: `import matplotlib.pyplot as plt; plt.imshow(similarity.cpu())`
4. Check gradients: `print(param.grad.norm())`

### Common Pitfalls to Avoid:
- ❌ Skipping the math in PKT_explained.md (it's essential!)
- ❌ Not running code examples (reading alone isn't enough)
- ❌ Ignoring hyperparameter meanings (cls, div, beta, etc.)
- ❌ Not comparing different methods side-by-side

---

## 🎯 Learning Milestones

### Milestone 1: Beginner ✓
- [ ] Successfully train a teacher model
- [ ] Successfully train a student with PKT
- [ ] Understand the three loss components
- [ ] Can explain what knowledge distillation is to someone

### Milestone 2: Intermediate ✓
- [ ] Understand PKT algorithm line-by-line
- [ ] Can implement a simple distillation method from scratch
- [ ] Know when to use each distillation method
- [ ] Can debug training issues

### Milestone 3: Advanced ✓
- [ ] Successfully extend codebase with new method/model
- [ ] Understand distributed training internals
- [ ] Can reproduce paper results
- [ ] Know all 10 distillation methods in detail

### Milestone 4: Expert ✓
- [ ] Can derive PKT's theoretical properties
- [ ] Designed and ran original ablation studies
- [ ] Contributed novel research idea
- [ ] Ready to publish/extend this work

---

## 📚 Recommended Reading Order

### Session 1 (30-60 min):
1. CODEBASE_FLOW_EXPLANATION.md (Sections 1-3)
2. train_teacher_explained.md (Complete training example)

### Session 2 (1-2 hours):
1. dataset_cifar100_explained.md
2. models_explained.md
3. Run teacher training

### Session 3 (2-3 hours):
1. distiller_zoo/PKT_explained.md (Mathematical foundation + Code structure)
2. Implement PKT loss yourself

### Session 4 (2-3 hours):
1. train_student_explained.md (All sections)
2. Run student training with PKT

### Session 5 (1-2 hours):
1. helper_util_explained.md
2. helper_loops_explained.md
3. Trace through one complete training iteration

### Session 6+ (Ongoing):
- Deep dive into other distillation methods
- Run experiments
- Extend codebase
- Read original papers

---

## 🔗 External Resources

### Papers to Read:
1. **PKT (ECCV 2018):** "Learning Deep Representations with Probabilistic Knowledge Transfer"
2. **Original KD (2015):** Hinton et al. "Distilling the Knowledge in a Neural Network"
3. **FitNets (2015):** "FitNets: Hints for Thin Deep Nets"
4. **Attention Transfer (2017):** "Paying More Attention to Attention"
5. **CRD (2020):** "Contrastive Representation Distillation"

### Concepts to Study:
- **ResNet architecture:** Skip connections, batch normalization
- **Knowledge distillation:** Teacher-student paradigm, dark knowledge
- **KL divergence:** Information theory basics
- **Cosine similarity:** Metric learning, pairwise distances
- **Distributed training:** DDP, gradient synchronization

### Tools to Learn:
- **PyTorch:** nn.Module, DataLoader, autograd
- **TensorBoard:** Logging and visualization
- **Git:** Version control for experiments
- **Weights & Biases (optional):** Experiment tracking

---

## ✨ Final Advice

**Remember:**
> "The only way to learn programming is by writing programs"
> - Dennis Ritchie

**Apply this to ML research:**
> "The only way to learn knowledge distillation is by running experiments"

**Key Principles:**
1. **Start simple:** Train on small datasets/epochs first
2. **Iterate quickly:** Run many short experiments > few long ones
3. **Compare systematically:** Change ONE thing at a time
4. **Document everything:** Keep a research log
5. **Ask questions:** Why does this work? When does it fail?

**Good luck with your studies! 🚀**

---

## 📞 Getting Help

If you encounter issues:
1. Check the relevant *_explained.md file
2. Look for similar issues in the code
3. Add debug print statements
4. Compare with working examples
5. Review the original paper

**Remember:** Every expert was once a beginner. Take your time, be patient, and enjoy the learning process!
