# AKD Loss — Line-by-Line Explanation

## Function Signature

```python
def akd_loss(target_net, output_net, anchor_target, anchor_net, opt, eps=1e-7):
```

**Inputs:**

| Argument | Shape | Description |
|---|---|---|
| `target_net` | `[B, D]` | Teacher's pooled feature vectors for the current mini-batch (B samples, D-dimensional). Detached — no gradients flow to the teacher. |
| `output_net` | `[B, D]` | Student's pooled feature vectors for the same mini-batch. Trainable. |
| `anchor_target` | `[A, D]` | Teacher's pooled features for the A anchor images. Precomputed once and frozen. |
| `anchor_net` | `[A, D]` | Student's pooled features for the anchor images. Recomputed every step through a learnable `AnchorNet`, so gradients flow back to it. |
| `opt` | object | Config object carrying the hyperparameters `l_1` and `l_2`. |
| `eps` | scalar | Small constant (1e-7) added to denominators/logs to prevent division by zero or log(0). |

---

## Step 1 — L2-Normalise all feature vectors (lines 22-25)

```python
anchor_target = F.normalize(anchor_target, p=2, dim=1)
output_net    = F.normalize(output_net,    p=2, dim=1)
target_net    = F.normalize(target_net,    p=2, dim=1)
anchor_net    = F.normalize(anchor_net,    p=2, dim=1)
```

Each feature vector is divided by its L2 norm so that it lies on the unit hypersphere. After this, `dot(a, b) = cosine_similarity(a, b)` because both vectors have magnitude 1. This removes magnitude differences and focuses purely on directional (angular) relationships.

---

## Step 2 — Compute pairwise cosine similarities, scaled to [0, 1] (lines 28-33)

### Batch-to-Anchor similarities (lines 28-30)

```python
a_student_sim = (torch.mm(output_net, torch.t(anchor_net)) + 1) / 2    # [B, A]
a_teacher_sim = (torch.mm(target_net, torch.t(anchor_target)) + 1) / 2  # [B, A]
```

- `torch.mm(output_net, torch.t(anchor_net))` computes a `[B, A]` matrix where entry `(i, j)` is the cosine similarity between batch sample `i` and anchor `j`.
- Raw cosine similarity ranges from -1 to +1. Adding 1 and dividing by 2 maps it to [0, 1], which is required because these values will later be normalised into probability distributions.
- `a_student_sim[i, j]` = "how similar is student's representation of batch sample `i` to student's representation of anchor `j`?"
- `a_teacher_sim[i, j]` = same question, but using the teacher's representations.

```python
a_teacher_sim_t, a_student_sim_t = torch.t(a_teacher_sim), torch.t(a_student_sim)  # [A, B]
```

- Transposes to get anchor-to-batch matrices `[A, B]`. Entry `(j, i)` now asks: "from anchor `j`'s perspective, how similar is batch sample `i`?"
- The transpose matters because the row-wise normalisation in step 3 will produce **different** distributions for each perspective: row `i` of `[B, A]` distributes sample `i`'s similarity mass across anchors, while row `j` of `[A, B]` distributes anchor `j`'s similarity mass across batch samples.

### Batch-to-Batch similarities (lines 32-33)

```python
b_student_sim = (torch.mm(output_net, torch.t(output_net)) + 1) / 2    # [B, B]
b_teacher_sim = (torch.mm(target_net, torch.t(target_net)) + 1) / 2    # [B, B]
```

- Self-similarity matrix within the batch. Entry `(i, j)` = how similar are samples `i` and `j` in the student's/teacher's feature space.
- The diagonal is always 1.0 (a sample is identical to itself after normalisation).

---

## Step 3 — Normalise rows into probability distributions (lines 36-41)

```python
a_student_sim   = a_student_sim   / torch.sum(a_student_sim,   dim=1, keepdim=True)
a_teacher_sim   = a_teacher_sim   / torch.sum(a_teacher_sim,   dim=1, keepdim=True)
a_teacher_sim_t = a_teacher_sim_t / torch.sum(a_teacher_sim_t, dim=1, keepdim=True)
a_student_sim_t = a_student_sim_t / torch.sum(a_student_sim_t, dim=1, keepdim=True)
b_student_sim   = b_student_sim   / torch.sum(b_student_sim,   dim=1, keepdim=True)
b_teacher_sim   = b_teacher_sim   / torch.sum(b_teacher_sim,   dim=1, keepdim=True)
```

Each row is divided by its sum so that it sums to 1, turning it into a valid probability distribution.

**Interpretation:** Row `i` of `a_teacher_sim` (after normalisation) is a probability distribution over anchors that encodes: "according to the teacher, how does batch sample `i` relate to each anchor?" The student has its own version of this distribution. The goal of distillation is to make the student's distribution match the teacher's.

This is why KL divergence is the natural loss: we are comparing probability distributions.

---

## Step 4 — KL divergence losses (lines 44-46)

```python
L_1 = torch.sum(b_teacher_sim * torch.log((b_teacher_sim + eps) / (b_student_sim + eps)))
L_2 = torch.sum(a_teacher_sim * torch.log((a_teacher_sim + eps) / (a_student_sim + eps)))
L_3 = torch.sum(a_teacher_sim_t * torch.log((a_teacher_sim_t + eps) / (a_student_sim_t + eps)))
```

Each line computes `KL(P_teacher || P_student) = sum( P_t * log(P_t / P_s) )` summed over all rows and columns of the matrix.

| Loss | Matrices | Shape | What it captures |
|---|---|---|---|
| **L_1** | `b_teacher` vs `b_student` | `[B, B]` | **Intra-batch structure.** Forces the student to preserve the teacher's pairwise relationships within the batch. If the teacher says samples 3 and 7 are very similar, the student should agree. |
| **L_2** | `a_teacher` vs `a_student` | `[B, A]` | **Batch-to-anchor positioning.** Each batch sample should be positioned relative to the anchors the same way the teacher positions it. This is the primary anchor-based loss. |
| **L_3** | `a_teacher_t` vs `a_student_t` | `[A, B]` | **Anchor-to-batch positioning.** The reverse perspective: from each anchor's viewpoint, the distribution over batch samples should match. This provides complementary gradient signal because the normalisation is done across a different axis. |

**Why eps is needed:** If any entry in `b_student_sim` is 0, `log(P_t / 0)` would be infinity. Adding `eps` to both numerator and denominator prevents this while having negligible effect on the values (since eps = 1e-7).

---

## Step 5 — Weighted combination (line 48)

```python
AKD_loss = opt.l_1 * L_1 + L_2 * (1 - opt.l_2) + L_3 * opt.l_2
```

The three losses are combined with two hyperparameters:

- **`opt.l_1`** controls the weight of the intra-batch loss L_1. Setting `l_1 = 0` disables intra-batch distillation entirely, using only anchor-based losses.
- **`opt.l_2`** balances between L_2 (batch-to-anchor) and L_3 (anchor-to-batch). With `l_2 = 0`, only L_2 is used. With `l_2 = 1`, only L_3 is used. Intermediate values blend both perspectives.

Note that L_2 and L_3 weights always sum to 1: `(1 - l_2) + l_2 = 1`. This means only the relative balance between the two anchor perspectives is controlled, not their total magnitude (which is implicitly 1.0 relative to `l_1 * L_1`).

---

## Summary of the full pipeline

```
Raw features [B,D] / [A,D]
        |
        v
   L2 normalise          (unit hypersphere)
        |
        v
   Cosine similarity      (dot product = cosine for unit vectors)
        |
        v
   Scale to [0, 1]        (add 1, divide by 2)
        |
        v
   Row-normalise           (turn into probability distributions)
        |
        v
   KL divergence           (teacher || student, for 3 matrix pairs)
        |
        v
   Weighted sum             (l_1 * L_1 + (1-l_2) * L_2 + l_2 * L_3)
```
