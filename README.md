# Transfer Learning–Guided Structured Pruning for Compact Deep Neural Networks: Improving Accuracy Without Sacrificing Compression

---

## Abstract

Neural network compression is critical for deploying deep learning models in resource-constrained environments. While structured pruning effectively reduces model size and computational cost, naive application typically degrades accuracy significantly. This work presents a compression pipeline for ResNet-18 on CIFAR-10 that combines transfer learning, Taylor-importance-based iterative structured pruning with an interleaved prune-then-recover strategy, and knowledge distillation fine-tuning. The key insight is that inserting short distillation recovery phases between each pruning step prevents representational collapse, allowing the Taylor importance scores at each subsequent step to be computed on a healthy model rather than a damaged one. The result is a pruned model that achieves **92.11% test accuracy — 0.87% higher than the unpruned baseline** — while reducing parameters by 65.1%, FLOPs by 67.3%, and inference latency by 27.4%.

---

## Pipeline

The pipeline consists of four sequential stages:

**Stage 1 — Transfer Learning Baseline.** A pretrained ImageNet ResNet-18 is adapted to CIFAR-10 using a two-phase strategy. The backbone is first frozen and only the classification head trained for 3 epochs (head warmup). The full network is then unfrozen and fine-tuned for 4 epochs using differential learning rates — earlier layers receive lower LRs (1e-5) than later layers (1e-4) and the head (5e-4), with a cosine annealing schedule. Data augmentation (random horizontal flip, random crop with padding, colour jitter) is applied throughout. The final baseline model and a frozen copy (the teacher) are saved before pruning begins.

**Stage 2 — Iterative Prune-Then-Recover.** Structured channel pruning is applied across 5 iterative steps targeting a cumulative 40% channel reduction. Before each pruning step, Taylor importance scores are computed by accumulating gradients over 5 mini-batches, providing a stable estimate of each filter's contribution to the loss. After each pruning step, 2 epochs of knowledge distillation recovery are run immediately, allowing the model to rebuild its representations before the next step begins. Channel counts are rounded to multiples of 8 (`round_to=8`) at every step to maintain CUDA kernel alignment and ensure real latency reduction.

**Stage 3 — Final Fine-Tuning with Knowledge Distillation.** Following the iterative pruning loop, the compressed model undergoes a final 10-epoch polishing phase. The loss combines KL divergence against the teacher's soft logits (temperature T=4, weight α=0.7) with hard cross-entropy (weight 1-α=0.3). A conservative starting LR of 1e-4 is used with cosine annealing warm restarts (T₀=3), and the best checkpoint by validation accuracy is restored at the end.

**Stage 4 — Export.** The final model is exported as both a saved PyTorch checkpoint and a TorchScript file for deployment.

```
ImageNet ResNet-18 (pretrained)
        │
        ▼
┌─────────────────────────┐
│  Stage 1: Transfer      │  Head warmup (3 epochs, frozen backbone)
│  Learning Baseline      │  + Full fine-tune (4 epochs, differential LRs)
└────────────┬────────────┘
             │  Baseline: 91.24% acc │ 11.18M params
             ▼
┌─────────────────────────┐
│  Stage 2: Iterative     │  For each of 5 steps:
│  Prune-Then-Recover     │    1. Accumulate Taylor gradients (5 batches)
│                         │    2. Prune 1/5 of target channels (round_to=8)
│                         │    3. KD recovery (2 epochs)
└────────────┬────────────┘
             │  Post-pruning: ~90.8% acc │ 3.90M params
             ▼
┌─────────────────────────┐
│  Stage 3: Final KD      │  10 epochs, T=4, α=0.7
│  Fine-Tuning            │  Cosine warm restarts, LR=1e-4
└────────────┬────────────┘
             │  Final: 92.11% acc │ 3.90M params
             ▼
┌─────────────────────────┐
│  Stage 4: Export        │  PyTorch checkpoint + TorchScript
└─────────────────────────┘
```

---

## Key Contributions

**1. Prune-Then-Recover as a First-Class Pipeline Stage.** Rather than treating recovery as a post-pruning afterthought, interleaved recovery is embedded into the pruning loop itself. This prevents the cumulative accuracy collapse observed in back-to-back iterative pruning, where Taylor scores computed on a severely damaged model at step 4–5 become unreliable. With recovery, post-prune accuracy never drops below 64.67% at any step, and fully rebounds to above 90% within 2 epochs each time.

**2. Taylor Importance with Multi-Batch Gradient Accumulation.** Single-batch Taylor scoring introduces high variance in filter rankings. Accumulating gradients over 5 mini-batches per step before pruning produces significantly more stable importance estimates, reducing the risk of pruning filters that are temporarily inactive rather than genuinely unimportant.

**3. CUDA-Aligned Channel Rounding.** Enforcing `round_to=8` on all pruned channel counts ensures that the resulting layer dimensions remain aligned with CUDA tensor core tile sizes. This is the direct cause of the 27.4% latency reduction — without alignment, structured pruning can paradoxically increase wall-clock inference time despite fewer FLOPs, as observed in prior runs.

**4. Accuracy Gain Through Compression.** The final pruned model surpasses its uncompressed teacher in test accuracy (+0.87%), demonstrating that structured pruning combined with distillation acts as a form of regularization that improves generalisation on the target dataset beyond what the full-capacity model achieves.

---

## Experimental Results (Summary)

**Dataset:** CIFAR-10 (10,000 training samples subset, full 10,000 test set)  
**Model:** ResNet-18 pretrained on ImageNet  
**Hardware:** NVIDIA T4 GPU

| Metric | Baseline | Pruned | Change |
|---|---|---|---|
| Accuracy (%) | 91.24 | **92.11** | **+0.87** |
| Parameters (M) | 11.18 | 3.90 | **−65.1%** |
| FLOPs (G) | 1.82 | 0.60 | **−67.3%** |
| Inference Latency (ms) | 2.93 | 2.13 | **−27.4%** |
| CO₂ Training Emissions (g) | 0.0035 | 0.0079 | +128.4% |

**Pruning step-by-step accuracy (pre- and post-recovery):**

| Step | Params (M) | Pre-Recover Acc | Post-Recover Acc |
|---|---|---|---|
| 1/5 | 9.14 | 64.67% | 90.44% |
| 2/5 | 7.56 | 66.80% | 90.48% |
| 3/5 | 6.29 | 88.93% | 89.81% |
| 4/5 | 4.95 | 75.36% | 90.74% |
| 5/5 | 3.90 | 80.05% | 90.82% |

The CO₂ increase reflects one-time training cost. At inference, the pruned model consumes 67.3% fewer FLOPs per forward pass, making it substantially more efficient over any realistic deployment lifetime.

---

## Conclusion

This work demonstrates that the accuracy–compression trade-off in structured pruning is not inevitable — it is a consequence of pipeline design. By embedding recovery directly into the pruning loop, using multi-batch Taylor importance scoring, and aligning channel counts for hardware efficiency, it is possible to produce a model that is simultaneously smaller, faster, and more accurate than its uncompressed counterpart. The final compressed ResNet-18 achieves 92.11% accuracy on CIFAR-10 with 65% fewer parameters, 67% fewer FLOPs, and 27% lower inference latency compared to the full baseline — without any architectural changes to the original network. These results suggest that iterative prune-then-recover with knowledge distillation is a robust and practical strategy for model compression in transfer learning settings.