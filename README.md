# Transfer-Learning-Guided Structured Pruning for Efficient and Accurate Deep Neural Network Compression

## Abstract

Deep neural networks often achieve high predictive performance at the cost of large model sizes and heavy computational requirements, which limits their deployment in resource-constrained environments. This work proposes a transfer-learning-guided structured pruning framework to compress deep neural networks while preserving or improving classification accuracy. A pretrained ResNet-18 model is adapted to the CIFAR-10 dataset using transfer learning and subsequently compressed using structured channel pruning. The pruned network is then fine-tuned to recover performance and improve generalization. Experimental evaluation demonstrates that the proposed approach significantly reduces model complexity while maintaining strong predictive performance. The pruned model reduces parameters by approximately 58%, decreases computational cost by about 56%, and improves test accuracy from 77.49% to 83.62%. In addition, inference latency is reduced by roughly 25%, confirming the effectiveness of the method for efficient deep learning deployment. These results show that transfer-learning-guided structured pruning provides an effective strategy for improving both model efficiency and predictive performance.

---

## Pipeline

The proposed framework follows a structured experimental pipeline consisting of five main stages.

### 1. Data Preparation

The CIFAR-10 dataset is used as the benchmark dataset. Images are normalized and resized to match the input resolution required by the pretrained backbone network. The dataset is split into training and test sets, and a subset of the training data is used to reduce training time while preserving representative samples.

### 2. Transfer Learning Initialization

A pretrained ResNet-18 model trained on ImageNet is used as the base network. The convolutional backbone is retained to leverage previously learned visual representations, while the final classification layer is replaced with a new fully connected layer suitable for the CIFAR-10 classes. Initially, only the classifier layer is trained to adapt the model to the new dataset.

### 3. Baseline Model Training

The adapted network is trained for several epochs using the CIFAR-10 training subset. This stage establishes the baseline model and records performance metrics including accuracy, parameter count, floating-point operations (FLOPs), and inference latency.

### 4. Structured Channel Pruning

Structured pruning is applied to convolutional layers to remove less important channels. This reduces model size and computational cost without disrupting the network architecture. The pruning process is guided by channel importance metrics derived from weight magnitudes.

### 5. Fine-Tuning and Model Recovery

After pruning, the entire network is fine-tuned for several epochs to recover lost performance and allow the model to adapt to the new architecture. This stage often improves generalization by eliminating redundant parameters.

### 6. Evaluation and Efficiency Analysis

The final pruned model is evaluated on the full CIFAR-10 test set. Performance is assessed using multiple metrics including:

* Test accuracy
* Parameter count
* FLOPs
* Inference latency
* Carbon emission estimates

These metrics provide a comprehensive evaluation of the trade-off between model efficiency and predictive performance.

---

## Key Contributions

* Introduces a **transfer-learning-guided structured pruning framework** for efficient neural network compression.
* Demonstrates that pruning can **improve model accuracy while reducing computational complexity**.
* Achieves **significant reductions in parameters, FLOPs, and inference latency**.
* Provides an **energy-aware evaluation using carbon emission tracking**.

---

## Experimental Results (Summary)

| Metric     | Baseline | Pruned  | Change |
| ---------- | -------- | ------- | ------ |
| Accuracy   | 77.49%   | 83.62%  | +6.13% |
| Parameters | 11.18M   | 4.71M   | −57.9% |
| FLOPs      | 1.82G    | 0.79G   | −56.7% |
| Latency    | 2.86 ms  | 2.13 ms | −25.5% |

---

## Conclusion

The experimental results demonstrate that transfer-learning-guided structured pruning can significantly reduce model complexity while maintaining or even improving classification accuracy. This approach enables efficient deep neural network deployment in environments where computational resources and energy consumption are critical considerations.

---
