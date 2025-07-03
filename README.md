# CNN for MNIST Digit Classification

This project implements a Convolutional Neural Network (CNN) **from scratch** in PyTorch to classify handwritten digits (0–9) from the **MNIST dataset**.

---

## 🧠 Why This Project?

The goal was to:
- Practice building a CNN architecture from scratch using PyTorch
- Understand how convolutional layers, pooling, flattening, and fully connected layers interact
- Train and evaluate a model on a real-world dataset (MNIST)
- Keep it minimal, readable, and efficient

---

## 📦 Dependencies

Make sure you have:
```bash
torch
torchvision
matplotlib
```

> If you're using Google Colab, these are preinstalled.

---

## 📌 Dataset

We're using the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which consists of:
- 60,000 training images
- 10,000 test images
- Each image is 28x28 grayscale of handwritten digits

---

## 🔧 Transforms

To prepare the data:
```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
```
- `ToTensor`: Converts images to PyTorch tensors.
- `Normalize`: Standardizes the pixel values using MNIST’s mean and std.

---

## 🧪 Dataloaders

We batch the dataset into chunks of **128 images** for training/testing using:
```python
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)
```

---

## 🏗️ Model Architecture

The CNN is built **from scratch** using `torch.nn.Module`. Here's the design:

```
Input: 28x28 grayscale image (MNIST)

Conv2d(1, 6, kernel=3, stride=1)
→ BatchNorm
→ ReLU
→ MaxPool(2x2)
→ Output: 13x13x6

Conv2d(6, 18, kernel=3, stride=1)
→ BatchNorm
→ ReLU
→ MaxPool(2x2)
→ Output: 5x5x18 = 450 features

Flatten → Fully Connected (450 → 128)
→ Dropout
→ Fully Connected (128 → 65)
→ Dropout
→ Fully Connected (65 → 10)
```

> We use ReLU activations, dropout regularization (0.25), and softmax is applied via CrossEntropyLoss.

---

## 🧠 Thought Process Behind the Model

- Start small: 6 filters in the first conv layer to limit overfitting and observe patterns
- Batch normalization helps stabilize training
- MaxPooling reduces dimensionality while keeping features
- Flattening turns image features into a vector for dense layers
- Dropout prevents overfitting between dense layers
- Output is a 10-dimensional logit vector for digits 0–9

---

## 🏃 Training Loop

- Trains for `num_epochs = 10`
- Uses Adam optimizer with learning rate = 0.001
- Calculates and prints:
  - Running loss (every 500 batches)
  - Train accuracy
  - Test accuracy after each epoch

---

## 📊 Results (Sample Output)

```
Epoch [1/10] 
Train Accuracy: 97.56%
Test Accuracy: 97.53%
```

Model converges quickly and performs well on both training and test sets.

---

## ✅ Summary

- A simple, clean CNN built from scratch
- Trained on MNIST for 10-class digit classification
- Great for beginners learning how to design CNNs in PyTorch

---

## 📁 File Structure

```
📦 CNN-for-MNIST-Dataset
 ┣ 📄 MNIST_CNN.ipynb      ← Main notebook
 ┗ 📄 README.md            ← You're reading it
```

---

## 🔗 Future Work (Optional Ideas)

- Add visualization of predictions
- Try different architectures (deeper networks, residual blocks)
- Apply to other datasets (e.g., Fashion-MNIST, CIFAR-10)
