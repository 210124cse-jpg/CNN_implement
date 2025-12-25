# Assignment 6 — CNN Implementation (10 Fruit Classification)

**Student ID:** 210124  
**Course:** Machine Learning / Deep Learning  
**Assignment:** CNN Implementation (Assignment 6)  
**Framework:** PyTorch  
**Notebook:** `210124_CNN_implementation.ipynb`  
**Repository:** `CNN_implement`

---

## 1. Objective

The goal of this assignment is to implement a **Convolutional Neural Network (CNN)** for image classification and evaluate its performance using:

- A **training dataset** and **test dataset** (provided in the repo)
- A set of **10 custom images** (captured/collected by me)
- Training + validation curves
- Confusion matrix
- Correct/incorrect prediction analysis

This implementation performs **10-class fruit classification**:
- apple, avocado, banana, cherry, kiwi, mango, orange, pineapple, strawberries, watermelon

---

## 2. Repository Structure
CNN_implement/
│── dataset-4/
│ ├── train/
│ │ ├── Apple/ Banana/ ... (10 classes)
│ ├── test/
│ │ ├── apple/ banana/ ... (10 classes)
│ ├── predict/ (optional)
│
│── custom_data/
│ ├── apple.jpg
│ ├── avocado.jpg
│ ├── banana.jpg
│ ├── cherry.jpg
│ ├── kiwi.jpg
│ ├── mango.jpg
│ ├── orange.jpg
│ ├── pineapple.jpg
│ ├── strawberry.jpg
│ ├── watermelon.jpg
│
│── model/
│ └── 210124.pth
│
│── 210124_CNN_implementation.ipynb
│── README.md


---

## 3. Dataset Used

### Standard Dataset (Provided)
- Stored inside `dataset-4/train` and `dataset-4/test`
- Total classes: **10 fruits**
- Folder names are slightly different in train vs test (capital/lowercase),
  so the notebook builds a **case-insensitive class mapping** for consistent labels.

### Custom Images (My Dataset)
- Stored inside `custom_data/`
- Total images: **10**
- These are used for **final prediction + confidence output**.

---

## 4. Preprocessing & DataLoader

### Image Preprocessing
- Resize: `160 × 160`
- Augmentation (Train only):
  - RandomHorizontalFlip
  - RandomRotation
  - ColorJitter
- Normalization:
  - Mean = `[0.485, 0.456, 0.406]`
  - Std  = `[0.229, 0.224, 0.225]`

### Train/Validation Split
- Training split: **80%**
- Validation split: **20%**

### Batch Size
- `batch_size = 32` (used for training/validation/testing)

---

## 5. CNN Model Architecture

A custom CNN model (**not pretrained**) was implemented using:

- Convolution layers + ReLU
- Batch Normalization
- MaxPooling
- Adaptive Average Pooling
- Dropout
- Final fully connected layer for 10 classes

The model is designed to avoid overfitting and not reach artificial 100% accuracy.

---

## 6. Training Configuration

- Optimizer: **Adam**
- Learning Rate: `1e-3`
- Weight Decay: `1e-4`
- Loss: **CrossEntropyLoss** with label smoothing `0.02`
- Epochs: `20`
- Scheduler: **ReduceLROnPlateau**
  - Reduces LR when validation accuracy plateaus

Best model is selected automatically based on **highest validation accuracy**.

---

## 7. Results

### Final Test Accuracy
- **TEST Accuracy ≈ 68–70%** (varies slightly depending on run)

> Note: This dataset has 10 visually similar classes (apple/orange/mango etc.),
so confusion among similar fruits is expected.

---

## 8. Required Outputs (Assignment Tasks)

All required tasks from the assignment are implemented in the notebook:

✅ **Train CNN model using dataset**  
✅ **Train + Validation DataLoaders**  
✅ **Loss vs Epochs graph**  
✅ **Accuracy vs Epochs graph**  
✅ **Confusion Matrix (Test set)**  
✅ **Prediction on 10 custom images with confidence**  
✅ **Incorrect prediction analysis (3 incorrect samples from test set)**  

### Additional Task (New Requirement)
✅ **Show total 10 correct and 3 incorrect predictions**

---

## 9. Figures (Screenshots Added)

The following figures are generated in the notebook and will be added to this README:

1. **Loss vs Epochs + Accuracy vs Epochs (Train & Validation)**
   <img width="699" height="754" alt="image" src="https://github.com/user-attachments/assets/715f7ada-e5b6-4328-aab4-1ee337e19c2a" />

2. **Confusion Matrix (Test Set)**

   <img width="630" height="519" alt="image" src="https://github.com/user-attachments/assets/050ae61b-e59d-4a84-8660-6921590d0aae" />
 
4. **10 Correct Predictions (Test Set)**
   <img width="1424" height="607" alt="image" src="https://github.com/user-attachments/assets/5a043178-3678-4931-ab61-24e549467de3" />

5. **3 Incorrect Predictions (Test Set)**
   <img width="1131" height="421" alt="image" src="https://github.com/user-attachments/assets/440ae35d-84bd-4b3d-8fe1-bf7e2b1b4465" />

7. **Predictions on 10 Custom Images with Confidence**

<img width="1477" height="570" alt="image" src="https://github.com/user-attachments/assets/cf4b0890-5892-47f7-83db-ee164b92468b" />

## 10. How to Run

### Run on Google Colab
1. Open `210124_CNN_implementation.ipynb` in Colab
2. Run all cells (Runtime → Run all)
3. Notebook automatically:
   - clones repo
   - loads datasets
   - trains model
   - generates all plots & outputs
   - saves model weights

### Run Locally (optional)
Install dependencies:
```bash
pip install torch torchvision scikit-learn matplotlib pillow


