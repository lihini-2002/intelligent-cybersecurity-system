📄 SMS Spam Detection Report (Phase 1)
Subgroup: 22 – Group F  
Project: ML-Based Phishing Detection  
Dataset: SMS Spam Collection Dataset(from Kaggle)
Model: BERT (bert-base-uncased)  
Notebook: `explore_spam_sms.ipynb`

---

📊 Dataset Overview

- Total Samples: 5,574
- Classes: Ham (0), Spam (1)
- Split: 80% Train / 20% Test

---

⚙️ Preprocessing

- Tokenized using `BertTokenizer`
- Labels encoded: `ham → 0`, `spam → 1`
- Converted into Hugging Face-compatible `Dataset` objects

---

🧠 Training Configuration

- Epochs: 3
- Batch Size: 16
- Learning Rate Strategy: Warmup (500 steps)
- Optimizer: AdamW
- Weight Decay: 0.01
- Logging: every 10 steps → `../results/sms/logs/`
- Evaluation Strategy: Epoch
- Best Model Selection: Based on `F1-score`

---

✅ Final Evaluation Metrics (Best Epoch: 3)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 99.46% |
| Precision | 99.31% |
| Recall    | 96.64% |
| F1-score  | 97.96% |
| Eval Loss | 0.0274 |

---

🔍 Confusion Matrix

|             | Predicted Ham | Predicted Spam |
| ----------- | ------------- | -------------- |
| Actual Ham  | 965           | 1              |
| Actual Spam | 5             | 144            |

---

📉 Loss Curves

- Training loss: logged per step
- Validation loss: evaluated per epoch
- See plots under: `results/sms/loss_curve.png` and `confusion_matrix.png`

---

📁 Saved Artifacts

- Model: `models/sms-bert-model/`
- Checkpoints: `results/sms/checkpoint-*`
- Plots: `results/sms/`
- Logs: `results/sms/logs/`
- Notebook: `notebooks/explore_spam_sms.ipynb`

---

_Report generated automatically using Hugging Face Trainer logs._
