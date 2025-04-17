Phishing Detection (Partial Progress)

This submodule is developed for an Intelligent Cybersecurity Threat Detection System.  
The current goal is to build an ML-based system to classify incoming content as either phishing or legitimate, focusing on:

- SMS-based phishing detection
- URL-based phishing detection

Other phishing modalities (e.g., email content, web form spoofing) may be explored in later stages depending on time and scope.

---

Current Project Scope

- SMS Phishing Detection

  - Model: `bert-base-uncased`
  - Dataset: SMS Spam Collection Dataset (from Kaggle)
  - Result: F1 Score ≈ 98.7%
  - Final Model: `sms-bert-model`

- URL Phishing Detection\*\*
  - Model: `distilbert-base-uncased`
  - Dataset: Combined Kaggle + OpenPhish phishing URLs
  - Incremental training in 3 phases
  - Evaluation against unseen PhiUSIIL phishing URL dataset
  - Final Model: `phishing_model_v1_after_phase1`

📌 See [`url_training_report.md`](./reports/url_training_report.md) for a detailed breakdown of the URL model training pipeline.

---

Google Drive Files

Due to GitHub's file size limits, some trained model files (e.g., `model.safetensors`) are hosted on Google Drive:

[Google Drive – Large Files](https://drive.google.com/drive/folders/1JjpGG69Nxp_wgp2x_QFvQ93F6K2RZ5ny?usp=share_link)

To use a model:

1. Clone this repository
2. Download the corresponding model folder from Drive
3. Place it in the correct location (e.g., `models/sms-bert-model/`)
4. Use with Hugging Face `from_pretrained()`
