url_report.md

URL Phishing Detection Model Training Report

--Overview--
This report documents the complete training process of a BERT-based model for detecting phishing URLs. It includes the sources of the data, preprocessing steps, splitting strategies, incremental training phases, model checkpointing practices, and all associated datasets and models.

The goal is to ensure full traceability, clarity, and reproducibility of the training workflow, so that the work can be reviewed, improved, or extended in the future.

--Phase 0: Dataset Collection and Preparation--
Step 1: Downloaded Datasets - Kaggle phishing URL dataset: `phishing_site_urls.csv` - OpenPhish feed file: `feed.txt`

    Step 2: Preprocessing
    - Removed duplicates within each dataset
    - Normalized all URLs (lowercase, removed protocol prefixes and trailing slashes)
    - Assigned labels: `1 = phishing`, `0 = legitimate`
    - Dropped NaN rows and inconsistencies

    Step 3: Combined Cleaned Dataset
    - Concatenated Kaggle and OpenPhish datasets
    - Deduplicated the combined result
    - Final combined dataset saved as:
    phishing_urls_combined_cleaned.csv

--Early Training Attempt (Local Jupyter Notebook with CPU)

    Objective
    To begin model training using Jupyter Notebook on a local machine.

    Outcome and Learnings
    - Successfully set up the dataset and tokenizer
    - Model training on the full dataset using local CPU was very slow
    - High memory consumption during tokenization and training caused crashes or freezing

    Key Learning
    - Local CPU-based training was not practical for large-scale BERT training
    - Decided to switch to Google Colab with GPU to accelerate training and prevent resource bottlenecks

    Jupyter Notebook used
    -explore_phishing_urls.ipynb

--Pre-Phase Experiment (Initial Full-Set Training with BERT)--

    Objective
        To initially fine-tune `bert-base-uncased` on the entire combined phishing dataset without splitting, using Hugging Face's `Trainer` API.

    Outcome and Learnings
    - Dataset was successfully loaded and tokenized
    - The full dataset was too large to fit in Colab's RAM and GPU constraints
    - Tokenization caused session crashes due to memory overflow
    - Training showed extremely long runtime estimates (~135 hours)

    Key Learning
    - Direct training on the full dataset in a single run was impractical on free Colab
    - Decided to switch to DistilBERT and adopt incremental training strategy with dataset chunks to better manage memory and training time

    Google Colab notebook used
    -colab_prePhase.ipynb

--Phase 1: Training--

    Step 1: Chunk Creation
    - Split the combined dataset into 3 equal parts using pandas
    - Saved as:
    phishing_urls_part1.csv`, `part2.csv`, and `part3.csv`

    Step 2:
        Training Phase 1

            Google Colab notebook: training_phase1.ipynb

            - Model used: `distilbert-base-uncased`
            - Dataset used: `phishing_urls_part1.csv`
            - Training ran for `1 epoch`
            - Tokenized in batches using custom function
            - Trainer API from Hugging Face was used with GPU via Google Colab
            - Model was saved to:
            phishing_model_v1_after_phase1

            - Also stored interim in `phishing_model_checkpoint`

            Evaluation after Phase 1
                - Accuracy: ~98.17%
                - F1 Score: ~98.08%
                - Precision and Recall: strong values

        Training Phase 2: Training on Chunk 2 (Issue Encountered)

            Google Colab notebook: training_phase2.ipynb

            Step 1: Loaded Chunk 2
            - Dataset used: `phishing_urls_part2.csv`
            -This chunk contained only legitimate (label = 0) URLs

            Step 2: Training Outcome
            - Accuracy = 1.0, but
            - Precision, Recall, and F1 Score = 0.0
            - Model failed to learn phishing characteristics due to lack of positive examples
            -Model saved as 'phishing_model_v2_after_phase2'

            Step 3: Recovery Plan
            - Discarded `phishing_model_checkpoint/` after Phase 2
            - Reverted back to `phishing_model_v1_after_phase1` as the stable base model


        Training Phase 3: Combined Training on Chunks 2 and 3

            Google colab notebook :training_phase3.ipynb

            Step 1: Combine and Balance
            - Combined: `phishing_urls_part2.csv + phishing_urls_part3.csv`
            - Saved as: `phase2_combined.csv`
            - Stratified split into:
            - `phase2_train.csv`
            - `phase2_test.csv`

            Step 2: Training
            - Model resumed from: `phishing_model_v1_after_phase1`
            - Trained on: `phase2_train.csv`
            - Evaluated on: `phase2_test.csv`
            - Final model saved as:
            phishing_model_v2_after_phase3

            Evaluation after training:
            eval_loss: 0.027734674513339996
            eval_accuracy: 0.9933316426237823
            eval_precision: 0.9802350427350427
            eval_recall: 0.9424756034925527
            eval_f1: 0.9609845509295627

--Phase 3: Evaluation and Model Selection--

    To objectively compare the models trained in Phase 1 and Phase 3, an external phishing URL dataset was used for evaluation:

    Dataset: PhiUSIIL Phishing URL Dataset (from Kaggle)

    Purpose: This dataset was not used in any phase of model training. It served as a true  holdout set to benchmark final model performance on unseen phishing and legitimate URLs.

    Preprocessing:
        Lowercased all URLs
        Removed protocol prefixes (http://, https://)
        Removed trailing slashes
        Kept only url and label columns

        Preprocessed dataset : PhiUSIIL_Phishing_URL_Eval_Preprocessed.csv
        Notebook used for cleaning the dataset: eval_dataset_cleaning.ipynb

    Evaluation :
        Both phishing_model_v1_after_phase1 and phishing_model_v2_after_phase3 were evaluated using Hugging Face's Trainer.evaluate() on this full evaluation set.

        Colab Notebook: model_evaluation_url.ipynb


        Final Evaluation Results (on PhiUSIIL Dataset)

        Metric          model_v1            model_v2

        Accuracy        0.5831              0.5831
        Precision       0.5817              0.5817
        Recall          0.9643              0.9643
        F1 score        0.7257              0.7257
        Eval loss       2.8182              2.8182

    Final Model Selection:
        -The models performed identically on the unseen PhiUSIIL dataset.
        -Given that phishing_model_v1_after_phase1 is lighter, trained on less data, and already performs optimally, it was selected as the final model for deployment and integration.

Selected Final Model: phishing_model_v1_after_phase1
