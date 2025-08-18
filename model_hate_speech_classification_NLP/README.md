### Hate Speech Detection Project

This project focuses on detecting hate speech in text data through binary classification (hate speech vs. non-hate speech, **Task A**) and multi-class classification (identifying the type of hate speech, **Task B**). The project leverages machine learning models, large language models (LLMs), and data augmentation techniques to achieve robust classification performance.

### Project Overview

The project includes:
- **Binary Classification (Task A)**: Determines whether a given text contains hate speech or not.
- **Multi-Class Classification (Task B)**: Identifies the specific type of hate speech (e.g., racism, sexism, etc.) in the text.
- Implementation of various machine learning and deep learning models for classification.
- Use of LLMs for embedding generation and data augmentation.
- A Django-based web application for real-time hate speech detection.
- Model interpretability using SHAP and LIME.
- Comprehensive evaluation of model performance.

### Folder Structure

- **colab_notebooks/**: Contains Jupyter notebooks for experimentation and model development:
  - `falcon_llm.ipynb`: Tests embedding generation using the Falcon LLM and XGBoost for classification.
  - `falcon_template.ipynb`: Adds a prompt to the Falcon model and evaluates its responses.
  - `Pegasus_data_generation_type_of_hate.ipynb`: Uses the Pegasus paraphraser model to generate synthetic data for hate speech types.
  - `Retrieval_and_classification.ipynb`: Implements a retrieval and classification pipeline inspired from RAG (retreival and generation).
- **Data/**: Stores all datasets, including:
  - Initial dataset.
  - Generated synthetic data.
  - Embedding data for models.
- **Embeddings/**: Contains scripts for generating embeddings:
  - BERT embeddings.
  - RoBERTa embeddings.
  - DeBERTa embeddings.
- **Interface/**: Contains the Django-based web application:
  - Run the app by installing dependencies from `requirements.txt` and executing `python manage.py runserver`.
  - Else you can watch the demo video: `Hate_Speech_Detector.mp4`.
- **Models/**: Includes classes for classification models with methods for training, evaluation, and prediction.
- **Preprocess/**: Contains classes and scripts for:
  - Data reading and cleaning.
  - Data visualization.
  - SHAP and LIME for model interpretability.
- **Prog/**: Contains the `ModelEvaluator` class to display model accuracies in a DataFrame.

### Main Notebooks

- **hate_speech_notebook.ipynb**: Full implementation of binary classification (Task A).
- **type_of_hate_notebook.ipynb**: Full implementation of multi-class classification (Task B).

### Saved Models

- Best-performing models are saved in `.h5` and `.pkl` formats for use in the Django web application.

### Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt