# ITCS5665 - Natural Language Processing

## Project 1: Text Classification

- Name - Kaung Htet Tha
- Email - kaunghtet.tha@student.mahidol.ac.th

## Title - IMDB Movie Review Sentiment Analysis

Dataset - [IMDB 50K Movie Reviews (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)

## Project Structure

```
imdb_sentiment_project/
├── data/
│   └── sample_imdb_tiny.csv
│   └── IMDB Dataset.csv
├── notebooks/
│   └── 01_EDA.ipynb
│   └── 02_preprocessing.ipynb
│   └── 03_evaluation.ipynb
│   └── 04_interpretability.ipynb
│   └── 05_error_analysis.ipynb
├── reports/
│   ├── figures/
│   └── metrics/
├── models/
├── scripts/
│   ├── run_baselines.sh
│   └── run_infer.sh
├── src/
│   ├── data_utils.py
│   ├── train_baselines.py
│   ├── infer.py
├── requirements.txt
└── README.md
```


## How to use

1. Create a conda or venv virtual environment (Optional)

2. Install requirements ``` pip install -r requirements.txt ```

3. Run ```01_EDA.ipynb, 02_preprocessing.ipynb``` under ```\notebooks```

4. Train the models ```python src\train_baselines.py --data_path data\IMDB_clean.csv --val_size 0.1 --test_size 0.2 --seed 42 --max_features 200000 --ngrams 2 --n_jobs -1```

5. Test after training ```python src/infer.py --model_path models/nb_tfidf_model.joblib --text "This movie was absolutely fantastic! I loved every moment of it."``` (Optional)

6. Run ```03_evaluation.ipynb, 04_interpretability.ipynb, 05_error_analysis.ipynb``` under ```\notebooks```

