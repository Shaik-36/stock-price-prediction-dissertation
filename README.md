# Stock Price Prediction Using Sentiment Analysis and LSTM

Welcome to the **stock-price-prediction-dissertation** repository! This project demonstrates how to leverage sentiment analysis on Twitter data combined with machine learning (particularly an LSTM model) to predict stock prices. The repository includes code for data collection, preprocessing, feature engineering, sentiment analysis, and time-series modeling of stock prices.

## Table of Contents

- [Overview](#overview)
- [Thesis Abstract](#thesis-abstract)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Usage Instructions](#usage-instructions)
  1. [Sentiment Analysis (Part 1)](#1-sentiment-analysis-part-1)
  2. [Stock Data Analysis (Part 2)](#2-stock-data-analysis-part-2)
  3. [Merging Sentiment & Stock Data (Part 3)](#3-merging-sentiment--stock-data-part-3)
  4. [LSTM Modeling & Evaluation](#4-lstm-modeling--evaluation)
- [Key Results](#key-results)
- [License](#license)
- [Contact](#contact)

## Overview

Predicting stock prices can be extremely challenging due to the dynamic nature of the financial markets and the influence of external factors such as social media sentiment. This repository explores a Long Short-Term Memory (LSTM) based approach for stock price prediction, enhanced by sentiment scores extracted from a large Twitter dataset.

By combining VADER and TextBlob sentiment analysis tools, tweets are scored and then merged with stock price data. The integrated dataset is used to train an LSTM model that forecasts future stock prices, demonstrating how sentiment signals can improve traditional time-series predictions.

## Thesis Abstract

Predicting stock prices is a difficult task that is influenced by dynamic financial forces. The integration of sentiment analysis of a large social media dataset, namely Dow30 Stocks, with advanced Natural Language Processing (NLP) methods, such as VADER and TextBlob, and a machine learning model, specifically Long Short-Term Memory (LSTM), is investigated in this paper.

The research questions seek to understand how a large amount of sentiment data analysis improves stock market predictions, the specific improvements obtained by combining sentiment analysis and LSTM models, and the extent to which optimising large-scale data processing techniques improves accuracy.

This thesis proposes a methodology that combines two datasets from the same day: tweets and stock data. Over 100,000 tweets for each stock were collected from the Dow 30 Stocks dataset. Text cleaning and tokenization are performed using multiple Python library tools, followed by feature extraction from Twitter tweets. Using VADER and TextBlob, the cleaned tweets are then labelled with sentiment scores. The acquired scores are weighted and summed to get an overall sentiment score, which is then fed into an LSTM model with predefined hyperparameters.

Our LSTM-based model demonstrates superior performance in forecasting McDonald’s (MCD) stock prices among Dow 30 Stocks, achieving reduced MSE and RMSE by tuning hyperparameters. The utilisation of cross-validation additionally enhances the R² value for McDonald's stock from 0.821 to 0.916.

## Repository Structure

```yaml
stock-price-prediction-dissertation/
│
├── datasets/
│   ├── <Twitter CSV files>        # Place your tweet datasets here
│   ├── <Other relevant CSV files> # Additional data can be stored here
│   └── ...
│
├── ML Models/
│   ├── <Saved model files>        # (Optional) Trained model weights, if you choose to save them
│   └── ...
│
├── LICENSE                        # MIT License
├── MS Thesis - Stock Price Prediction - Imammuddin.pdf
├── README.md                      # You are here!
└── code/ or root-level scripts
    ├── [Part 1: Sentiment Analysis Code]
    ├── [Part 2: Stock Data Analysis Code]
    ├── [Part 3: Merging Data & Feature Engineering Code]
    └── [Part 4: LSTM Model & Evaluation Code]
```

**Note:**  
- You can place your raw Twitter datasets inside the `datasets/` folder.  
- Any pre-trained or saved ML models can go in the `ML Models/` folder.  
- The main code can reside in separate Python scripts or Jupyter Notebooks.

## Prerequisites

- **Python 3.7+**

**Libraries** (install via `pip install <library>` or `conda install <library>`):
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `nltk`
- `textblob`
- `spacy`
- `yfinance`
- `tensorflow` (or `keras`)
- `statsmodels`
- `wordcloud`
- `inflect`
- and other standard Python libraries (`re`, `datetime`, `tqdm`).

**NLTK Data:**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt')
```

**SpaCy Model:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## Usage Instructions

### 1. Sentiment Analysis (Part 1)
- **Load Twitter Datasets:**  
  Ensure your CSV files (e.g., `IBM.csv`, `MCD.csv`) are in the `datasets/` folder.  
  Update the code to point to the correct file paths (e.g., `df = pd.read_csv('datasets/IBM.csv')`).
- **Data Cleaning & Preprocessing:**  
  Removing URLs, mentions, hashtags, special characters.  
  Tokenizing, lemmatizing, removing stop words.
- **Sentiment Scoring:**  
  - **VADER:** Generates `vader_sentiment_score`.  
  - **TextBlob:** Generates `TextBlob_sentiment_score`.  
  Combine them into a single `final_sentiment_score` (weighted average).
- **Export Cleaned Data:**  
  Optionally save your processed dataset as a CSV (e.g., `df.to_csv("IBM_P1.csv", index=False)`).

### 2. Stock Data Analysis (Part 2)
- **Fetch Historical Stock Data:**  
  Use `yfinance` to pull historical stock data.  
  ```python
  import yfinance as yf
  stock_data = yf.download("IBM", start="YYYY-MM-DD", end="YYYY-MM-DD")
  ```
- **Feature Engineering:**  
  Calculate moving averages (MA), exponential moving average (EMA), Bollinger Bands, RSI, MACD, ATR, CCI, Stochastic Oscillator, etc.
- **Visualizations:**  
  Plot technical indicators against stock prices.

### 3. Merging Sentiment & Stock Data (Part 3)
- **Load the Cleaned Sentiment CSV:**  
  E.g., `Sentiment_df = pd.read_csv('IBM_P1.csv')`.
- **Aggregate Sentiment Scores:**  
  Group by Date to get daily or weekly average sentiment.
- **Merge with Stock Data:**  
  Perform an inner join on Date:  
  ```python
  final_df = pd.merge(stock_data, Sentiment_df, on='Date', how='inner')
  ```
- **Correlation Analysis & Additional Plots:**  
  Check how sentiment correlates with stock prices.

### 4. LSTM Modeling & Evaluation
- **Data Preparation:**  
  Select features (e.g., `[Adj Close, final_sentiment_score]`).  
  Scale the data using `MinMaxScaler`.  
  Create time-series sequences for the LSTM (`create_time_series_data` function).
- **Train-Test Split:**  
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
  ```
- **Build & Train the LSTM:**  
  ```python
  model = Sequential()
  model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(LSTM(units=50))
  model.add(Dense(units=1))
  model.compile(optimizer='adam', loss='mean_squared_error')
  history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
  ```
- **Predictions & Inverse Transform:**  
  Compare predicted vs. actual prices in the original scale.
- **Evaluation Metrics:**  
  MSE, RMSE, R² for numerical performance.  
  Percentage of Correct Direction (PCD) to gauge directional accuracy.
- **Cross-Validation:**  
  Uses `KFold` to evaluate model stability and performance across multiple folds.

## Key Results

- **Enhanced Accuracy:** The combined sentiment + LSTM approach often reduces MSE and RMSE compared to LSTM alone.
- **R² Improvements:** Cross-validation and hyperparameter tuning (optimizer, learning rate, batch size) significantly improved the R² scores, particularly notable in the McDonald’s (MCD) stock example (R² from 0.821 to 0.916).
- **Sentiment Impact:** The final sentiment score (VADER + TextBlob) provided additional predictive power for stock movements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions, clarifications, or collaborations, please feel free to reach out via:

- Name: Imamuddin Shaik
- Email: imamshan369@gmail.com
- LinkedIn: www.linkedin.com/in/shaik-imam


**Happy Predicting!**  
We hope this repository helps you explore the intersection of social media sentiment and financial forecasting.
