# NLP Projects Repository

A comprehensive collection of Natural Language Processing (NLP) projects demonstrating various text classification techniques, from binary to multi-label classification, using real-world datasets.

## Table of Contents
- [Overview](#overview)
- [Projects](#projects)
  - [1. Movie Genre Classifier](#1-movie-genre-classifier)
  - [2. Fake News Detector](#2-fake-news-detector)
  - [3. Restaurant Review Sentiment Analysis](#3-restaurant-review-sentiment-analysis)
  - [4. SMS Spam Classifier](#4-sms-spam-classifier)
  - [5. Stock Sentiment Analysis](#5-stock-sentiment-analysis)
- [Installation](#installation)
- [Technologies Used](#technologies-used)
- [How to Use](#how-to-use)

## Overview

This repository contains five NLP projects that showcase different text classification techniques and real-world applications. Each project demonstrates the complete machine learning pipeline from data preprocessing to model evaluation.

## Projects

### 1. Movie Genre Classifier
**Multi-Label Classification** | **F1-Score: 47.36%**

Predicts multiple movie genres from plot summaries using multi-label classification techniques.

#### Dataset
- **Source**: CMU Movie Summary Corpus
- **Size**: 41,793 movies (cleaned)
- **Features**: Movie ID, title, plot summary
- **Target**: 363 unique genres (multi-label)
- **Files**:
  - `movie_metadata.tsv` - 81,741 movies metadata
  - `plot_summaries.tsv` - 42,303 plot descriptions

#### Techniques & Implementation
- Text preprocessing: cleaning, normalization, stopword removal
- Porter Stemming for text normalization
- TF-IDF vectorization (10,000 features)
- Multi-label binarization for genre encoding
- Logistic Regression with OneVsRest strategy
- Threshold optimization (0.5 → 0.2) for improved performance

#### Results
- Initial F1-score: 32.45%
- Optimized F1-score: 47.36%
- Challenge: Complex multi-label combinations in movie genres

---

### 2. Fake News Detector
**Binary Classification** | **Accuracy: 93.63%**

Identifies fake news articles based on their titles using advanced NLP techniques.

#### Dataset
- **Source**: Kaggle Fake News Dataset
- **Size**: 18,285 articles (cleaned from 20,800)
- **Features**: Article ID, title, author, text content
- **Target**: Binary (0 = Real, 1 = Fake)
- **Distribution**: Balanced classes

#### Techniques & Implementation
- Text preprocessing: special character removal, lowercase conversion
- Porter Stemming for word normalization
- Bag of Words with CountVectorizer (5,000 features)
- N-grams (1,2,3) for phrase pattern capture
- Hyperparameter tuning with GridSearchCV

#### Model Comparison
| Model | Initial Accuracy | Best Accuracy | Precision | Recall |
|-------|-----------------|---------------|-----------|--------|
| Multinomial Naive Bayes | 90.16% | 90.59% | - | - |
| **Logistic Regression** | 93.52% | **93.63%** | 0.89 | 0.97 |

**Best Parameters**: C=0.8 for Logistic Regression, alpha=0.3 for Naive Bayes

---

### 3. Restaurant Review Sentiment Analysis
**Binary Classification** | **Accuracy: 78.5%**

Analyzes customer reviews to determine positive or negative sentiment about dining experiences.

#### Dataset
- **Source**: Restaurant_Reviews.tsv
- **Size**: 1,000 restaurant reviews
- **Features**: Review text
- **Target**: Binary (1 = Positive, 0 = Negative)
- **Format**: Tab-separated values

#### Techniques & Implementation
- Comprehensive text preprocessing pipeline
- Tokenization and stop words removal
- Porter Stemmer for root form reduction
- Bag of Words with 1,500 max features
- Multinomial Naive Bayes classifier
- Hyperparameter tuning (alpha: 0.1 to 1.0)

#### Results
- **Best Accuracy**: 78.5%
- **Precision**: 0.76
- **Recall**: 0.79
- **Best Alpha**: 0.2
- **Confusion Matrix**: TN:72, FP:25, FN:22, TP:81

#### Sample Usage
```python
sample_review = 'The food is really good here.'
predict_sentiment(sample_review)
# Output: POSITIVE review
```

---

### 4. SMS Spam Classifier
**Binary Classification** | **F1-Score: 99.4%**

High-performance spam detection system for SMS messages with engineered features.

#### Dataset
- **Source**: Spam SMS Collection
- **Size**: 5,572 messages (9,307 after oversampling)
- **Distribution**: 86.6% ham, 13.4% spam (originally imbalanced)
- **Format**: Tab-separated values

#### Feature Engineering
- **word_count**: Number of words per message
- **contains_currency_symbol**: Presence of £, $, ¥, €, ₹
- **contains_number**: Presence of numeric digits
- **TF-IDF**: 500 features max

#### Key Insights
- Spam messages typically contain 15-30 words
- 33% of spam contains currency symbols vs. rare in ham
- Most spam messages contain numbers

#### Model Performance
| Algorithm | F1-Score | Precision | Recall |
|-----------|----------|-----------|--------|
| Multinomial Naive Bayes | 94.3% | - | - |
| Decision Tree | 98.0% | - | - |
| **Random Forest** | **99.4%** | 99% | 99% |
| Voting Classifier | 98.0% | - | - |

---

### 5. Stock Sentiment Analysis
**Binary Classification** | **Financial NLP Application**

Predicts stock market movement based on financial news headlines sentiment.

#### Dataset
- **Source**: Daily stock market news headlines (CSV)
- **Features**: Multiple daily news headlines
- **Target**: Binary (0 = down/same, 1 = up)
- **Encoding**: ISO-8859-1
- **Size**: Several thousand rows (post-cleaning)

#### Workflow
1. **Data Exploration**: Structure analysis, class distribution visualization
2. **Preprocessing**: 
   - Combined multiple daily headlines
   - Tokenization, stopword removal, stemming/lemmatization
   - Text normalization
3. **Feature Engineering**:
   - Bag-of-Words representation
   - TF-IDF vectorization
4. **Model Training**:
   - Logistic Regression
   - Random Forest
   - Naïve Bayes
5. **Evaluation**:
   - Confusion Matrix
   - Classification Report
   - Cross-validation

#### Results
- Successfully captured sentiment signals from news
- Logistic Regression and Naïve Bayes showed strong baseline performance
- Demonstrates feasibility of news-based stock prediction

---

## Installation

### Prerequisites
- Python 3.6+
- Jupyter Notebook or Google Colab

### Setup Instructions

```bash
# Clone the repository
git clone <repository-url>
cd nlp-projects

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Requirements File
```txt
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
nltk>=3.6.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## Technologies Used

### Core Libraries
- **Pandas & NumPy**: Data manipulation and numerical operations
- **NLTK**: Natural language processing toolkit
- **Scikit-learn**: Machine learning algorithms and utilities
- **Matplotlib & Seaborn**: Data visualization

### NLP Techniques
- Text Preprocessing (cleaning, normalization)
- Tokenization and Stemming
- Stop Words Removal
- Bag of Words (BoW)
- TF-IDF Vectorization
- N-gram Feature Extraction

### Machine Learning Algorithms
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Decision Trees
- Voting Classifiers
- OneVsRest for Multi-label Classification

## How to Use

### Running Individual Projects

```python
# Example: Running the Fake News Classifier
jupyter notebook fake_news_classifier.ipynb

# Or in Python script
from fake_news_classifier import FakeNewsClassifier
classifier = FakeNewsClassifier()
classifier.train()
prediction = classifier.predict("Your news headline here")
```

### Project Structure
```
nlp-projects/
├── README.md
├── requirements.txt
├── data/
│   ├── movie_metadata.tsv
│   ├── plot_summaries.tsv
│   ├── Restaurant_Reviews.tsv
│   ├── spam_sms_collection.tsv
│   └── stock_news.csv
├── notebooks/
│   ├── 1_movie_genre_classifier.ipynb
│   ├── 2_fake_news_detector.ipynb
│   ├── 3_restaurant_sentiment.ipynb
│   ├── 4_sms_spam_classifier.ipynb
│   └── 5_stock_sentiment.ipynb
└── models/
    └── saved_models/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CMU for the Movie Summary Corpus
- Kaggle for various datasets
- NLTK and Scikit-learn communities for excellent documentation
- All contributors and researchers in the NLP field

---

**Note**: For detailed implementation and code, please refer to individual notebook files in the `notebooks/` directory.
