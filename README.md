<div align="center" id="top">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B1020,35:21164B,70:5B21B6,100:38BDF8&height=230&section=header&text=SARCASM%20LENS&fontSize=58&fontColor=FFFFFF&animation=twinkling&fontAlignY=36&desc=READ%20BETWEEN%20THE%20REVIEWS&descSize=18&descAlignY=58" width="100%" alt="Sarcasm Lens futuristic animated header"/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=15&duration=2600&pause=850&color=A78BFA&center=true&vCenter=true&width=720&lines=Decoding+the+hidden+meaning+of+reviews...;Detecting+sarcasm+through+NLP...;Comparing+6+machine+learning+models...;Teaching+machines+to+read+between+the+lines." alt="Animated project tagline"/>

<br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/PYTHON-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/></a>
<a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/MACHINE_LEARNING-6_MODELS-7C3AED?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Six machine learning models"/></a>
<a href="https://www.nltk.org/"><img src="https://img.shields.io/badge/NLP-SENTIMENT_ANALYSIS-EC4899?style=for-the-badge" alt="NLP sentiment analysis"/></a>
<a href="https://jupyter.org/"><img src="https://img.shields.io/badge/JUPYTER-NOTEBOOK-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook"/></a>

<br/><br/>

<img src="https://img.shields.io/badge/STATUS-COMPLETED-22C55E?style=flat-square" alt="Project status completed"/>
<img src="https://img.shields.io/badge/BUILT_WITH-CURIOSITY_%26_COFFEE-BA8B5A?style=flat-square" alt="Built with curiosity and coffee"/>

<br/><br/>

[ **Overview** ](#-the-intelligence-behind-the-words) ·
[ **Models** ](#-the-model-laboratory) ·
[ **Results** ](#-performance-matrix) ·
[ **Setup** ](#-get-started) ·
[ **Team** ](#-the-minds-behind-the-machine)

</div>

---

## ✦ THE INTELLIGENCE BEHIND THE WORDS

> **A review can say one thing and mean another.**

“Great, it broke on day one.”  
Is that positive? Definitely not. 🙃

**Sarcasm Lens** is an NLP and machine-learning project that explores how models classify sentiment in text, including reviews containing sarcasm and emojis.

The project uses the IMDb movie-review dataset as a benchmark and a separately annotated custom dataset to investigate how models perform across normal, sarcastic, emoji-only, and mixed reviews.

<div align="center">

| 🎬 IMDb benchmark | 🎭 Sarcasm-aware data | 💬 Emoji-aware features |
|:---:|:---:|:---:|
| **50,000** reviews | **1,168** annotated reviews | **4** review categories |

</div>

The goal isn't just to classify words as positive or negative. It's to explore how well different models interpret the sentiment behind a review.

## ⚡ WHAT MAKES IT INTERESTING?

- **Six model architectures** — compare traditional machine learning with Word2Vec-based approaches and a neural sequence model.
- **Sarcasm-aware experimentation** — evaluate sarcastic and non-sarcastic reviews separately.
- **Emoji and linguistic features** — explore emoji sentiment, punctuation, capitalisation, sarcasm cues, and VADER sentiment contrast.
- **Cross-dataset testing** — train models on IMDb and test their ability to transfer to the custom reviews.
- **Leakage-aware evaluation** — keep near-duplicate review families out of both training and testing at the same time.
- **Local prediction** — run the final saved model without an external inference API.

---

## 🧬 THE MODEL LABORATORY

Six candidates are trained and compared on the custom dataset using the same training and held-out test partition.

| # | Model | What it explores |
|:--:|---|---|
| 01 | **TF-IDF + Logistic Regression** | A strong lexical-frequency baseline |
| 02 | **TF-IDF + Linear SVM** | Maximum-margin text classification |
| 03 | **TF-IDF + Random Forest** | An ensemble of decision trees |
| 04 | **Word2Vec + SVM** | Semantic word embeddings with SVM |
| 05 | **Word2Vec + handcrafted features + Logistic Regression** | Embeddings combined with linguistic and emoji signals |
| 06 | **Word2Vec + BiGRU + Attention** | A sequence model that preserves word order and uses attention |

### 🔍 Feature engineering

The handcrafted feature set contains eight features covering:

- Emoji count and positive/negative emoji signals
- Presence of irony-related emojis
- Exclamation and question mark counts
- Uppercase-letter ratio
- Sarcasm phrase indicators
- VADER sentiment contrast between sentence parts

The feature set is experimental and uses simple heuristics; it is not a complete sarcasm-understanding system.

### 🗺️ Experiment pipeline

```mermaid
flowchart TD
    A["🎬 IMDb benchmark + custom reviews"] --> B["Text preprocessing"]
    B --> C["Feature extraction"]
    C --> D["TF-IDF"]
    C --> E["Word2Vec"]
    C --> F["Handcrafted linguistic & emoji features"]
    D --> G["Logistic Regression"]
    D --> H["Linear SVM"]
    D --> I["Random Forest"]
    E --> J["Word2Vec + SVM"]
    E --> K["Word2Vec + features + Logistic Regression"]
    E --> L["BiGRU + Attention"]
    G --> M["Evaluation & comparison"]
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M
    M --> N["Sentiment prediction"]
```

---

## 🧪 THE DATA BEHIND THE EXPERIMENT

### 01 / IMDb movie reviews

- **50,000** labelled movie reviews.
- Balanced positive and negative sentiment classes.
- Used as the conventional sentiment-analysis benchmark.
- A stratified 65:35 train/test split gives 32,500 training reviews and 17,500 test reviews.

### 02 / Custom annotated review dataset

- **1,168** reviews.
- Four categories, with 292 reviews in each category.
- Both positive and negative sentiment examples are represented in every category.
- Includes sentiment, sarcasm, emoji, source-group, and split annotations.

| Category | What it contains | Example |
|---|---|---|
| 🟢 Normal | No sarcasm or emoji | `The story was beautifully written.` |
| 🙄 Sarcastic | Sarcasm annotation | `Great, it broke on day one.` |
| 😍 Emoji | Emoji without sarcasm | `Such a lovely film ❤️` |
| 🎭 Mix | Sarcasm and emoji | `Oh great, another delay 🙄` |

> **Important:** the sentiment label describes the meaning of the complete review. Sarcasm is a separate annotation, not the sentiment label itself.

---

## 🛡️ EVALUATION WITHOUT DATA LEAKAGE

> A perfect score is not always a perfect model.

During early experiments, random row-level splitting produced suspiciously high scores because variations of the same base reviews could appear in both training and testing.

That is **data leakage**: the model may memorise similar examples instead of learning patterns that generalise.

### What changed?

- Reviews are assigned to source groups / near-duplicate families.
- Entire families are kept on one side of the train/test split.
- The notebook checks that no source group overlaps between training and testing.
- The held-out custom test set contains **234 reviews**.
- The measured share of test reviews with a near-copy in training (similarity ≥ 0.8) is **0%**.
- Grouped 5-fold cross-validation is used as an additional sanity check.

This makes the evaluation more meaningful, although the custom dataset is still relatively small.

---

## 📊 PERFORMANCE MATRIX

### IMDb benchmark — 17,500 held-out reviews

| Model | Accuracy | F1-score |
|---|:---:|:---:|
| 🥇 **TF-IDF + Logistic Regression** | **88.78%** | **0.8893** |
| TF-IDF + Linear SVM | 88.71% | 0.8881 |
| Word2Vec + BiGRU + Attention | 87.90% | 0.8849 |
| TF-IDF + Random Forest | 84.92% | 0.8490 |
| Word2Vec + features + Logistic Regression | 84.46% | 0.8454 |
| Word2Vec + SVM | 84.06% | 0.8416 |

### Custom dataset — 234 held-out reviews

| Model | Accuracy | F1-score |
|---|:---:|:---:|
| 🥇 **TF-IDF + Logistic Regression** | **86.32%** | **0.8710** |
| TF-IDF + Random Forest | 81.62% | 0.8072 |
| Word2Vec + BiGRU + Attention | 80.77% | 0.8018 |
| TF-IDF + Linear SVM | 76.92% | 0.7805 |
| Word2Vec + features + Logistic Regression | 65.81% | 0.6800 |
| Word2Vec + SVM | 50.85% | 0.6102 |

### Sarcastic vs non-sarcastic performance

The best overall model on the held-out custom test set was **TF-IDF + Logistic Regression**.

| Review group | Accuracy |
|---|:---:|
| Sarcastic (118 reviews) | **88.98%** |
| Non-sarcastic (116 reviews) | **83.62%** |
| Overall (234 reviews) | **86.32%** |

<details>
<summary><strong>📈 Grouped cross-validation results</strong></summary>

The 5-fold grouped cross-validation results for the TF-IDF baselines were:

| Model | Mean accuracy | Standard deviation |
|---|:---:|:---:|
| TF-IDF + Logistic Regression | 73.43% | ±8.71% |
| TF-IDF + Linear SVM | 75.17% | ±5.17% |

The single held-out test result should not be treated as a guaranteed estimate of real-world performance. The cross-validation results show that performance varies across unseen groups.

</details>

<details>
<summary><strong>🌍 Cross-dataset experiment: IMDb-trained models tested on custom reviews</strong></summary>

All six models were trained on IMDb and then evaluated on the custom reviews without retraining on the custom dataset.

| Model | Accuracy |
|---|:---:|
| TF-IDF + Linear SVM | **66.44%** |
| TF-IDF + Logistic Regression | 65.75% |
| Word2Vec + SVM | 65.50% |
| Word2Vec + features + Logistic Regression | 65.50% |
| Word2Vec + BiGRU + Attention | 62.16% |
| TF-IDF + Random Forest | 54.97% |

This is a separate transfer experiment from the custom-dataset model comparison above.

</details>

---

## 🧰 THE NEURAL TOOLKIT

<div align="center">

<em>Tools that power Sarcasm Lens — tap a sticker to explore its documentation.</em>

<br/><br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
<a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/></a>
<a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"/></a>
<a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/></a>

<br/>

<a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit-learn"/></a>
<a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/></a>
<a href="https://keras.io/"><img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/></a>
<a href="https://radimrehurek.com/gensim/"><img src="https://img.shields.io/badge/Gensim-Word2Vec-7C3AED?style=for-the-badge" alt="Gensim Word2Vec"/></a>

<br/>

<a href="https://www.nltk.org/"><img src="https://img.shields.io/badge/NLTK-85BDA6?style=for-the-badge" alt="NLTK"/></a>
<a href="https://github.com/cjhutto/vaderSentiment"><img src="https://img.shields.io/badge/VADER-Sentiment-EC4899?style=for-the-badge" alt="VADER"/></a>
<a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge" alt="Matplotlib"/></a>
<a href="https://seaborn.pydata.org/"><img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge" alt="Seaborn"/></a>

</div>

---

## 🚀 GET STARTED

### 01 / Clone this repository

```bash
git clone https://github.com/YOUR-USERNAME/sarcasm-lens.git
cd sarcasm-lens
```

### 02 / Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud openpyxl emoji vaderSentiment gensim tensorflow
```

### 03 / Add the datasets

Keep these files in the project directory expected by the notebook:

- `IMDB Dataset.csv`
- `own_dataset.xlsx`

### 04 / Launch the notebook

```bash
jupyter notebook IMDB_Sentiment_Analysis.ipynb
```

Run the notebook cells from top to bottom.

> ⏳ Training the IMDb BiGRU model can take several minutes or longer depending on your system and available hardware.

---

## 💬 TRY THE SENTIMENT PREDICTOR

The notebook includes an interactive prediction section.

Enter a review and the final saved **Word2Vec + BiGRU + Attention** model returns:

- Predicted sentiment: Positive or Negative
- Prediction confidence
- A simple sarcasm-cue indicator

**Example input**

```text
Great, another movie that wasted two hours of my life 🙄
```

The final predictor uses the saved model `final_word2vec_bigru.keras` and preprocessing bundle `final_word2vec_bigru_bundle.pkl`. Prediction runs locally; no external inference API is used.

> **Note:** the final interactive predictor is the BiGRU model. The best overall model in the custom held-out comparison is TF-IDF + Logistic Regression. These are different roles, so they should not be confused.

---

## ⚠️ LIMITATIONS & HONEST NOTES

- The custom dataset contains 1,168 reviews, which is limited for broad real-world conclusions.
- The initial 768 reviews were generated from 64 base sentences with variations. The dataset was expanded with 400 individually authored reviews, and family-aware splitting is used to reduce leakage.
- Sarcasm depends on context, culture, and tone; simple phrase and emoji cues cannot capture every case.
- Word2Vec is trained on the available project reviews, so its vocabulary may not generalise well to unfamiliar domains.
- Averaging word vectors loses word order and some contextual information.
- The emoji and handcrafted features did not consistently improve model accuracy.
- The custom test set is small, so results should be interpreted cautiously.

---

## 🌱 WHAT THIS PROJECT HELPED EXPLORE

- Text preprocessing and NLP workflows
- TF-IDF and word embeddings
- Traditional ML model comparison
- Feature engineering for text
- Sequence modelling with BiGRU and Attention
- Cross-dataset evaluation
- Data leakage and group-aware validation
- Model saving and local inference

---

## 👩‍🚀 THE MINDS BEHIND THE MACHINE

<div align="center">

| Team Members | GitHub |
|:---|:---:|
| **Shibani Mishra** | <a href="https://github.com/Shibani-4331"><img src="https://img.shields.io/badge/GitHub-SHIBANI-white?style=for-the-badge&logo=github&logoColor=181717&labelColor=555555" width="220" height="35" alt="Shibani GitHub"/></a> |
| **Ashutosh Mohanty** | <a href="https://github.com/Ashutosh-1505"><img src="https://img.shields.io/badge/GitHub-ASHUTOSH-white?style=for-the-badge&logo=github&logoColor=181717&labelColor=555555" width="220" height="35" alt="Ashutosh GitHub"/></a> |
| **Deepakshi Nayak** | <a href="https://github.com/DeepakshiiiGit"><img src="https://img.shields.io/badge/GitHub-DEEPAKSHI-white?style=for-the-badge&logo=github&logoColor=181717&labelColor=555555" width="220" height="35" alt="Deepakshi GitHub"/></a> |
| **Smruti Ranjan Nayak** | <a href="https://github.com/Smruti-1163"><img src="https://img.shields.io/badge/GitHub-SMRUTI-white?style=for-the-badge&logo=github&logoColor=181717&labelColor=555555" width="220" height="35" alt="Smruti GitHub"/></a> |

</div>

---

<div align="center">

### ✦ EVERY REVIEW HAS A STORY.

**Let's teach machines to read between the lines.**

<a href="#top"><img src="https://img.shields.io/badge/%E2%AC%86%20BACK%20TO%20TOP-7C3AED?style=for-the-badge&logo=github&logoColor=white" alt="Back to top"/></a>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=500&size=14&duration=3200&pause=1000&color=A78BFA&center=true&vCenter=true&width=650&lines=Built+with+Python%2C+curiosity%2C+and+sarcasm.;Thanks+for+visiting+Sarcasm+Lens+%E2%9C%A8;Leave+a+star+if+you+liked+the+project+%E2%AD%90" alt="Animated footer message"/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:38BDF8,50:5B21B6,100:0B1020&height=110&section=footer" width="100%" alt="Animated purple and blue footer"/>

</div>
