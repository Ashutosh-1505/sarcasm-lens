<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=230&section=header&text=Sentiment%20Analysis&fontSize=62&fontAlignY=38&desc=Sarcasm%20%26%20Emoji-Aware%20Evaluation%20on%20IMDb%20%2B%20a%20Custom%20Dataset&descAlignY=60&descSize=18&animation=fadeIn" width="100%" alt="header"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3200&pause=900&color=A78BFA&center=true&vCenter=true&width=700&lines=Six+models.+Two+datasets.+One+honest+evaluation.;TF-IDF+%E2%9A%94%EF%B8%8F+Word2Vec+%E2%9A%94%EF%B8%8F+BiGRU+%2B+Attention;Can+machines+tell+when+%22Great%2C+it+broke+on+day+one%22+is+negative%3F+%F0%9F%99%84;Leakage-safe+splits.+Realistic+scores.+No+fake+1.0." alt="Typing animation"/>
</a>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Gensim](https://img.shields.io/badge/Gensim-Word2Vec-7C3AED?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## ✨ About the Project

This project asks a simple question with a tricky answer: **how well do classic and neural sentiment models cope with sarcasm and emoji?**

We train and compare **six models** on the standard **IMDb** movie-review benchmark, and on a **custom, annotated review dataset** built to include four kinds of text:

| Category | Sarcasm | Emoji | Example |
|:--|:--:|:--:|:--|
| 🟢 **Normal** | ✗ | ✗ | *The charger stopped working after two days.* |
| 🙄 **Sarcastic** | ✓ | ✗ | *Great, it broke on day one.* |
| 😍 **Emoji** | ✗ | ✓ | *Such a cozy cafe, the lattes are amazing ☕❤️* |
| 🎭 **Mix** | ✓ | ✓ | *Oh great, another delay, my favorite part of any trip 🙄✈️* |

Everything runs **locally** with open-source libraries. No external API or hosted model is used.

---

## 🧠 The Six Models

| # | Model | Idea |
|:-:|:--|:--|
| 1 | **TF-IDF + Logistic Regression** | Word-frequency baseline |
| 2 | **TF-IDF + Linear SVM** | Maximum-margin linear classifier |
| 3 | **TF-IDF + Random Forest** | Tree ensemble on sparse features |
| 4 | **Word2Vec + SVM** | Averaged word embeddings |
| 5 | **Word2Vec + handcrafted features + LR** | Embeddings plus emoji, punctuation, caps, sarcasm-cue and VADER-contrast features |
| 6 | **Word2Vec + BiGRU + Attention** | Sequence model that keeps word order |

---

## 🔬 What Makes This Evaluation Honest

Our first run scored a perfect **1.0000** on several models. That was a red flag, not a victory.

> 🚨 **The cause:** the custom dataset was built from a small set of base sentences repeated with different openers, closers and emoji. A random split put near-identical reviews in both train and test, so the models simply memorised them.

**The fix**

- 🧬 Reviews are grouped into **near-duplicate families**, and whole families go to either train or test.
- 🔒 An assertion checks that **no family appears on both sides**.
- 📏 The notebook reports how similar test reviews are to their closest training review (**0% near-copies**).
- 🧪 Grouped **5-fold cross-validation** is added as a sanity check.
- 📈 The dataset was **expanded to 1,168 reviews** (364 families) with individually written reviews.

---

## 📊 Results

### IMDb benchmark (17,500 held-out reviews)

| Model | Accuracy | F1 |
|:--|:-:|:-:|
| TF-IDF + LR | **0.8878** | 0.8893 |
| TF-IDF + SVM | 0.8871 | 0.8881 |
| Word2Vec + BiGRU + Attention | 0.8790 | 0.8849 |
| TF-IDF + RF | 0.8492 | 0.8490 |
| Word2Vec + features + LR | 0.8446 | 0.8454 |
| Word2Vec + SVM | 0.8406 | 0.8416 |

### Custom dataset (234 held-out reviews from unseen families)

| Model | Overall | Sarcastic | Non-sarcastic |
|:--|:-:|:-:|:-:|
| TF-IDF + LR | **0.863** | 0.890 | 0.836 |
| TF-IDF + RF | 0.816 | 0.805 | 0.828 |
| Word2Vec + BiGRU + Attention | 0.808 | 0.797 | 0.819 |
| TF-IDF + SVM | 0.769 | 0.771 | 0.767 |
| Word2Vec + features + LR | 0.658 | 0.695 | 0.621 |
| Word2Vec + SVM | 0.509 | 0.559 | 0.457 |

> 💡 **Read with care:** with 234 test rows the margin of error is roughly ±5 points. Grouped cross-validation for TF-IDF LR gives **0.734 ± 0.087**, so treat the top three models as close rather than clearly ranked.

---

## 🗂️ Repository Structure

```
📦 project
 ┣ 📓 IMDB_Sentiment_Analysis_Six_Models_FIXED_1.ipynb   # main notebook
 ┣ 📄 IMDB Dataset.csv                                   # 50,000 IMDb reviews
 ┣ 📄 own_dataset.xlsx                                   # custom annotated dataset
 ┗ 📄 README.md
```

`own_dataset.xlsx` contains the review text, `true_sentiment`, `is_sarcastic`, `has_emoji`, `category`, a `source_group` family id, a `train`/`test` `split`, and an `origin` column marking original versus expanded rows.

---

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone <your-repo-link>
cd <your-repo-folder>
```

**2. Install the dependencies**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud openpyxl emoji vaderSentiment gensim tensorflow
```

**3. Run the notebook**

Keep `IMDB Dataset.csv` and `own_dataset.xlsx` in the same folder as the notebook, then run it from top to bottom.

> ⏱️ The IMDb BiGRU section is the slowest part and can take around 20 minutes on CPU.

---

## ⚠️ Limitations

- 🧪 The custom dataset is **synthetic and manually authored**, not collected from real users. It is a controlled benchmark, so independent human-written reviews would be needed for real-world claims.
- 🎭 Sarcasm is subjective and context-dependent. Our handcrafted cues are simple heuristics.
- 🔤 Word2Vec is trained only on the project's own reviews, so its vocabulary is small and its embeddings are limited.
- 📉 The emoji features did not clearly improve accuracy on this dataset.

---

## 👩‍💻 Meet the Team

<div align="center">

| 👤 Contributor | 🛠️ Role | 🔗 GitHub |
|:--|:--|:--|
| **Shibani Mishra** | Contributor | _add GitHub link_ |
| **Deepakshi Nayak** | Contributor | _add GitHub link_ |
| **Smruti Ranjan Nayak** | Contributor | _add GitHub link_ |

</div>

---

<div align="center">

### ⭐ If you found this useful, consider starring the repo!

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&duration=4000&pause=1000&color=94A3B8&center=true&vCenter=true&width=520&lines=Built+with+curiosity%2C+Python%2C+and+a+lot+of+coffee+%E2%98%95" alt="footer typing"/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,24&height=110&section=footer" width="100%" alt="footer"/>

</div>
