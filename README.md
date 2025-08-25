# 📰 Day 13: Topic Modeling on News Headlines

This is the thirteenth project of my #30DaysOfAI challenge. This project explores **Unsupervised Learning** to automatically discover hidden topics from a large collection of news headlines.

### ✨ Key Concepts
* **Unsupervised Learning:** The model learns patterns from data without any explicit labels. The goal is to discover the underlying structure of the data.
* **Topic Modeling:** An unsupervised technique used to find abstract "topics" that occur in a collection of documents.
* **Latent Dirichlet Allocation (LDA):** A popular generative statistical model for topic modeling. It treats documents as mixtures of topics, and topics as mixtures of words.

### 💻 Tech Stack
- Python, Pandas, Scikit-learn, Streamlit

### 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/KULLANICI_ADIN/Day13-Topic-Modeling-News.git](https://github.com/KULLANICI_ADIN/Day13-Topic-Modeling-News.git)
    cd Day13-Topic-Modeling-News
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Process the data and train the model:**
    *Note: This script requires the `abcnews-date-text.csv` file from [this Kaggle dataset](https://www.kaggle.com/datasets/therohk/a-million-news-headlines) to be in the root directory. The file is ignored by git.*
    ```bash
    python process_data.py
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
