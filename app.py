import streamlit as st
import joblib

st.set_page_config(
    page_title="Topic Modeling on News Headlines",
    page_icon="📰",
    layout="wide"
)

@st.cache_resource
def load_resources():
    """Loads the LDA model and the vectorizer from disk."""
    try:
        lda_model = joblib.load('lda_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return lda_model, vectorizer
    except FileNotFoundError:
        return None, None

lda_model, vectorizer = load_resources()

st.title("📰 Haber Başlıklarından Gizli Konuları Keşfetme")

if lda_model is None or vectorizer is None:
    st.error(
        "Gerekli model dosyaları (.pkl) bulunamadı. "
        "Lütfen önce `python process_data.py` komutunu çalıştırdığınızdan emin olun."
    )
    st.stop()

tab1, tab2 = st.tabs(["🔍 **Keşfedilen Konular**", "🎯 **Proje Detayları**"])

with tab1:
    st.header("Yapay Zekanın Keşfettiği 10 Ana Konu")
    st.write(
        "LDA modeli, 20,000 haber başlığını analiz ederek aşağıdaki 10 gizli konuyu ve "
        "bu konuları en iyi tanımlayan anahtar kelimeleri tamamen kendi başına buldu."
    )
    st.markdown("---")

    feature_names = vectorizer.get_feature_names_out()
    
    cols = st.columns(5)
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-10 - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        
        with cols[topic_idx % 5]:
            st.subheader(f"Konu #{topic_idx + 1}")
            st.info(", ".join(top_words))

with tab2:
    st.header("Projenin Amacı ve Teknik Detaylar")
    st.markdown("""
    Bu projenin amacı, büyük bir metin koleksiyonu içindeki soyut "konuları" otomatik olarak keşfeden bir **Konu Modelleme (Topic Modeling)** sistemi oluşturmaktır.
    
    - **Yöntem:** Bu bir **Gözetimsiz Öğrenme (Unsupervised Learning)** projesidir. Modele "bu başlık sporla ilgili" gibi etiketler vermedik. Model, kelimelerin bir arada bulunma sıklıklarına bakarak bu grupları tamamen kendi başına keşfetti.
    
    - **Model:** **LDA (Latent Dirichlet Allocation)**, konu modellemesi için en yaygın kullanılan olasılıksal modellerden biridir. Temel varsayımı, her dokümanın birden çok konunun bir karışımı olduğu ve her konunun da belirli kelimelerin bir dağılımı olduğudur.
    
    - **Önemli Teknik Detay:** Metinleri modele vermeden önce, kelimelerin sayısal bir temsilini oluşturmak için **CountVectorizer** kullanılmıştır. Bu, her kelimenin bir belgede kaç kez geçtiğini sayarak bir "belge-terim matrisi" oluşturur. Bu matris, LDA modelinin girdisidir.
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*0fUaG3pS2T133IpHs3G2Zw.png", caption="LDA Modelinin Kavramsal Çalışması")