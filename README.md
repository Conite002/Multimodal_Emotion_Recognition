### **📌 Introduction - Multimodal Emotional Recognition (MER)**  

Emotions play a **crucial role** in human interactions, influencing communication, decision-making, and social connections. The ability to **automatically recognize emotions** from various data sources—such as **speech, facial expressions, and textual content**—is fundamental for applications in **human-computer interaction, mental health monitoring, and affective computing**.  

The **Multimodal Emotional Recognition (MER) project** leverages **Machine Learning (ML), Deep Learning (DL), and Graph Neural Networks (GNNs)** to develop **robust models capable of accurately detecting and classifying emotional states** across multiple modalities:  

- 🎤 **Speech (Audio)** – Extracting prosodic and spectral features for vocal emotion detection.  
- 🎭 **Facial Expressions (Video)** – Analyzing facial micro-expressions and visual cues using CNNs.  
- 📝 **Linguistic Cues (Text)** – Processing sentiment, semantics, and tone from transcriptions.  
- 🔗 **Graph-Based Speaker Relationships** – Capturing conversational dependencies using GNNs.  

Unlike unimodal emotion recognition, **MER combines multiple modalities** to improve classification accuracy and **handle challenges such as noisy data, speaker variations, and contextual ambiguity**. The integration of **graph-based learning** further enhances the model’s ability to **leverage long-distance dependencies** in conversations, providing deeper contextual awareness.  

This project aims to push the boundaries of **affective computing** by exploring:  
✅ **Multimodal Fusion Techniques** – Early, late, and hybrid fusion strategies.  
✅ **Deep Learning Architectures** – BiGRU, Transformers, and CNN-based models.  
✅ **Graph-Based Emotion Modeling** – Graph Attention Networks (GAT) & Relational GCNs (R-GCN).  
✅ **Handling Class Imbalance** – Weighted Random Sampling, Focal Loss, and Data Augmentation.  






---

### **Feature Extraction**
- **Audio**: Wav2vec, MFCCs, Spectrograms, Prosodic Features (`Librosa`)
- **Text**: Word Embeddings (`BERT`, `GloVe`), Sentiment Analysis (`NLTK`, `SpaCy`)
- **Video**: Video Transformers (ViT), CNN-based embeddings (`VGGFace`, `OpenCV`)
- **Graph Representation**: Speaker Dependency Graphs (`GNN`, `R-GCN`)

### **Multimodal Deep Learning Architectures**
- **BiGRU-Based Feature Extraction** for text, audio, and video.
- **Self-Attention & Transformer Encoders** for linguistic emotion recognition.
- **Graph Neural Networks (GNNs)** for speaker relationship modeling.
- **Gated Fusion Networks** for adaptive late fusion of multiple modalities.

### **Graph-Based Emotion Recognition**
- **Graph Construction**: Speaker relation graphs from conversations.
- **Graph Neural Networks (GNNs)**: `R-GCN`, `GAT`, and `GFPush` for context propagation.
- **DropNodes**: Dynamic graph sparsification to optimize performance.

### **Handling Class Imbalance**
- **Weighted Random Sampler**: Dynamically balances minority and majority classes.
- **Focal Loss**: Adjusts model learning focus on rare emotions.
- **Data Augmentation**: Enhancing speech/text dataset diversity.

### **Model Training & Evaluation**
- **Custom Weighted Loss Functions** for improved balancing.
- **Evaluation Metrics**: Precision, Recall, F1-score, and Confusion Matrices.


---

## **Installation**
To set up the development environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Conite002/MER-project.git
cd MER-project

# Create a virtual environment
python -m venv mer-env
source mer-env/bin/activate  # On Windows use: mer-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## **📌 Training & Running the Model**
### **Preprocess Data & Create Data Loaders**


### **Evaluate the Model**

---

## **📌 Results & Observations**
| **Approach** | **Val Accuracy** | **Precision** | **Recall** | **F1 Score** |

### **Key Insights:**
- **Adding a graph model (`GRNN`) improves class differentiation**, but needs better fine-tuning.
- **The `WeightedRandomSampler` alone reduces performance**, as it disrupts the balance in learning.
- **Future improvements can combine specialized classifiers for underrepresented classes.**

---

## **📌 Future Improvements**
1. **Fine-tune Graph Weights (`graph_output * λ`)** to avoid dominating multimodal features.
2. **Train a Secondary Classifier for `Classes 3, 4, 5`** and use `Ensemble Learning` to merge predictions.
3. **Use Focal Loss** to better optimize the training for minority classes.
4. **Explore Contrastive Learning** for better feature separation in multimodal embeddings.

---

## **📌 Contributing**
If you'd like to contribute:
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

---

## **📌 Contact**
For questions, reach out to:
- **[Conité GBODOGBE](mailto:dsconite@gmail.com)**
- **GitHub Issues:** [Report an Issue](https://github.com/Conite002/MER-project/issues)

---

💡 **This README is now fully updated, clear, and well-structured. Let me know if you need additional modifications! 🚀**