import nltk
import os
import pandas as pd
from nltk import word_tokenize, sent_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from gensim.models.keyedvectors import KeyedVectors
import random
import tensorflow as tf
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_DISABLE_MKL'] = '1'
import tensorflow as tf
from simple_elmo import ElmoModel


# Cargar datos de entrenamiento
print("reading data...")
df = pd.read_parquet("datos/train").drop_duplicates("text").iloc[0:100_000,:]# https://huggingface.co/datasets/clapAI/MultiLingualSentiment, train guardado en "datos/train" 
print(f"columnas: {df.columns}")
print(f"{df.shape[0]} filas  y {df.shape[1]} columnas")
print("ejemplo:")
print(df.iloc[0,:])


# Preproceso
idioms = {'en':'english'}#'de':'german','es':'spanish','fr':'french','it':'italian','pt':'portuguese','ru':'russian','tr':'turkish'}
def tokenize(row):
    text = row["text"]
    lang_code = row["language"]
    nltk_lang = idioms.get(lang_code)
    if nltk_lang:
        stopwords = set(nltk.corpus.stopwords.words(nltk_lang))
        tokens = [word for sentence in sent_tokenize(str(text).lower()) for word in word_tokenize(sentence,language=nltk_lang) if word.isalpha() and len(word) > 1]
        tokens = [x for x in tokens if x not in stopwords]
    else:
        tokens = []
    return " ".join(tokens)

# Creacion de textos tokenizados
if not os.path.isfile("datos/train_clean/train.csv"):
    corpus = df.apply(tokenize, axis=1)
    corpus = corpus[corpus != '']
    corpus.to_csv("datos/train_clean/train.csv")
else:
    corpus = pd.read_csv("datos/train_clean/train.csv",index_col=0)
    corpus = corpus.iloc[:,0]
    corpus.name = None

reduced_corpus = corpus[0:500]
# Creacion de modelos TF-IDF y BoW
def create_and_save_model(model,pathv,pathm):
    print(os.path.isfile(pathv))
    if not os.path.isfile(pathv):
        print(f"creating and training {path}...")
        vectorizer = model(min_df=1)
        print(f"saving {path}...")
        matrix = vectorizer.fit_transform(reduced_corpus)
        with open(pathv,"wb") as file:
            pickle.dump(vectorizer,file)
        np.save(pathm,matrix.toarray())
        return matrix
    else:
        with open(pathv,"rb") as file:
            vectorizer = pickle.load(file)
        matrix = np.load(pathm)
        return matrix
    

path = "datos/model/"
pathtvfidf = path + "tfidf_vectorizer.pkl"
pathtmfidf = path + "tfidf_matrix.npy"
pathvbow = path + "bow_vectorizer.pkl"
pathmbow = path + "bow_matrix.npy"

matrixtfidf = create_and_save_model(TfidfVectorizer,pathtvfidf,pathtmfidf)
matrixbow = create_and_save_model(CountVectorizer,pathvbow,pathmbow)

# Creacion del modelo LDiA
print("creating and training LDiA...")
if not os.path.isfile("datos/model/ldia_vectorizer.pkl"):
    ldia = LDiA(n_components=7, learning_method='online',n_jobs=1,batch_size=128,max_iter=10,random_state=RANDOM_SEED)
    ldia = ldia.fit(matrixbow.astype('float32'))
    with open('datos/model/ldia_vectorizer.pkl','wb') as file:
        pickle.dump(ldia,file)
else:    
    with open('datos/model/ldia_vectorizer.pkl','rb') as file:
        ldia = pickle.load(file)

# Creacion del modelo word2vec
print("loading word2vec...")
word_vectors = KeyedVectors.load_word2vec_format('datos/model/GoogleNews-vectors-negative300.bin.gz', binary=True,limit=None)
if not os.path.isfile("datos/model/word2vec_matrix.csv"):
    text_vectors = []
    for text in corpus:
        existing_word_vectors = [word_vectors[word] for word in text.split(" ") if word in word_vectors]
        
        if existing_word_vectors:
            mean_word_vectors = np.mean(existing_word_vectors, axis=0)
        else:
            mean_word_vectors = np.zeros(word_vectors.vector_size)
        text_vectors.append(mean_word_vectors)
    df_text_vectors = pd.DataFrame(text_vectors,index=corpus.index)
    df_text_vectors.to_csv("datos/model/word2vec_matrix.csv")
else:
    df_text_vectors = pd.read_csv("datos/model/word2vec_matrix.csv",index_col=0)


# Creacion del modelo ELMo
print("loading ELMo...")
if not os.path.isfile("datos/model/elmo_matrix.csv"):
    model = ElmoModel()
    model.load("datos/model/209/")# http://vectors.nlpl.eu/repository/20/209.zip, lemmatization falso para modelos avanzados como elmo
    frases = [texto.split() for texto in reduced_corpus.tolist()]
    elmo_matrix = pd.DataFrame(model.get_elmo_vector_average(frases))
    elmo_matrix.to_csv("datos/model/elmo_matrix.csv")
else:
    elmo_matrix = pd.read_csv("datos/model/elmo_matrix.csv",index_col=0)
elmo_matrix.index = reduced_corpus.index

# Analisis, crear word2vec?
vocabulario_corpus = set([palabra for text in corpus for palabra in text.split()])
vocabulario_word2vec = word_vectors.index_to_key
palabras_oov = [p for p in vocabulario_corpus if p not in vocabulario_word2vec]
tasa_oov_unicas = (len(palabras_oov) / len(vocabulario_corpus)) * 100

print(f"--- REPORTE DE PALABRAS NO REPRESENTABLES ---")
print(f"# de palabras únicas en el corpus: {len(vocabulario_corpus)}")
print(f"# de palabras unicas en el modelo: {len(vocabulario_word2vec)}")
print(f"# de palabras únicas OOV (No representables): {len(palabras_oov)}")
print(f"% de palabras únicas sin vector: {tasa_oov_unicas:.2f}%")
#print(f"Lista de palabras desconocidas para el modelo: {palabras_oov}")