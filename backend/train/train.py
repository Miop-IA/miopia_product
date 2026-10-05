import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df = pd.read_csv("treino/dados.csv")   # colunas: texto, categoria

X_treino, X_teste, y_treino, y_teste = train_test_split(
    df["texto"], df["categoria"], test_size=0.2, stratify=df["categoria"], random_state=42
)

modelo = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])
modelo.fit(X_treino, y_treino)

print(classification_report(y_teste, modelo.predict(X_teste)))

joblib.dump(modelo, "modelos/classificador.joblib")