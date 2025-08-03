from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dotenv
import os
dotenv.load_dotenv()
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

# setup the .env to reference the api secret key
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

df = pd.read_csv("data/moderation_dataset.csv")
df.head(15)
print(df)