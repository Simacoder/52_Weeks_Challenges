import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (9, 5)
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["figure.dpi"] = 100

# Set Streamlit page config
st.set_page_config(layout="wide")
st.title("Netflix Data Exploration & African Trends")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/netflix_titles.csv")
    df["rating"].replace(np.nan, "TV-MA", inplace=True)
    df["country"].replace(np.nan, "United States", inplace=True)
    df.rename(columns={"listed_in": "genre"}, inplace=True)
    return df

netflix_df = load_data()

# Show Data Sample
if st.checkbox("Show Data Sample"):
    st.dataframe(netflix_df.head(10))

# Basic stats
st.subheader("Basic Stats")
st.write("**Movies vs TV Shows**")
st.bar_chart(netflix_df["type"].value_counts())

# Country analysis
st.subheader("Top Countries on Netflix")
top_countries = netflix_df["country"].value_counts().head(10)
st.bar_chart(top_countries)

# Top 5 in Africa
st.subheader("Top 5 African Countries on Netflix")
african_countries = [
    'Nigeria', 'South Africa', 'Egypt', 'Kenya', 'Morocco', 'Ghana',
    'Tunisia', 'Algeria', 'Uganda', 'Ethiopia'
]
africa_df = netflix_df[netflix_df['country'].isin(african_countries)]
top5_african = africa_df['country'].value_counts().iloc[:5].index

fig_africa, ax = plt.subplots()
sns.countplot(
    x='country',
    data=africa_df,
    hue='type',
    order=top5_african,
    palette='Set2',
    ax=ax
)
ax.set_title('Top 5 Netflix Country Distribution in Africa')
plt.xticks(rotation=45)
st.pyplot(fig_africa)

# Machine Learning: Predict next booming country
st.subheader("Predict the Next Booming African Country")
africa_df = africa_df.dropna(subset=['date_added'])
africa_df['country'] = africa_df['country'].str.strip()
africa_df['year_added'] = pd.to_datetime(africa_df['date_added']).dt.year

country_yearly = africa_df.groupby(['country', 'year_added']).size().reset_index(name='title_count')

from sklearn.linear_model import LinearRegression

predictions = []
for country in country_yearly['country'].unique():
    df_country = country_yearly[country_yearly['country'] == country]
    X = df_country['year_added'].values.reshape(-1, 1)
    y = df_country['title_count'].values

    if len(X) < 3:
        continue

    model = LinearRegression()
    model.fit(X, y)
    future_year = np.array([[2025]])
    predicted = model.predict(future_year)[0]
    predictions.append((country, predicted))

pred_df = pd.DataFrame(predictions, columns=['country', 'predicted_titles'])
pred_df = pred_df.sort_values(by='predicted_titles', ascending=False)

st.write("### Predicted Most Booming Country in 2025:")
st.success(f"{pred_df.iloc[0]['country']} with approx. {int(pred_df.iloc[0]['predicted_titles'])} titles")

# Optional - Show prediction table
if st.checkbox("Show all predictions"):
    st.dataframe(pred_df)

st.markdown("---")
st.caption("Created with ❤️ using Streamlit and Netflix Dataset")
