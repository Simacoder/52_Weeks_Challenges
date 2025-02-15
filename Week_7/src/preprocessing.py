import pandas as pd

def preprocess_data(input_path, output_path):
    # Laod the raw sales data
    df = pd.read_csv(input_path, encoding='windows-1252')

    # droping missing values
    df.dropna(inplace=True)

    # Feature engineering: moving average of sales
    df['sales_moving_avg'] = df['Sales'].rolling(window=7).mean()

    # Normalize sales data
    df['Sales'] = (df['Sales'] - df['Sales'].mean()) / df['Sales'].std()

    # Saved processed data
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data('data/raw/sales_data.csv', 'data/processed/cleaned_sales_data.csv')