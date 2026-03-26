import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('movies.csv')

print("First 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())


drop_cols = ['id', 'title', 'homepage', 'overview']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')


df = df.dropna()


possible_features = ['budget', 'popularity', 'runtime']
target = 'revenue'


features = [col for col in possible_features if col in df.columns]


if target not in df.columns:
    print("ERROR: Target column 'revenue' not found!")
    exit()


if len(features) == 0:
    print("ERROR: No valid feature columns found!")
    exit()

print("\nUsing features:", features)


X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


print("\nModel Coefficients:")
for i, col in enumerate(features):
    print(f"{col}: {model.coef_[i]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")


y_pred = model.predict(X_test)

print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.show()


print("\nEnter movie details:")

user_input = {}
for col in features:
    value = float(input(f"Enter {col}: "))
    user_input[col] = [value]

user_df = pd.DataFrame(user_input)

predicted_revenue = model.predict(user_df)
print(f"Predicted Revenue: {predicted_revenue[0]:.2f}")