import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks': [35, 40, 45, 50, 55, 60, 70, 80, 85, 90]
}
df = pd.DataFrame(data)

plt.scatter(df['Hours'], df['Marks'])
plt.title("Study Hours vs Marks")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.show()

X = df[['Hours']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)

hours = 9
predicted_marks = model.predict([[hours]])
print(f"If a student studies {hours} hours, they might score {predicted_marks[0]:.2f} marks.")
