import numpy as np
from sklearn.linear_model import LogisticRegression

# create input dataset (x1 = Active Minutes, x2 = Calories Consumed in thousands)
X = np.array([
    [60, 1.5],
    [15, 2.8],
    [55, 1.6],
    [10, 3.0],
    [70, 1.4],
    [20, 2.6],
    [65, 1.7],
    [12, 3.2],
    [50, 1.8],
    [25, 2.5],
])

# create output dataset (0 = Missed Goal, 1 = Achieved Goal)
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# create model
model = LogisticRegression()

# train model
model.fit(X, y)

# create variable that has data to predict
new_user = np.array([[40, 2.1]])

# prediction
prediction = model.predict(new_user)
print("prediction about calorie goal", prediction)

probability = model.predict_proba(new_user)
print("probability of achieving goal = ", probability)

# probability in percentage
prob_miss_pct = probability[0][0] * 100
prob_achieve_pct = probability[0][1] * 100

print(f"probability of missing goal = {prob_miss_pct:.2f}%")
print(f"probability of achieving goal = {prob_achieve_pct:.2f}%")