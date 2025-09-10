from sklearn.linear_model import  LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# loading data
data = pd.read_csv("college_student_placement_dataset.csv")

# dependent and independent variable
x = data[['IQ']]
y = data['CGPA']

# initialize the linear regression object
model = LinearRegression()

# training the model on the data set
model.fit(x,y)

# user IQ level
IQ = int(input("enter you IQ = "))
# /making prediction
predicted_marks = model.predict(pd.DataFrame([[IQ]], columns=["IQ"]))

# output
print(f"coefficient: {model.coef_}\n intercept: {model.intercept_}")
print(f"based on your IQ , you may get {predicted_marks} CGPA")

# visualization
fig , ax = plt.subplots(1,2,figsize = (5,5))
ax[0].scatter(x, y, color=['lightblue'])
ax[0].set_title("scatter plot of IQ and CGPA")
ax[0].set_xlabel('IQ')
ax[0].set_ylabel("CGPA")

ax[1].scatter(x=IQ, y= predicted_marks, color=['lightblue'])
ax[1].set_title("scatter plot of expected_IQ and predicted CGPA")
ax[1].set_xlabel('IQ')
ax[1].set_ylabel("CGPA")

fig.suptitle("Comparison of actual data and predicted data")
plt.show()