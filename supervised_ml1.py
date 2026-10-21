from sklearn.linear_model import LinearRegression

# features (exam done)
x=[[1],[2],[3],[4],[5],[6]]

#  label (scores)
y=[40,55,60,70,80,85]

# model training
model=LinearRegression()
model.fit(x,y)

# prediction exam 7
prediction=model.predict([[7]])
print(prediction[0])

import matplotlib.pyplot as plt
import numpy as np 

x=np.array([[1],[2],[3],[4],[5],[6]])
y=np.array([40,55,60,70,80,85])

# calculate line y=mx+b m=slope ,b=where the line start
m=model.coef_[0]
b=model.intercept_
y_pred=m*x+b

plt.scatter(x,y,color="red",label="actual data")
plt.plot(x,y_pred,color="green",label="regression line")

plt.xlabel('exams  done')
plt.ylabel('exams score')
plt.title('linear regression')
plt.legend()
plt.show()

score=model.predict([[7]])
print("predicted score",score[0])