import pandas as pd
import tensorflow.keras as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_csv('dataset.csv')

data.head()
data.info()
data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
#Divide the data into features and targets
x = data.iloc[:,2:3]
y = data.iloc[:,3]
print(x.head())
print(y.head())
#### split the data into training and testing sample
xtrain,xtest,ytrain,ytest =train_test_split(x,y,train_size=0.80,random_state=1)
print(xtrain)
print(ytrain)
print(xtest)
print(ytest)

#### Create the model
model = tf.models.Sequential()
### Add the layers
model.add(tf.layers.Dense(16,input_dim=1,activation='relu'))     ## input and a hidden layer
model.add(tf.layers.Dense(32,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(64,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(64,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(128,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(128,activation='relu'))   ## hidden layer
model.add(tf.layers.Dense(1,activation='sigmoid')) ## output layer

### compile the model
model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=2000)

ypred = model.predict(xtest)
ypred = ypred.round()
print(ypred)

data1=data.iloc[:51,:]

#graph

plt.figure(figsize=(50,10))
plt.title("Time vs Temperature")
plt.xlabel("Date Time",fontsize=15)
plt.ylabel("Module Temperature",fontsize=15)
plt.xticks(rotation=90)
plt.bar(data1['DATE_TIME'],data1['MODULE_TEMPERATURE'],color="orange",lw=5)
plt.show()

plt.figure(figsize=(10,6))
plt.title("histogram of Temperature")
plt.hist(data1['MODULE_TEMPERATURE'],rwidth=0.9)
plt.show()

plt.figure(figsize=(8,4))
plt.title("Line Plot Graph(Date time vs Temperature)",fontsize=20)
plt.xlabel("Date Time",fontsize=15)
plt.ylabel("Module Temperature",fontsize=15)
plt.plot(data1['DATE_TIME'],data1['MODULE_TEMPERATURE'],label="Line Plot",color="red",lw=5)
plt.xticks(rotation=90)
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(50,10))
plt.title("Temperature vs DC Power")
plt.xlabel("Module Temperature",fontsize=15)
plt.ylabel("DC Power",fontsize=15)
plt.xticks(rotation=90)
plt.bar(data1['MODULE_TEMPERATURE'],data1['DC_POWER'],color="orange",lw=5)
plt.show()

plt.figure(figsize=(8,4))
plt.title("Line Plot Graph(Temperaure vs DC Power)",fontsize=20)
plt.xlabel("Module Temperature",fontsize=15)
plt.ylabel("DC_Power",fontsize=15)
plt.plot(data1['MODULE_TEMPERATURE'],data1['DC_POWER'],label="Line Plot",color="red",lw=2)
plt.xticks(rotation=90)
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(8,4))
plt.title("Scatter Plot Graph",fontsize=20)
plt.xlabel("MODULE_TEMPERATURE",fontsize=15)
plt.ylabel("'DC_POWER",fontsize=15)
plt.scatter(data1['MODULE_TEMPERATURE'],data1['DC_POWER'],label="Scatter Plot",color="red",s=150,marker='*')
plt.legend(loc='best')

plt.figure(figsize=(10,6))
plt.title("histogram of DC_POWER")
plt.hist(data1['DC_POWER'],rwidth=0.9)
plt.show()


xtest

cm = confusion_matrix(ytest,ypred)
print(cm)


print("Please give the Temperature value:")
a=float(input())
b=[]
b.append(a)
#print(b)
ypred = model.predict(b)
ypred = ypred.round()
power=0
if (ypred[0,0]==0):
    if (a<20):
        power=a*0 
    elif ((a>=20)&(a<25)):
        power=a*16.22
    elif ((a>=25)&(a<30)):
        power=a*62.07
    elif ((a>=30)&(a<35)):
        power=a*92.74
   
elif (ypred[0,0]==1):
    if ((a>=30)&(a<35)):
        power=a*92.74
    elif ((a>=35)&(a<40)):
        power=a*115.38
    elif ((a>=40)&(a<45)):
        power=a*121.06
    elif ((a>=45)&(a<50)):
        power=a*144.78
    elif ((a>=50)&(a<55)):
        power=a*176.41
print("The expected Power from the solar panel for the corresponding temperature is : "+str(power))

