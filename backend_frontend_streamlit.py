import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
df = pd.read_csv("heart.csv")
df=df.replace('?',0) #replace null values with 0
df['trtbps']=df['trtbps'].astype(int) #converting to int from string in dataset all the numbers r stored as strings
df['chol']=df['chol'].astype(int)
df[['trtbps','exng','cp','age']]=df[['trtbps','exng','cp','age']].astype(float)
y=df['output'].astype(float)
df=df.replace('M',0)
df=df.replace('F',1)
x=df[['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']]
sclar=MinMaxScaler() 
x_small=sclar.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x_small,y,test_size=0.3,random_state=1)
X_train=pd.DataFrame(X_train)
model = DecisionTreeClassifier(criterion="entropy") 
grad=GradientBoostingRegressor()
model.fit(X_train,y_train)
st.header("HEART exng prediction")
age=st.number_input("Enter age")
sex = st.number_input("Enter you sex")
cp=st.number_input("Enter cp")
trtbps=st.number_input("Enter trtbps")
chol=st.number_input("Enter chol")
fbs=st.number_input("Enter fbs")
restecg=st.number_input("Enter restecg")
thalachh=st.number_input("Enter thalachh")
exng=st.number_input("Enter exng")
oldpeak=st.number_input("Enter oldpeak")
slp=st.number_input("Enter slp")
caa=st.number_input("Enter caa")
thall=st.number_input("Enter thall")
inputs={
    'age':[age],
    'sex':[sex],
    'cp':[cp],
    'trtbps':[trtbps],
    'chol':[chol],
    'fbs':[fbs],
    'restecg':[restecg],
    'thalachh':[thalachh],
    'exng':[exng],
    'oldpeak':[oldpeak],
    'slp':[slp],
    'caa':[caa],
    'thall':[thall]
}
df_in = pd.DataFrame.from_dict(inputs)
val = model.predict(df_in)

if st.button("Predict"):
    st.success(f"Heart stroke predicted is : {val[0]}")