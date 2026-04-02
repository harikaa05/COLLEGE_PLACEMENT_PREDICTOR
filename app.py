# ===============================
# Student Placement Prediction App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score


# ===============================
# Page Configuration
# ===============================

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Placement Prediction System")
st.markdown("### AI Based Placement & Salary Predictor")


# ===============================
# Load Dataset
# ===============================

df = pd.read_csv("student_placement_synthetic.csv")

st.sidebar.header("Dataset Info")

st.sidebar.write("Rows :", df.shape[0])
st.sidebar.write("Columns :", df.shape[1])


# ===============================
# Encode Categorical Data
# ===============================

le_branch = LabelEncoder()
le_tier = LabelEncoder()

df['branch'] = le_branch.fit_transform(df['branch'])
df['college_tier'] = le_tier.fit_transform(df['college_tier'])


# ===============================
# Feature Selection
# ===============================

X = df.drop(['placement_status','salary_package_lpa'],axis=1)

y = df['placement_status']

salary_data = df.dropna()

Xs = salary_data.drop(['placement_status','salary_package_lpa'],axis=1)
ys = salary_data['salary_package_lpa']


# ===============================
# Train Test Split
# ===============================

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

Xs_train,Xs_test,ys_train,ys_test = train_test_split(Xs,ys,test_size=0.2)


# ===============================
# Model Training
# ===============================

model = RandomForestClassifier()
model.fit(X_train,y_train)

salary_model = RandomForestRegressor()
salary_model.fit(Xs_train,ys_train)


# ===============================
# Accuracy
# ===============================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

st.sidebar.write("Model Accuracy :",round(accuracy*100,2),"%")



# ===============================
# User Input Section
# ===============================

st.sidebar.header("Enter Student Details")

branch = st.sidebar.selectbox("Branch",le_branch.classes_)

tier = st.sidebar.selectbox("College Tier",le_tier.classes_)

cgpa = st.sidebar.slider("CGPA",0.0,10.0,7.0)

backlogs = st.sidebar.slider("Backlogs",0,10,0)

coding = st.sidebar.slider("Coding Skills",0,10,5)

dsa = st.sidebar.slider("DSA Score",0,100,50)

aptitude = st.sidebar.slider("Aptitude Score",0,100,50)

communication = st.sidebar.slider("Communication Skills",0,10,5)

ml = st.sidebar.slider("ML Knowledge",0,10,3)

system = st.sidebar.slider("System Design",0,10,3)

internships = st.sidebar.slider("Internships",0,5,1)

projects = st.sidebar.slider("Projects",0,10,2)

certifications = st.sidebar.slider("Certifications",0,10,2)

hackathons = st.sidebar.slider("Hackathons",0,10,1)

opensource = st.sidebar.slider("Open Source Contributions",0,20,0)

extra = st.sidebar.slider("Extracurriculars",0,10,2)



# ===============================
# Prediction
# ===============================

input_data = np.array([[
le_branch.transform([branch])[0],
le_tier.transform([tier])[0],
cgpa,
backlogs,
coding,
dsa,
aptitude,
communication,
ml,
system,
internships,
projects,
certifications,
hackathons,
opensource,
extra
]])


if st.sidebar.button("Predict Placement"):

    result = model.predict(input_data)

    if result[0] == 1:

        st.success("🎉 Student Will Get Placed")

        salary = salary_model.predict(input_data)

        st.info(f"💰 Expected Salary : {round(salary[0],2)} LPA")

    else:

        st.error("❌ Student May Not Get Placement")



# ===============================
# Data Visualization
# ===============================

st.header("📊 Data Visualization")

col1,col2 = st.columns(2)

with col1:

    st.subheader("CGPA vs Placement")

    fig = plt.figure()

    sns.boxplot(x=df['placement_status'],y=df['cgpa'])

    st.pyplot(fig)



with col2:

    st.subheader("Internships vs Placement")

    fig = plt.figure()

    sns.countplot(x=df['internships'],hue=df['placement_status'])

    st.pyplot(fig)



col3,col4 = st.columns(2)

with col3:

    st.subheader("Branch Distribution")

    fig = plt.figure()

    sns.countplot(x=df['branch'])

    st.pyplot(fig)



with col4:

    st.subheader("Salary Distribution")

    fig = plt.figure()

    sns.histplot(df['salary_package_lpa'],bins=30)

    st.pyplot(fig)



# ===============================
# Feature Importance
# ===============================

st.header("📈 Feature Importance")

importance = model.feature_importances_

features = X.columns

imp_df = pd.DataFrame({

'Feature':features,

'Importance':importance

})

imp_df = imp_df.sort_values(by='Importance',ascending=False)

fig = plt.figure(figsize=(10,6))

sns.barplot(x='Importance',y='Feature',data=imp_df)

st.pyplot(fig)



# ===============================
# Dataset Preview
# ===============================

st.header("Dataset Preview")

st.dataframe(df.head(50))