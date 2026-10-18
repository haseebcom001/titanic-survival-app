import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

categories = {
    "PClass": ["First", "Second", "Third"],
    "Gender": ["Male", "Female"],
    "Sibling": ["Zero", "One", "Two", "Three"],
    "Embarked": ["Southampton", "Cherbourg", "Queenstown"],
}
encoders = {}
for column, values in categories.items():
    enc = LabelEncoder()
    enc.fit(values)
    encoders[column] = enc

@st.cache_resource
def load_model():
    with open("svc_trained_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🚢 Titanic Passenger Survival Prediction")
pclass = st.selectbox("Passenger class (PClass)", categories["PClass"])
gender = st.selectbox("Gender", categories["Gender"])
sibling = st.selectbox("Siblings/spouses (Sibling)", categories["Sibling"])
embarked = st.selectbox("Port of embarkation (Embarked)", categories["Embarked"])

if st.button("Predict"):
    user_input = pd.DataFrame({"PClass": [pclass], "Gender": [gender],
                               "Sibling": [sibling], "Embarked": [embarked]})
    for column in categories:
        user_input[column] = encoders[column].transform(user_input[column])
    prediction = model.predict(user_input)[0]
    if prediction == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")
