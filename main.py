import pickle as pkl
import streamlit as st
import numpy as np
import pandas as pd
import librosa.display

scaler = pkl.load(open("dump/scaler.pkl", 'rb'))
kpca = pkl.load(open("dump/kernel_pca.pkl", 'rb'))
knn_model = pkl.load(open("dump/knn_model.sav", 'rb'))
rf_model = pkl.load(open("dump/RF_model.sav", 'rb'))
nn_model = pkl.load(open("dump/NN_model.sav", 'rb'))

st.title("Analyzing Gender Detection in Speech in Adult Filipino Citizens")
uploaded_files = st.file_uploader(label="Upload voice files (.wav)", type="wav", accept_multiple_files=True)
button = st.button("Predict")
if button:
    if uploaded_files is not None:
        name = []
        data = []
        for file in uploaded_files:
            y, sr = librosa.load(file, mono=True, duration=30)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            temp = []
            for e in mfcc:
                temp.append(np.mean(e))
            name.append(file.name)
            data.append(temp)

        X = np.array(data)
        print(X.shape)
        X = scaler.transform(X)
        X = kpca.transform(X)

        knn_pred = knn_model.predict(X)
        rf_pred = rf_model.predict(X)
        nn_pred = nn_model.predict(X)

        results = pd.DataFrame({"File": name,
                                "KNN Prediction:": knn_pred,
                                "Random Forest Prediction:": rf_pred,
                                "ANN Prediction:": nn_pred}).set_index("File")

        mapper = {0.0: "Female", 1.0: "Male"}

        results["KNN Prediction:"] = results["KNN Prediction:"].map(mapper)
        results["Random Forest Prediction:"] = results["Random Forest Prediction:"].map(mapper)
        results["ANN Prediction:"] = results["ANN Prediction:"].map(mapper)

        st.header("Predicted Results")
        st.table(results)
    else:
        st.write("Upload a file!")