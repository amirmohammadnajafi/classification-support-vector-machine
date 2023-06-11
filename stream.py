import streamlit as st
import pandas as pd
from class_bank import bank_loan_long_term
def main():
    st.title("predict_log_tem_client")
    data = st.file_uploader("Upload file", type=["csv", "txt"])
    result = ""
    global model
    if data is not None:
        df = pd.read_csv(data)
        model=bank_loan_long_term("model_svm")
        model.clean_log_toready(df)

    if st.button("Click Here"):
        result = model.predic()
        st.balloons()

    st.success("Prediction Result")
    st.write(result)

if __name__ == "__main__":
    main()