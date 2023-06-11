class bank_loan_long_term:
    def __init__(self,model_file):
        import pickle
        with open('model_svm','rb') as model_file:
            self.svm=pickle.load(model_file)
        self.data=None
        self.x = None
    def clean_log_toready (self,data_file):
        import pandas as pd
        import numpy as np
        data=pd.read_csv(data_file)
        data.drop(data[data["job"]=="unknown"].index,inplace=True)
        data.drop(data[data["marital"]=="unknown"].index,inplace=True)
        data.drop(data[data["education"]=="unknown"].index,inplace=True)
        data.drop(data[data["default"]=="unknown"].index,inplace=True)
        data.drop(data[data["housing"]=="unknown"].index,inplace=True)
        data.drop("poutcome",axis=1,inplace=True)
        data["log_dur"]=np.log10(data.duration.replace(0, np.nan))
        data["y"].replace("yes",1,inplace=True)
        data["y"].replace("no",0,inplace=True)
        data=data[data["log_dur"]!=0]
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(data.mean(numeric_only=True), inplace=True)
        data.dropna(inplace=True)
        self.data=data
        self.x=self.data[["log_dur","pdays","age"]]
        return self.data
    def predic(self):
        self.data["predicted"]=self.svm.predict(self.x)
        return self.data


n=bank_loan_long_term("model_svm")
n.clean_log_toready(r"E:\proge bank gharz\new_train.csv")
n=n.predic()
print(n)

