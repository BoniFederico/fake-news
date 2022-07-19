from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

class RobertaFineModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
        self.model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

    #From HuggingFace doc: https://huggingface.co/hamzab/roberta-fake-news-classification
    def predict_article(self,title,text):
        input_str = "<title>" + title + "<content>" +  text + "<end>"

        input_ids = self.tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
        device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        with torch.no_grad():
            output = self.model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
        res=  [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])] ;
        return 0 if int(res[0])<int(res[1]) else 1;

    #Modified version that allow batch predictions (faster); input as panda dataframes.
    def predict(self,dataset,metrics=False):
        input_str =  "<title>" + dataset['title'] + "<content>" + dataset['text'] + "<end>" ; 
        input_ids = self.tokenizer.batch_encode_plus(input_str.values.tolist(), max_length=512, padding="max_length", truncation=True, return_tensors="pt");
        device =  'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        with torch.no_grad():
            output = self.model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
        #results= pd.DataFrame([[x.item() for x in tensors] for tensors in torch.nn.Softmax()(output.logits)],columns=["fake","real"]);
        results= [(1 if tensors[0]>tensors[1] else 0) for tensors in torch.nn.Softmax()(output.logits)];
        if metrics:
            return self.__compute_metrics(dataset,results);

        return results;

    def __compute_metrics(self,dataset,results):
        if self.results is None:
            return None
        stat = {};
        predictions_res=results;
        predictions=(predictions_res["fake"]>=predictions_res["real"])*1;
        real=dataset["label"].tolist();
        true_fake=sum(real*predictions);
        true_real=sum((real +predictions)==0);
        false_fake=sum(np.array(predictions-real).clip(min=0));
        false_real=sum(np.array(real-predictions).clip(min=0));
        stat["accuracy"]=(true_fake+true_real)/(true_fake+true_real+false_fake+false_real);
        stat["precision"]=(true_fake)/(true_fake+false_real);
        stat["recall"]=(true_fake)/(true_fake+false_fake);
        stat["f1"]=2*(stat["precision"]*stat["recall"])/(stat["precision"]+stat["recall"]);
        return stat;