import pandas as pd;
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import numpy as np
import spacy

nlp = spacy.load(  "en_core_web_sm")

#Regex: 
URL_REGEX="http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+" 
NUMBER_REGEX="[0-9]"
ALPHA_NUM_REGEX="[^A-Za-z0-9 ]+"
WORDS_REGEX=r"\b\w\b";
#Paths:
PATH_DATASET_0="datasets/dataset_0/"
PATH_DATASET_1="datasets/dataset_1/"
PATH_DATASET_2="datasets/dataset_2/";


class Dataset:
    
    def __init__(self,n,drop_dup=True,concat_title=True):
        match n:
            case 0:
                self.dataset= self.__load_dataset_0();
            case 1:
                self.dataset= self.__load_dataset_1();
            case 2:
                self.dataset= self.__load_dataset_2();
            case _:
                self.dataset= self.__load_dataset_0();
        if drop_dup is True:
            self.dataset.drop_duplicates(subset=["text"], keep='first', inplace=True);
        if concat_title is True:
            self.dataset["text"]=self.dataset["title"]+" "+self.dataset["text"];

    def append(self,n=None,ds=None):
        if n is not None:
            match n:
                case 0:
                    self.dataset=pd.concat([self.dataset, self.__load_dataset_0()]);
                case 1:
                    self.dataset=pd.concat([self.dataset, self.__load_dataset_1()]);
                case 2:
                    self.dataset=pd.concat([self.dataset, self.__load_dataset_2()]);
                case _:
                    self.dataset=pd.concat([self.dataset,self.__load_dataset_0()]);
        if ds is not None:
            self.dataset=pd.concat([self.dataset,ds]);

        self.dataset.drop_duplicates(subset=["text"], keep='first', inplace=True);
        return self;

    def sort(self):
        self.dataset=self.dataset.sample(frac=1);
        return self;

    def plot_lengths(self, fig_name):
        plt.hist([len(x) for x in self.dataset["text"]], bins=300);
        plt.savefig(fig_name);
        plt.show();

    def filter(self,only_fake=False, only_real=False):
        if only_fake is True:
            self.dataset= self.dataset.query("label == 1");
        if only_real is True:
            self.dataset= self.dataset.query("label == 0");
        return self;

    def get(self,first=None):
        if first is None:
            return self.dataset;
        if first is not None:
            return self.dataset.head(first);

    def remove_stop_words(self):
        self.dataset['text']=[remove_stopwords(text) for text in self.dataset['text'] ];
        self.dataset['title']=[remove_stopwords(text) for text in self.dataset['title'] ];
        self.dataset['text'] = self.dataset['text'].str.replace(WORDS_REGEX, '',regex=True);
        self.dataset['title'] = self.dataset['title'].str.replace(WORDS_REGEX, '',regex=True);
        return self;

    def limit(self,n):
        self.dataset=self.dataset.head(n);
        return self;

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer();
        self.dataset['text']=[lemmatizer.lemmatize(text) for text in self.dataset['text'] ];
        self.dataset['title']=[lemmatizer.lemmatize(text) for text in self.dataset['title'] ];
        return self;

    def remove_numbers(self):
        self.dataset['text'] = self.dataset['text'].str.replace(NUMBER_REGEX, '',regex=True);
        return self;

    def drop_empty_rows(self):
        self.dataset["text"].replace("",float("NaN"),inplace=True);
        self.dataset["title"].replace("",float("NaN"),inplace=True);
        self.dataset.dropna(subset=['text','title'], inplace=True);
        return self;

    def clean(self):
        self.drop_empty_rows();
        self.dataset['text'] = self.dataset['text'].str.replace(URL_REGEX, ' ', regex=True);
        self.dataset['text'] = self.dataset['text'].str.replace(NUMBER_REGEX, '0',regex=True);
        self.dataset['text'] = self.dataset['text'].str.replace(ALPHA_NUM_REGEX, '',regex=True);
        self.dataset['text']= [text.lower() for text in self.dataset["text"]];


        self.dataset['title'] = self.dataset['title'].str.replace(URL_REGEX, ' ',regex=True);
        self.dataset['title'] = self.dataset['title'].str.replace(NUMBER_REGEX, '0',regex=True);
        self.dataset['title'] = self.dataset['title'].str.replace(ALPHA_NUM_REGEX, '',regex=True);

        return self;

    def __load_dataset_0(self): #DATASET 0 (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset):
        dataset_fake= pd.read_csv(PATH_DATASET_0+"Fake.csv", usecols=["title","text"]);
        dataset_true =pd.read_csv(PATH_DATASET_0+"True.csv",usecols=["title","text"]);

        dataset_fake['label']=1;
        dataset_true['label']=0;

        dataset=pd.concat([dataset_fake,dataset_true]);
        dataset.reset_index(inplace=True);
        return dataset;

    def __load_dataset_1(self): #DATASET 1 (https://www.kaggle.com/c/fake-news/overview):
    
        dataset_train= pd.read_csv(PATH_DATASET_1+"train.csv",usecols=["title","text","label"]);
        dataset_test =pd.read_csv(PATH_DATASET_1+"test.csv",usecols=["title","text"]);
        dataset_test["label"]=pd.read_csv(PATH_DATASET_1+"submit.csv",usecols=["label"]);

        dataset=pd.concat([dataset_test,dataset_train]);
        return dataset;

    def __load_dataset_2(self): #DATASET 2 (https://www.kaggle.com/datasets/jruvika/fake-news-detection):
    
        dataset= pd.read_csv(PATH_DATASET_2+"data.csv",usecols=["Headline","Body","Label"]);
        dataset.rename(columns={"Headline": "title","Body":"text","Label":"label"}, inplace=True);

        dataset["label"]=1-dataset["label"]
        return dataset;

    def word_cloud(self,fig_name):
        #Word Cloud
        text = ''
        for art in self.dataset["text"]:
            text += art
        wordcloud = WordCloud(width = 3000,height = 2000,background_color = 'white',stopwords=set(STOPWORDS)).generate(text)
        fig = plt.figure(figsize = (40, 30),facecolor = 'k',edgecolor = 'white')
        plt.imshow(wordcloud, interpolation = 'bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(fig_name)
        plt.show()

    def num_of_unique_tokens(self):
        tkz= Tokenizer();
        tkz.fit_on_texts([document.split() for document in self.dataset["text"]])  
        return len(tkz.word_index);
    

    def count_occurrences(self,regex):
        return np.sum([len(re.findall(regex, tx)) for tx in self.dataset["text"]])

    def avg_length_of_text(self):
        return np.average([len(document.split())  for document in self.dataset["text"]])

    def dic_of_named_entities(self):
        entities= list();
        for sentence in self.dataset["text"]:
            entities.extend([(entity.label_, entity.text) for entity in nlp(sentence).ents]);
        return self.__build_dict_from_named_entities(entities);

    def __build_dict_from_named_entities(self,ners):
        outer_dict = {}
        for ner in ners:
            entity_type = ner[0]
            entity_name = ner[1]
            if entity_type in outer_dict:
                if entity_name in outer_dict[entity_type]:
                    outer_dict[entity_type][entity_name] += 1
                else:
                    outer_dict[entity_type][entity_name] = 1
            else:
                outer_dict[entity_type] = {entity_name: 1}
        return outer_dict
 