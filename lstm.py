from re import T
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics, layers, Sequential,utils
import numpy as np;
import matplotlib.pyplot as plt
from keras import callbacks
import pandas as pd
import os
MODEL_NAME="lstm_fake_news";
  
DEFAULT_ART_LEN=500;
DEFAULT_TRAIN_SIZE=0.5;
DEFAULT_EMB_DIM=64;
DEFAULT_DROP_OUT=0.1;
DEFAULT_LSTM_UNITS=16;
DEFAULT_EPOCHS=5;
DEFAULT_BATCH=512;  
DEFAULT_METRICS=[metrics.BinaryAccuracy(),metrics.Precision(),metrics.Recall()];

class Lstm:
    def __init__(self,train=None,test=None):
        self.voc_dim=None;
        self.art_len=DEFAULT_ART_LEN;
        self.train_size=DEFAULT_TRAIN_SIZE;
        self.emb_dim=DEFAULT_EMB_DIM;
        self.drop_out=DEFAULT_DROP_OUT;
        self.lstm_units=DEFAULT_LSTM_UNITS;
        self.epochs=DEFAULT_EPOCHS;
        self.batch=DEFAULT_BATCH;
        self.dataset=pd.concat([test,train]);

        if (train is not None and test is not None):
            self.train=train["text"].tolist(); 
            self.test=test["text"].tolist(); 
            self.train_labels=train["label"].tolist();
            self.test_labels=test["label"].tolist();

        self.model=None;

    def prepare(self, art_len=None,train_size=None, emb_dim=None, drop_out=None,lstm_units=None,epochs=None,batch=None):
        if art_len is not None:
            self.art_len=art_len;
        if train_size is not None:
            self.train_size=train_size;
        if emb_dim is not None:
            self.emb_dim=emb_dim;
        if drop_out is not None:
            self.drop_out=drop_out;
        if lstm_units is not None:
            self.lstm_units=lstm_units;
        if epochs is not None:
            self.epochs=epochs;
        if batch is not None:
            self.batch=batch;
        self.__prepare();
        return self;

    def build(self):
        self.__build_net();
        return self;

    def __prepare(self):
            if (self.train is None and self.test is None):
                train_s=int(self.train_size*len(self.dataset.index) );
                test_s= int((1-self.train_size)*len(self.dataset.index) );
                sets= self.dataset["text"].tolist();
                labels= self.dataset["label"].tolist();   
                self.train, self.test, self.train_labels, self.test_labels = train_test_split(sets,labels, train_size=train_s, test_size=test_s, random_state=0);
            
            self.tokenizer = Tokenizer(); 
            self.tokenizer.fit_on_texts(self.dataset["text"].tolist());
            self.voc_dim=len(self.tokenizer.word_index);


            self.train=self.__to_padded_seq(self.train);
            self.test=self.__to_padded_seq(self.test);

            self.train = np.array(self.train)
            self.train_labels = np.array(self.train_labels)
            self.test = np.array(self.test)
            self.test_labels = np.array(self.test_labels)

    def __to_padded_seq(self, seq):
        return pad_sequences(self.tokenizer.texts_to_sequences(seq), maxlen=self.art_len, padding="post", truncating="post");

    def __build_net(self):
        self.model = Sequential([
            layers.Embedding(self.voc_dim+1, self.emb_dim, input_length=self.art_len, trainable=True,name="embedding_layer"),
            layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True),name="lstm_1_layer"),
            layers.Bidirectional(layers.LSTM(self.lstm_units),name="lstm_2_layer"),
            layers.Dropout(self.drop_out,name="drop_out_2_layer"),
            layers.Dense(32, activation='relu',name="dense_1_layer"),
            layers.Dense(1, activation='sigmoid',name="dense_2_layer")
        ],name=MODEL_NAME);
        self.model.compile(loss='binary_crossentropy',optimizer='adam',metrics=DEFAULT_METRICS);

    def get_model_summary(self):
        if self.model is None:
            return None;
        return self.model.summary();

    def save_model(self,to_file=None):
        if self.model is None:
            return None;
        if to_file is not None:
            utils.plot_model(self.model, to_file=to_file, show_shapes=True, show_layer_names=True);
        if to_file is None:
            utils.plot_model(self.model, show_shapes=True, show_layer_names=True);
    
    def train_model(self):
        earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 2, 
                                        restore_best_weights = True);
        self.history = self.model.fit(self.train, self.train_labels, validation_split=0.3,epochs=self.epochs,batch_size=self.batch, 
        validation_data=(self.test, self.test_labels),callbacks =[earlystopping],verbose=1);
        return self;
    
    def __plot_metric(self,metric):
        plt.plot(self.history.history[metric]); # Plot metrics from training 
        plt.plot(self.history.history['val_'+metric]); #Plot metrics from test
        plt.xlabel("Epochs");
        plt.ylabel(metric);
        plt.legend(["train_"+metric, "test_"+metric]);

        if not(os.path.exists("train_result")):
            os.mkdir("train_result")
            if not(os.path.exists("train_result/model_1")):
                os.mkdir("train_result/model_1")
        plt.savefig("train_result/model_1/"+metric+".png")

        return plt;

    def plot_metrics(self,metrics=None):
        if metrics is None:
            metrics=DEFAULT_METRICS;
        for metric in metrics:
            self.__plot_metric(metric.name).show();
    def test(self,dataset):
        stat = {};
        t=np.array(self.__to_padded_seq( dataset["text"].tolist()));
        l=np.array( dataset["label"].tolist());
        res=self.model.evaluate(t, l);
        stat["accuracy"]=res[1];
        stat["precision"]=res[2];
        stat["recall"]=res[3];
        stat["f1"]=2*(stat["precision"]*stat["recall"])/(stat["precision"]+stat["recall"]) if (stat["precision"]+stat["recall"])!=0 else None;
        return stat;
    def eval(self,dataset):
        t=np.array(self.__to_padded_seq( dataset["text"].tolist()));        
        res=self.model.predict (t);
        return res;
    def get_vocab_size(self):
        return len(self.tokenizer.word_index);