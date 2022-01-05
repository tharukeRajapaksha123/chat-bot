import random
import json 
import pickle
from google.protobuf import message
import numpy as np 

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model


lemmerizer = WordNetLemmatizer() 

#load the json file
intents = json.loads(open("intents.json").read())

words =pickle.load(open('words.pkl',"rb"))
classes =pickle.load(open('classes.pkl',"rb"))
model = load_model("chatbotmodel.h5")

#clean sentence 
def cleanup_sentenc(sentance):
    sentance_wors = nltk.word_tokenize(sentance)
    sentance_wors = [lemmerizer.lemmatize(word) for word in sentance_wors]
    return sentance_wors

#create bag of words
def bag_of_words(sentence):
    sentence_words = cleanup_sentenc(sentence)
    bag = [0] * len(words)
    for i in sentence_words:
        for j,word in enumerate(words):
            if word == i: 
                bag[j] = 1
    
    return np.array(bag)

#prefice class based on the senenntce
def prefict_classs(sentence):
    bow =bag_of_words(sentence)
    res= model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25 
    results = [[i,r] for i,r in enumerate(res) if r> ERROR_THRESHOLD]

    #get highest probability first
    results.sort(key = lambda x:x[1],reverse = True) 
    result_list = [] 
    for result in results:
        result_list.append({'intent' : classes[result[0]],"probability" : str(result[1])})
    
    return result_list

#get the response from the bot
def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent["responses"])
            break
    return result


#acitvate the chat bot
while True:
    message = input("You : ")
    inits = prefict_classs(message)
    res =get_response(inits,intents)
    print("bot > ",res)