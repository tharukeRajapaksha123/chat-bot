#this file is for training the bot
import random 
import json
import pickle 
import numpy as np 

import nltk
#nltk.download('omw-1.4')
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lammatizer = WordNetLemmatizer()

#load the json file
intents = json.loads(open("intents.json").read())

#declarring global variabls
words = []
classes = []
documents = []
ignore_letters = ['?',',','.',' ']

#reading intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent["tag"]))
        if intent['tag'] not in classes:
            classes.append(intent["tag"])

#lamerticing the indivual words
words = [lammatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

#sorting classes
classes = sorted(set(classes))

#save words and classes as bytes files
pickle.dump(words,open("words.pkl","wb"))
pickle.dump(classes,open("classes.pkl","wb"))

#training list
training = [] 
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lammatizer.lemmatize(word.lower()) for word in word_patterns]
    #chck the word in the pattern
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag,output_row])

random.shuffle(training)
#convert to num py array
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

#starting builidn the netural network
model =Sequential()
model.add(Dense(128,input_shape =(len(train_x[0]),),activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(64,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = "softmax"))

sgd = SGD(lr=0.01, decay = 1e-6,momentum = 0.9,nesterov = True)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"])

hist = model.fit(np.array(train_x),np.array(train_y),epochs = 200,batch_size=5,verbose = 1)

model.save("chatbotmodel.h5",hist)
print("----- bot trained ------")