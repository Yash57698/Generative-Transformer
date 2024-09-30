import tensorflow as tf
import numpy as np
from Transformer2 import *
MAXCONTEXTSIZE = 256
#------------------------------------------
f = open('./input.txt','r')
t = f.readlines()
text = ""
for l in t:
    text += l
itos = {i:chr for i,chr in enumerate(sorted(set(text)))}
stoi = {chr:i for i,chr in enumerate(sorted(set(text)))}

encode = lambda s : np.array([stoi[i] for i in s])
decode = lambda s : ''.join([itos[i] for i in s])

SAMPLES = 200000
trainx = []
trainy = np.zeros(shape=SAMPLES-1)
print(len(text))

for i in range(1,SAMPLES):
    if i%100000 == 0:
        print(i)
    if i <= MAXCONTEXTSIZE:
        x = text[:i]
        y = text[i]
    else:
        x = text[i-MAXCONTEXTSIZE:i]
        y = text[i]
    trainx.append(encode(x))
    trainy[i-1] = (encode(y)[0])

trainxn = np.array(trainx[MAXCONTEXTSIZE+1:])
trainyn = trainy[MAXCONTEXTSIZE+1:]
#-----------------------------------------

trainxn = trainxn[-100000:]
trainyn = trainyn[-100000:]

Tokensize = 65
De = 336
d = 8
noofheads = 5
nooflayers = 5
BatchSize = 16
LearningRate = 1e-4
Epochs = 20

weightpath = "./kaggleResults/results/oioio/intro_Transformer/trial_0024/checkpoint.weights.h5"

model = tf.keras.Sequential()
model.add(Embed(De,Tokensize,MAXCONTEXTSIZE,usesine=False))
for i in range(nooflayers):
    model.add(AttentionBlock(De,d,noofheads))
model.add(Interpret(De,Tokensize))  
# model.load_weights(weightpath)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = LearningRate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
model.build((None,256))
model.load_weights("test.weights.h5")
# model.fit(
#     trainxn,
#     trainyn, 
#     batch_size=BatchSize,
#     epochs=Epochs, 
#     validation_split=0.2
#     )
# model.save_weights("test2.weights.h5")
model.evaluate(trainxn,trainyn)


text = "FIRST"
i = 0
while i <= 15000:
    i+=1
    print(i,end='\r')
    inpu = encode(text[-(256 if len(text) > 256 else 0):]).reshape((1,-1))
    out = model(inpu)
    a = np.argmax(out)
    text += itos[a]
    if(i%1000 == 0):
        print(text)
print(text)

