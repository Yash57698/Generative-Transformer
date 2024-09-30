import tensorflow as tf
import numpy as np
import keras
import time
from Transformer2 import *

#hyperparameters
MAXCONTEXTSIZE = 256
Tokensize = 65
De = 384
d = 8
noofheads = 6
nooflayers = 6
BatchSize = 16
LearningRate = 1e-3
Epochs = 5
#-----------------------

f = open('./input.txt','r')
t = f.readlines()
text = ""
for l in t:
    text += l
itos = {i:chr for i,chr in enumerate(sorted(set(text)))}
stoi = {chr:i for i,chr in enumerate(sorted(set(text)))}

encode = lambda s : np.array([stoi[i] for i in s])
decode = lambda s : ''.join([itos[i] for i in s])

SAMPLES = len(text)
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


model = tf.keras.Sequential()
model.add(Embed(De,Tokensize,MAXCONTEXTSIZE))
model.add(DotProductAttention(De,d))
# model.add(AttentionBlock(De,d,noofheads))
# model.add(AttentionBlock(De,d,noofheads))
# model.add(AttentionBlock(De,d,noofheads))
# model.add(MultiHeadAttention(De,d,noofheads))
model.add(Interpret(De,Tokensize))


model.build((None,MAXCONTEXTSIZE))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = LearningRate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

model.summary()

# logdir="./logs/fit/heyo"
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# for j in range(Epochs):
#     print(f"Sample : {j}/{Epochs}")

model.fit(
    trainxn,
    trainyn, 
    batch_size=BatchSize,
    epochs=Epochs, 
    validation_split=0.15
    )
        
model.evaluate(trainxn[:10000],trainyn[:10000],batch_size = BatchSize)

model.save_weights('./oioi.weights.h5')

text = "First Citizen:"
for i in range(10000):
    inpu = encode(text[-(256 if len(text) > 256 else 0):]).reshape((1,-1))
    out = model(inpu)
    a = np.argmax(out)
    text += itos[a]
print(text)


