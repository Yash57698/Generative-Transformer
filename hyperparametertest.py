import tensorflow as tf
import keras_tuner as kt
from Transformer2 import *

#Constants--------------------
MAXCONTEXTSIZE = 256
Tokensize = 65
BatchSize = 16
Epochs = 5
#-----------------------------

def model_builder(hp):
    model = tf.keras.Sequential()
    #Hyperparameters:
    De = hp.Int('Embedding_dimension', min_value=256, max_value=512, step=16)
    d = hp.Choice('model_dimension', values=[8,16,32,40,48])
    noofheads = hp.Choice('noofheads', values=[4,5,6])
    nooflayers = hp.Choice('nooflayers', values=[0,1,2,3,4,5])
    LearningRate = hp.Choice('learning_rate', values=[1e-3, 1e-4,1e-5])
    usesine = hp.Choice('usesine', values=[True,False])
    #----------------------------
    
    model.add(Embed(De,Tokensize,MAXCONTEXTSIZE,usesine))       
    for i in range(nooflayers):
        model.add(AttentionBlock(De,d,noofheads))
    if nooflayers == 0:
        model.add(DotProductAttention(De,d))
    model.add(Interpret(De,Tokensize))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = LearningRate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy','loss'])
    
    return model


tuner = kt.Hyperband(model_builder,
                    objective='val_accuracy',
                    max_epochs=15,
                    factor=3,
                    directory='Dir',
                    project_name='intro_Transformer')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#Loading Data-----------------------------------------------------------------------------------------
f = open('./input.txt','r')
t = f.readlines()
text = ""
for l in t:
    text += l
itos = {i:chr for i,chr in enumerate(sorted(set(text)))}
stoi = {chr:i for i,chr in enumerate(sorted(set(text)))}

encode = lambda s : np.array([stoi[i] for i in s])
decode = lambda s : ''.join([itos[i] for i in s])

SAMPLES = 10000
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
#-----------------------------------------------------------------------------------------------------


tuner.search(trainxn, trainyn, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
params = tuner.get_best_hyperparameters(num_trials=1)
for i in range(5):
    print(f"Parameters of {i}th hps are")
    best_hps=params[i]
    print(f"Embedding dim : {best_hps.get('Embedding_dimension')}")
    print(f"Model dim : {best_hps.get('model_dimension')}")
    print(f"No of Heads : {best_hps.get('noofheads')}")
    print(f"No of Layers : {best_hps.get('nooflayers')}")
    print(f"Learning Rate : {best_hps.get('learning_rate')}")
    print(f"Use Sine : {best_hps.get('usesine')}")

