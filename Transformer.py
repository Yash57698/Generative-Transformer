import tensorflow as tf
import numpy as np
print("yash")

MAXCONTEXTSIZE = 8
f = open('./input.txt','r')
t = f.readlines()
text = ""
for l in t:
    text += l
itos = {i:chr for i,chr in enumerate(sorted(set(text)))}
stoi = {chr:i for i,chr in enumerate(sorted(set(text)))}

encode = lambda s : np.array([stoi[i] for i in s])
decode = lambda s : ''.join([itos[i] for i in s])


SAMPLES = 2000
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

class AttentionHead(tf.keras.layers.Layer):

    def __init__(self, dim,De):
        super(AttentionHead, self).__init__()
        self.dim = dim
        self.De = De

    def build(self, input_shape):
        # initialize the weights
        Wq_init = tf.random_normal_initializer()
        self.Wq = self.add_weight(name="Query_Weights",shape=(self.dim,self.De),initializer=Wq_init,trainable=True)

        Wk_init = tf.random_normal_initializer()
        self.Wk = self.add_weight(name="Key_Weights",shape=(self.dim,self.De),initializer=Wk_init,trainable=True)

        Wv_init = tf.random_normal_initializer()
        self.Wv = self.add_weight(name="Value_Weights",shape=(self.De, self.De),initializer=Wv_init,trainable=True)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        Q = tf.matmul(self.Wq,inputs,transpose_b=True)
        K = tf.matmul(self.Wk,inputs,transpose_b=True)
        score = tf.matmul(Q,K,transpose_a = True)
        score = tf.scalar_mul(1/tf.sqrt(tf.cast(self.dim,tf.float32)),score)
        score = tf.linalg.band_part(score, -1, 0)
        score = tf.where(score!=0, score, tf.float64.min)
        score = tf.nn.softmax(score,axis = 1)
        V = tf.matmul(self.Wv,inputs,transpose_b=True)
        deltaE = tf.matmul(score,V,transpose_b = True)

        return deltaE

class MultiHead(tf.keras.layers.Layer):
    def __init__(self, dim,De,NoofHeads):
        super(MultiHead, self).__init__()
        self.dim = dim
        self.De = De
        self.NoofHeads = NoofHeads

    def build(self, input_shape):
        # initialize the weights
        self.heads = [AttentionHead(dim = self.dim,De = self.De) for i in range(self.NoofHeads)]

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        outputs = [layer(inputs) for layer in self.heads]
        output = tf.add(inputs,tf.add_n(outputs))
        return output

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, De = 10):
        '''Initializes the instance attributes'''
        super(FeedForward, self).__init__()
        self.De = De

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        W_init = tf.random_normal_initializer()
        self.W = self.add_weight(name="Weights",shape=(self.De,self.De),initializer=W_init,trainable=True)
        self.B = self.add_weight(name="Bias",shape=(1,self.De),initializer=W_init,trainable=True)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        # output = [layer(inputs) for layer in self.heads]
        return tf.add(tf.nn.relu(tf.add(tf.matmul(inputs,self.W),self.B)),inputs)

class Interpret(tf.keras.layers.Layer):
    def __init__(self, De,TokenSize):
        '''Initializes the instance attributes'''
        super(Interpret, self).__init__()
        self.De = De
        self.TokenSize = TokenSize

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        W_init = tf.random_normal_initializer()
        self.W = self.add_weight(name="Weights",shape=(self.De,self.TokenSize),initializer=W_init,trainable=True)


    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        # output = [layer(inputs) for layer in self.heads]
        return (tf.matmul(inputs[:,-1,:],self.W))


TokenSize = len(set(text))
De = 256
d = 64
noofheads = 6
DROPOUT = 0.2

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(None,)))
model.add(tf.keras.layers.Embedding(TokenSize,De))

model.add(MultiHead(d,De,noofheads))
model.add(FeedForward(De))
model.add(tf.keras.layers.Normalization(mean = 0,variance = 1))
model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(MultiHead(d,De,noofheads))
model.add(FeedForward(De))
model.add(tf.keras.layers.Normalization(mean = 0,variance = 1))
model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(MultiHead(d,De,noofheads))
model.add(FeedForward(De))
model.add(tf.keras.layers.Normalization(mean = 0,variance = 1))
model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(MultiHead(d,De,noofheads))
model.add(FeedForward(De))
model.add(tf.keras.layers.Normalization(mean = 0,variance = 1))
model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(MultiHead(d,De,noofheads))
model.add(FeedForward(De))
model.add(tf.keras.layers.Normalization(mean = 0,variance = 1))
model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(MultiHead(d,De,noofheads))
model.add(FeedForward(De))
model.add(tf.keras.layers.Normalization(mean = 0,variance = 1))
model.add(tf.keras.layers.Dropout(DROPOUT))

model.add(Interpret(De,TokenSize))
model.add(tf.keras.layers.Softmax())

model.compile(
    optimizer='adam',  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.build((None,MAXCONTEXTSIZE))
model.summary()
text = "Our very"
print(decode(encode(text)))
print(model(encode(text).reshape(1,-1)))
print(np.sum(model(encode(text).reshape(1,-1))))
print(itos[np.argmax(model(encode(text).reshape(1,-1)))])
model.fit(trainxn,trainyn,batch_size = 256,epochs = 5)
# model.save_weights('./m1')
# model.load_weights('./m1')
# print(model(np.arange(20).reshape(1,-1)))

text = "Our very"
for i in range(30):
    inpu = encode(text).reshape((1,-1))
    out = model(inpu)
    # print(out)
    a = np.argmax(out)
    print(a)
    text += itos[a]
    print(text)