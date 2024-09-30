import tensorflow as tf
import numpy as np
import math

#Works well
class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, EmbedD , modelD):
        super(DotProductAttention,self).__init__()
        self.EmbedD = EmbedD
        self.modelD = modelD

    def build(self,input_shape):
        Wq_init = tf.random_normal_initializer()
        Wk_init = tf.random_normal_initializer()
        Wvd_init = tf.random_normal_initializer()
        Wvu_init = tf.random_normal_initializer()
        self.Wq = self.add_weight(name="Query_Weights",shape=(self.EmbedD,self.modelD),initializer = Wq_init,trainable = True)
        self.Wk = self.add_weight(name="Key_Weights",shape=(self.EmbedD,self.modelD),initializer = Wk_init,trainable = True)
        self.Wvdown = self.add_weight(name="Valuedown_Weights",shape=(self.EmbedD,self.modelD),initializer = Wvd_init,trainable = True)
        self.Wvup = self.add_weight(name="Valueup_Weights",shape=(self.modelD,self.EmbedD),initializer = Wvu_init,trainable = True)
    
    def call(self,inputs):
        Q = tf.matmul(inputs,self.Wq)
        K = tf.matmul(inputs,self.Wk)
        score = tf.matmul(Q,K,transpose_b=True)
        score = tf.scalar_mul(1/tf.sqrt(tf.cast(self.modelD,tf.float32)),score)        
        score = tf.linalg.band_part(score, -1, 0)
        score = tf.where(score!=0, score, tf.float64.min)
        score = tf.nn.softmax(score)
        V = tf.matmul(inputs,tf.matmul(self.Wvdown,self.Wvup))

        return tf.matmul(score,V)
    
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,EmbedD,modelD,noofheads,useweight = True):
        super(MultiHeadAttention,self).__init__()
        self.EmbedD = EmbedD
        self.modelD = modelD
        self.noofheads = noofheads
        self.useweight = useweight

    def build(self,input_shape):
        self.heads = [DotProductAttention(self.EmbedD,self.modelD) for i in range(self.noofheads)]
        W_init = tf.random_normal_initializer()
        if self.useweight:
            self.Wo = self.add_weight(name="ConcatWeight",shape=(self.noofheads,1),initializer = W_init,trainable = True)

    def call(self,inputs):
        if self.useweight:
            outputs = tf.stack([DPA(inputs) for DPA in self.heads],axis=3)
            outputs = tf.matmul(outputs,self.Wo)
            return tf.reshape(outputs,tf.shape(inputs))
        else:
            outputs = tf.add_n([DPA(inputs) for DPA in self.heads])
            return outputs

class FeedForward(tf.keras.layers.Layer):
    def __init__(self,EmbedD):
        super(FeedForward,self).__init__()
        self.EmbedD = EmbedD

    def build(self,input_shape):
        W1_init = tf.random_normal_initializer()
        W2_init = tf.random_normal_initializer()
        B1_init = tf.random_normal_initializer()
        B2_init = tf.random_normal_initializer()
        self.W1 = self.add_weight(name="Weight1",shape=(self.EmbedD,self.EmbedD),initializer = W1_init,trainable = True)
        self.W2 = self.add_weight(name="Weight2",shape=(self.EmbedD,self.EmbedD),initializer = W2_init,trainable = True)
        self.B1 = self.add_weight(name="Bias1",shape=(1,self.EmbedD),initializer = B1_init,trainable = True)
        self.B2 = self.add_weight(name="Bias2",shape=(1,self.EmbedD),initializer = B2_init,trainable = True)
    
    def call(self,inputs):
        output = tf.add(tf.matmul(inputs,self.W1),self.B1)
        output = tf.nn.relu(output)
        output = tf.add(tf.matmul(output,self.W2),self.B2)
        return output
        
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self,EmbedD,modelD,noofheads):
        super(AttentionBlock,self).__init__()
        self.EmbedD = EmbedD
        self.modelD = modelD
        self.noofheads = noofheads

    def build(self,input_shape):
        self.multihead = MultiHeadAttention(self.EmbedD,self.modelD,self.noofheads)
        self.feedforward = FeedForward(self.EmbedD)
        self.normlayer = tf.keras.layers.LayerNormalization(axis = 2)

    def call(self,inputs):
        output = tf.add(inputs,self.multihead(inputs))
        output = self.normlayer(output)
        
        output = tf.add(output,self.feedforward(output))
        output = self.normlayer(output)
        return output
    
class Interpret(tf.keras.layers.Layer):
    def __init__(self, De,TokenSize):
        super(Interpret, self).__init__()
        self.De = De
        self.TokenSize = TokenSize

    def build(self, input_shape):
        W_init = tf.random_normal_initializer()
        self.W = self.add_weight(name="Iterpret_Weight",shape=(self.De,self.TokenSize),initializer=W_init,trainable=True)


    def call(self, inputs):
        return (tf.matmul(inputs[:,-1,:],self.W))
    
class Embed(tf.keras.layers.Layer):
    def __init__(self, De,TokenSize,Maxcontext,usesine = False):
        super(Embed, self).__init__()
        self.De = De
        self.TokenSize = TokenSize
        self.maxcontext = Maxcontext
        self.usesine = usesine

    def build(self, input_shape):
        self.embedlayer = tf.keras.layers.Embedding(self.TokenSize,self.De)
        if self.usesine:
            sineposition = [math.sin(i/(math.pow(10000,2*i/self.maxcontext))) for i in range(self.maxcontext//2)]
            cosineposition = [math.cos(i/(math.pow(10000,2*i/self.maxcontext))) for i in range(self.maxcontext//2)]
            a = np.zeros((1,self.maxcontext))
            a[:,0::2] = sineposition
            a[:,1::2] = cosineposition
            a = np.array(a)
            a = a[...,np.newaxis]
            self.posembed = tf.Variable(initial_value=a,trainable=False,dtype=tf.float32)
        else:
            self.posembedlayer = tf.keras.layers.Embedding(self.maxcontext,self.De)
        
    def call(self, inputs):
        inputshape = tf.shape(inputs)
        if not self.usesine:
            self.posembed = self.posembedlayer(tf.range(self.maxcontext))[None,...]
            output = self.embedlayer(inputs)
            output = tf.add(output,self.posembed[:,:inputshape[-1],:])
        else:
            output = self.embedlayer(inputs)
            output = tf.add(output,self.posembed[:,:inputshape[-1],:])
        return output