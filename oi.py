#hyperparameters
MAXCONTEXTSIZE = 256
Tokensize = 65
De = 384
d = 16
noofheads = 6
nooflayers = 6
BatchSize = 16
LearningRate = 1e-1
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

SAMPLES = 50000
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