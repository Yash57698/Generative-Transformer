import json

dat = []
for i in range(30):
    id = str(i).zfill(4)
    f = open("./kaggleResults/results/oioio/intro_Transformer/trial_"+id+"/trial.json")
    data = json.load(f)
    dat.append([data["metrics"]["metrics"]["val_loss"]["observations"][0]["value"][0] , data["trial_id"] , data["hyperparameters"]["values"],data["metrics"]["metrics"]["val_accuracy"]["observations"][0]["value"][0],data["metrics"]["metrics"]["val_accuracy"]["observations"][0]["step"]])
    # print("Trial ID: ",data["trial_id"] ,"Score: ",data["score"],"val_loss: ",data["metrics"]["metrics"]["val_loss"]["observations"][0]["value"][0])

for i in sorted(dat):
    print(f"Trial ID:",i[1],", Val_Accuracy:",i[3],", De:",i[2]["Embedding_dimension"],", d:",i[2]["model_dimension"],", heads:",i[2]["noofheads"],", layers:",i[2]["nooflayers"],", sine:",i[2]["usesine"],"Learning Rate: ",i[2]["learning_rate"]," Steps:",i[4],"/",i[2]["tuner/epochs"])

'''
    "Embedding_dimension": 416,
    "model_dimension": 40,
    "noofheads": 5,
    "nooflayers": 1,
    "learning_rate": 0.001,
    "usesine": 0,
    "tuner/epochs": 15,
    "tuner/initial_epoch": 0,
    "tuner/bracket": 0,
    "tuner/round": 0
'''