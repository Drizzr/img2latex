#This code tokenizes the csv vormulas and saves them to new csv files
#It also filters out formulas that are too long

import os
import pandas as pd
import matplotlib.pyplot as plt
#Set file as root
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

MAX_LENGTH = 150 #Maximum length of the formulas

dataFilesPath = "../" #Path to the folder where all data files are stored
vocabFile = open(dataFilesPath + "vocab.txt", "r")
datasets = ["im2latex_train", "im2latex_test", "im2latex_validate"]

plotFormularLengths = True #Plot the lengths of the formulas to analyze the dataset

#Statistics
skippedFormulas = 0
longestFormula = 0
totalFormulas = 0
formularTokenLengths = []



#generate a list of all tokens with ids
def create_stoi(list):
    vocab = {}
    for i, token in enumerate(list):
        token = token.replace("\n", "")
        vocab[token] = i
    return vocab

stoi = create_stoi(vocabFile.readlines())

for dataset in datasets:
    df = pd.read_csv(dataFilesPath + dataset + ".csv")

    with open(dataFilesPath + "tokenized_data/" + dataset + "_tokenized.txt", "w") as file:
        for _ , row  in df.iterrows():
                tokenize_formula = []
                formula = str(row["formula"]).strip("\n").split()
                
                tokenize_formula.append(stoi['<sos>'])
                
                for token in formula:
                    #Check if token exists
                    if token not in stoi:
                        if token.startswith("\\"):
                            #When token starts with \ it is a latex command and it gets replaced with <unk>
                            tokenize_formula.append("<unk>")
                        else:
                            #split token into characters
                            for char in token:
                                if char in stoi:
                                    tokenize_formula.append(stoi[char])
                                else:
                                    print("unknown Character: " + char)
                                    tokenize_formula.append("<unk>")                        
                        continue
                    else:
                        tokenize_formula.append(stoi[token])
                    
                tokenize_formula.append(stoi['<eos>'])
                formularTokenLengths.append(len(tokenize_formula))
                
                if(len(tokenize_formula) > MAX_LENGTH):
                    skippedFormulas += 1
                    continue
                
                for _ in range(MAX_LENGTH - len(formula)):
                    tokenize_formula.append(stoi['<pad>'])
                    
                totalFormulas += 1
            
                file.write(str(tokenize_formula) + "," + row["image"] + "\n")

longestFormula = max(formularTokenLengths)

print("Tokenizer Finished:")
print("Skipped formulas: " + str(skippedFormulas))
print("Longest formula: " + str(longestFormula))
print("Total formulas: " + str(totalFormulas))


if plotFormularLengths:
    #plot the lengths of the formulas
    plt.hist(formularTokenLengths, bins=100)
    #Label axis
    plt.xlabel('Length of formulas')
    plt.ylabel('Number of formulas')
    plt.show()
    