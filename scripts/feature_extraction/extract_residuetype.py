'''
Author: Xianni Zhong, email: zhongxn@shanghaitech.edu.cn
Created on 2021.12.12

'''
import os
import numpy as np
from numpy import argmax

# define input string

# Filepath="./new-fasta"
Filepath="./fasta"

SequenceDict={}

for file in os.listdir(Filepath):
    try:
        if os.path.splitext(file)[1] == ".fasta":
            SequenceDict[os.path.splitext(file)[0]]=open(Filepath+'/'+file, 'r') .read()
    except UnicodeDecodeError as e:
        print(e, file)
# print([i for i in SequenceDict.keys()][:5])
# print([i for i in SequenceDict.values()][:5])

# define universe of possible input values
    ResidueType = 'ARNDCQEGHILKMFPSTWYVX'
    # define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(ResidueType))
    int_to_char = dict((i, c) for i, c in enumerate(ResidueType))

m=0
for SequenceKey in SequenceDict.keys():
    m+=1
    
    OriSequence=SequenceDict[SequenceKey]
    Sequence=OriSequence.replace('?','X')
    # print("OriSequenceKey",SequenceKey, '\n',"OriSequenceValue:", OriSequence)


    # integer encode input data
    integer_encoded = [char_to_int[char] for char in Sequence]
    # if (char_to_int['>'] in integer_encoded):
    #     print(SequenceKey," has >")
    # if (char_to_int['2'] in integer_encoded):
    #     print(SequenceKey," has 2")
    # print(integer_encoded)

    # one hot encode
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(ResidueType))]
        letter[value] = 1
        onehot_encoded.append(letter)
    # print(onehot_encoded)
    SequenceDict[SequenceKey]=onehot_encoded
    # invert encoding

    if (m<10) :
        inverted = int_to_char[argmax(onehot_encoded[0])]
        print(SequenceKey,": ",inverted)
    # print(inverted)

# print([i for i in SequenceDict.keys()][:1])
# print([i for i in SequenceDict.values()][:1])
# np.save('ResidueTypeExtraction.npy', SequenceDict)
np.save('ResidueTypeExtraction_part2.npy', SequenceDict)