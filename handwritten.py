from __future__ import division
from mnist import read, show, save_image
import numpy as np
import cv2
from enum import Enum
import time
import pickle
import matplotlib as mpl
from matplotlib import pyplot


np.set_printoptions(threshold=np.nan)

def getTrainData(count, digit):
    #count is the number of training data
    #digit is the specific nimber between 0 and 9 from training dataset
    trainingData = np.zeros((count,400), int)
    i=0
    for img in read("training"):
        if i<count:
            if img[0]==digit: #This is the label
                row, col = img[1].shape
                im = img[1]
                im = im.reshape(28,28)
                im2 = cv2.resize(im,(20,20), interpolation=cv2.INTER_CUBIC)
                im2 = im2.reshape(1,400)
                trainingData[i,:] = im2

                i+=1

    trainingData = (trainingData>125).astype(int)
    return trainingData
data = np.zeros((10,20,400), int)
for i in range(10):
    data[i,:] = getTrainData(20,i)

# !/usr/bin/env python



Clamp = Enum('Clamp', 'VISIBLE_UNITS NONE INPUT_UNITS')


class Step:
    def __init__(self, temperature, epochs):
        self.temperature = temperature
        self.epochs = epochs


numInputUnits = 400
numOutputUnits = 400
numHiddenUnits = 224

numVisibleUnits = numInputUnits + numOutputUnits
numUnits = numVisibleUnits + numHiddenUnits

annealingSchedule = [Step(20., 4),
                     Step(15., 4),
                     Step(12., 4),
                     Step(10., 4),
                     Step(5.,8)]

testAnnealingSchedule = [Step(35., 10),
                         Step(30., 10),
                         Step(25., 10),
                         Step(20., 10),
                         Step(15., 10),
                         Step(12., 10),
                         Step(10., 20),
                         Step(5., 20),
                         Step(1.,8)]

coocurranceCycle = Step(5., 10)

# with open('weights_400.pkl', 'rb') as f:
#     weights = pickle.load(f)

weights = np.zeros((numUnits, numUnits))
states = np.zeros(numUnits)
energy = np.zeros(numUnits)

connections = np.zeros((numUnits, numUnits), dtype=np.int)
for i in range(numInputUnits):
    for j in range(i + 1, numInputUnits):
        connections[i, j] = 1
    for j in range(1, numHiddenUnits + 1):
        connections[i, -j] = 1

for i in range(numOutputUnits):
    for j in range(i + 1, numOutputUnits):
        connections[i + numInputUnits, j + numInputUnits] = 1
    for j in range(1, numHiddenUnits + 1):
        connections[i + numInputUnits, -j] = 1

for i in range(numHiddenUnits, 0, -1):
    for j in range(i - 1, 0, -1):
        connections[-i, -j] = 1

valid = np.nonzero(connections)
numConnections = np.size(valid[0])
connections[valid] = np.arange(1, numConnections + 1)
connections = connections + connections.T - 1

def propagate(temperature, clamp):
    global energy, states, weights

    if clamp == Clamp.VISIBLE_UNITS:
        numUnitsToSelect = numHiddenUnits
    elif clamp == Clamp.NONE:
        numUnitsToSelect = numUnits
    else:
        numUnitsToSelect = numHiddenUnits + numOutputUnits

    for i in range(numUnitsToSelect):
        # Calculating the energy of a randomly selected unit
        unit = numUnits - np.random.randint(1, numUnitsToSelect + 1)
        energy[unit] = np.dot(weights[unit, :], states)

        p = 1. / (1. + np.exp(-energy[unit] / temperature))
        states[unit] = 1. if np.random.uniform() <= p else 0

        # Equivalent Energy calculation:
        # unit = numUnits - np.random.randint(1, numUnitsToSelect + 1)
        # energy[unit] = (1 - 2 * states[unit]) * (np.dot(weights[unit, :], states))
        #
        # p = 1. / (1. + np.exp(-energy[unit] / temperature))
        # states[unit] = (1. - states[unit]) if np.random.uniform() <= p else states[unit]

def anneal(annealingSchedule, clamp):
    for step in annealingSchedule:
        for epoch in range(step.epochs):
            propagate(step.temperature, clamp)


def sumCoocurrance(clamp):
    sums = np.zeros(numConnections)
    for epoch in range(coocurranceCycle.epochs):
        propagate(coocurranceCycle.temperature, clamp)
        for i in range(numUnits):
            if (states[i] == 1):
                for j in range(i + 1, numUnits):
                    if (connections[i, j] > -1 and states[j] == 1):
                        sums[connections[i, j]] += 1
    return sums


def updateWeights(pplus, pminus):
    global weights
    for i in range(numUnits):
        for j in range(i + 1, numUnits):
            if connections[i, j] > -1:
                index = connections[i, j]
                weights[i, j] += 2 * np.sign(pplus[index] - pminus[index])
                weights[j, i] = weights[i, j]


def recall(pattern):
    global states

    # Setting pattern to recall
    states[0:numInputUnits] = pattern

    # Assigning random values to the hidden and output states
    states[-(numHiddenUnits + numOutputUnits):] = np.random.choice([0, 1], numHiddenUnits + numOutputUnits)

    anneal(testAnnealingSchedule, Clamp.INPUT_UNITS)

    return states[numInputUnits:numInputUnits + numOutputUnits]


def addNoise(pattern):
    probabilities = 0.8 * pattern + 0.05
    uniform = np.random.random(numVisibleUnits)
    return (uniform < probabilities).astype(int)


def learn(patterns):
    global states, weights

    numPatterns = patterns.shape[0]
    trials = numPatterns * coocurranceCycle.epochs
    weights = np.zeros((numUnits, numUnits))

    for i in range(3000):
        print(i)
        start=time.time()
        # Positive phase
        pplus = np.zeros(numConnections)
        for pattern in patterns:
            # Setting visible units values (inputs and outputs)
            states[0:numVisibleUnits] = addNoise(pattern)

            # Assigning random values to the hidden units
            states[-numHiddenUnits:] = np.random.choice([0, 1], numHiddenUnits)

            anneal(annealingSchedule, Clamp.VISIBLE_UNITS)
            pplus += sumCoocurrance(Clamp.VISIBLE_UNITS)
        pplus /= trials

        # Negative phase
        states = np.random.choice([0, 1], numUnits)
        anneal(annealingSchedule, Clamp.NONE)
        pminus = sumCoocurrance(Clamp.NONE) / coocurranceCycle.epochs
        print("iteration time: ", time.time()-start)
        updateWeights(pplus, pminus)
        if np.remainder(i,100)==0:
            np.savez("weights_iter_"+str(i), weights)


patterns = np.empty((1,400), int)
patrns = np.empty((1,400), int)
for i in range(10):
    patrns = np.asarray(data[i,:, :])
    #print(patrns)
    patterns = np.append(patterns, patrns, axis = 0)

# patterns = np.append(patterns, patterns, axis=1)
# start = time.time()
#
# learn(patterns)
#
# print("total train time=", time.time() - start)
# print(weights)
# print("Time to train= ", time.time()-start)
# with open('weights_1000.pkl','wb') as f:
#     pickle.dump(weights, f)
wpath = "E:\\Fujitsu\\HandWrittenDigits"
for i in range(12):
    if i==0:
        wfile = wpath+"\\weights_iter_"+str(i)+".npz"
    else:
        wfile = wpath+"\\weights_iter_"+str(i)+"00.npz"
    npzfile = np.load(wfile)
    weights = npzfile['arr_0']
    recovered = recall(patrns[1,:])

    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(recovered.reshape(20,20), cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    # pyplot.show()
    pyplot.savefig("recovered_"+str(i))
    pyplot.close()
save_image(patrns[1,:].reshape(20,20), 'original')


