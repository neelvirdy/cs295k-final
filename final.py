import tensorflow as tf
import re
import numpy as np
import math
import mido
import sys
from collections import Counter

if len(sys.argv) < 3:
	print "Give input file and output destination"
	sys.exit(1)

class MyMessage:
	def __init__(self, m):
		self.mes = m
		self.mes.time = int(self.mes.time * 90 * 4 * 2)

	def __hash__(self):
		return hash(self.mes.hex()) * 31 + hash(self.mes.time) * 7

tokenized = list()
for msg in mido.MidiFile(sys.argv[1]):
	if not isinstance(msg, mido.MetaMessage):
		tokenized.append(MyMessage(msg))

# Convert the messages to ints
counts = Counter(tokenized)
vocab = dict()
lookup = dict()
vocabSize = len(counts)+2 # include START/STOP
index = 1 # starts at 1 to account for START token
for word in counts:
	vocab[word] = index
	lookup[index] = word
	index += 1
stopToken = index

# Build train/test int lists
trainInts1 = list()
trainInts1.append(0) # prepend START token
for word in tokenized:
	trainInts1.append(vocab[word])
trainInts1.append(stopToken) # append STOP token

trainInts = np.array(trainInts1)

# Inputs and outputs
batchSize = 30
numSteps = 30
x = tf.placeholder(tf.int32, [batchSize, None])
y = tf.placeholder(tf.int32, [batchSize, None])
keepProb = tf.placeholder(tf.float32)

# Embedding matrix
embedSize = 50
E = tf.Variable(tf.random_uniform([vocabSize, embedSize], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd = tf.nn.embedding_lookup(E, x)

lstmSize = 256
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstmSize, state_is_tuple=True)
initialState = lstm.zero_state(batchSize, tf.float32)

# Forward pass
W1 = tf.Variable(tf.truncated_normal([lstmSize, vocabSize], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, shape=[vocabSize]))

embdDrop = tf.nn.dropout(embd, keepProb)

rnn, outst = dyrnn = tf.nn.dynamic_rnn(lstm, embdDrop, initial_state=initialState)
rnn2D = tf.reshape(rnn, [-1, lstmSize])

logits = tf.matmul(rnn2D, W1) + B1

# Compute loss
weights = tf.Variable(tf.constant(1.0, shape=[batchSize*numSteps]))
y1D = tf.reshape(y, [batchSize*numSteps])
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [y1D], [weights])
l = tf.reduce_sum(loss)/(batchSize*numSteps)
perplexity = tf.exp(l)

# Setup training
sess = tf.InteractiveSession()
trainStep = tf.train.AdamOptimizer(1e-4).minimize(perplexity)
sess.run(tf.initialize_all_variables())

NUM_EPOCHS = 1000
for e in range(NUM_EPOCHS):
	state = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
	X = 0
	while X + batchSize * numSteps + 1 < len(trainInts):
		state, _, perp = sess.run([outst, trainStep, perplexity],
			feed_dict={ x: np.reshape(trainInts[X:X+batchSize*numSteps], (batchSize, numSteps)),
				y: np.reshape(trainInts[X+1:X+1+batchSize*numSteps], (batchSize, numSteps)),
				keepProb: 0.5,
				initialState: state })
		X += batchSize*numSteps
	print e, perp

state = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
curr_word_ids = np.empty((batchSize, 1))
curr_word_ids.fill(0)
gen_words = list()
nextToken = 0
while nextToken != stopToken:
    state, batchLogits = sess.run([outst, logits],
        feed_dict={x: curr_word_ids, keepProb: 1.0, initialState: state })
    logs = batchLogits[0]
    my_logits = logs - logs.min()
    my_logits[0] = 0
    prob_dist = np.divide(my_logits, my_logits.sum())
    nextToken = np.random.choice(vocabSize, 1, p=prob_dist)[0]
    curr_word_ids.fill(nextToken)
    if nextToken != stopToken:
    	gen_words.append(lookup[nextToken])

mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message('program_change', program=0, time=0))

for msg in gen_words:
	print(msg.mes)
	track.append(msg.mes)

mid.save(sys.argv[2])
