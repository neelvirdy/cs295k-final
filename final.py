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
		if m.type == 'note_off':
			m = mido.Message('note_on', note=m.note, velocity=0, time=m.time)
		self.mes = m
		self.mes.time = int(self.mes.time * 90 * 4 * 2)

	def __hash__(self):
		return hash(self.mes.hex()) * 31 + hash(self.mes.time) * 7

id_by_msg_type = {}
msg_type_by_id = {}
id_by_channel = {}
channel_by_id = {}
id_by_note = {}
note_by_id = {}
id_by_velocity = {}
velocity_by_id = {}

def extract_features(messageId, lookup):
	if messageId == 0 or messageId == stopToken:
		return np.zeros(5)
	message = lookup[messageId].mes
	print message
	if message.type not in id_by_msg_type:
		id_by_msg_type[message.type] = len(id_by_msg_type)+1
		msg_type_by_id[len(msg_type_by_id)] = message.type
	if hasattr(message, 'channel') and message.channel not in id_by_channel:
		id_by_channel[message.channel] = len(id_by_channel)+1
		channel_by_id[len(channel_by_id)] = message.channel
	if hasattr(message, 'note') and message.note not in id_by_note:
		id_by_note[message.note] = len(id_by_note)+1
		note_by_id[len(note_by_id)] = message.note
	if hasattr(message, 'velocity') and message.velocity not in id_by_velocity:
		id_by_velocity[message.velocity] = len(id_by_velocity)+1
		velocity_by_id[len(velocity_by_id)] = message.velocity
	return np.array([
		messageId,
		id_by_msg_type[message.type],
		id_by_channel[message.channel] if hasattr(message, 'channel') else 0,
		id_by_note[message.note] if hasattr(message, 'note') else 0,
		id_by_velocity[message.velocity] if hasattr(message, 'velocity') else 0,
	])

def next_batch(token_ids, features, i, batch_size, num_steps):
	max_start = len(token_ids) - batch_size - 1
	starts = np.random.randint(0, max_start, size=batch_size)
	ends = starts + num_steps
	indices = np.array([range(start, end) for start, end in zip(starts, ends)])
	features_x = np.take(features, indices, axis=0)
	ids_y = np.take(token_ids, indices+1)
	print indices, indices+1
	return features_x, ids_y

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

trainFeatures1 = list()
for trainInt in trainInts:
	trainFeatures1.append(extract_features(trainInt, lookup))
# trainFeatures = np.vectorize(extract_features)(trainInts, lookup)
trainFeatures = np.array(trainFeatures1)

typeSize = len(id_by_msg_type)+1
channelSize = len(id_by_channel)+1
noteSize = len(id_by_note)+1
velocitySize = len(id_by_velocity)+1

# Inputs and outputs
numFeatures = len(trainFeatures[0])
batchSize = 5
numSteps = 10
x = tf.placeholder(tf.int32, [batchSize, None, numFeatures])
y = tf.placeholder(tf.int32, [batchSize, None])
keepProb = tf.placeholder(tf.float32)

# Embedding matrix
embedSize = 50
E = tf.Variable(tf.random_uniform([vocabSize, embedSize], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd = tf.nn.embedding_lookup(E, x[:, :, 0])

E_type = tf.Variable(tf.random_uniform([typeSize, embedSize], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd_type = tf.nn.embedding_lookup(E_type, x[:, :, 1])

E_channel = tf.Variable(tf.random_uniform([channelSize, embedSize], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd_channel = tf.nn.embedding_lookup(E_channel, x[:, :, 2])

E_note = tf.Variable(tf.random_uniform([noteSize, embedSize], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd_note = tf.nn.embedding_lookup(E_note, x[:, :, 3])

E_velocity = tf.Variable(tf.random_uniform([velocitySize, embedSize], minval=-1, maxval=1, dtype=tf.float32, seed=0))
embd_velocity = tf.nn.embedding_lookup(E_velocity, x[:, :, 4])

e = tf.concat(2, [embd, embd_type, embd_channel, embd_note, embd_velocity])

lstmSize = 256
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstmSize, state_is_tuple=True)
initialState = lstm.zero_state(batchSize, tf.float32)

# Forward pass
W1 = tf.Variable(tf.truncated_normal([lstmSize, vocabSize], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, shape=[vocabSize]))

eDrop = tf.nn.dropout(e, keepProb)

rnn, outst = dyrnn = tf.nn.dynamic_rnn(lstm, eDrop, initial_state=initialState)
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

NUM_EPOCHS = 2000
for e in range(NUM_EPOCHS):
	i = 0
	state = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
	X = 0
	while X + batchSize * numSteps + 1 < len(trainInts):
		batch_x, batch_y = next_batch(trainInts, trainFeatures, i, batchSize, numSteps)
		state, _, perp = sess.run([outst, trainStep, perplexity],
		feed_dict={
			x: batch_x,
			y: batch_y,
			keepProb: 0.5,
			initialState: state
		})
		X += batchSize*numSteps
		i += 1
	if e % 20 == 0:
		print e, perp

state = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
curr_word_id = 0
curr_features = np.tile(extract_features(curr_word_id, lookup), (batchSize, 1, 1))
gen_words = list()
nextToken = 0
while nextToken != stopToken:
	state, batchLogits = sess.run([outst, logits],
		feed_dict={x: curr_features, keepProb: 1.0, initialState: state })
	logs = batchLogits[0]
	my_logits = logs - logs.min()
	my_logits[0] = 0
	sq_my_logits = np.multiply(my_logits, my_logits)
	prob_dist = np.divide(sq_my_logits, sq_my_logits.sum())
	nextToken = np.random.choice(vocabSize, 1, p=prob_dist)[0]
	# nextToken = np.argmax(my_logits)
	curr_features = np.tile(extract_features(nextToken, lookup), (batchSize, 1, 1))
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
