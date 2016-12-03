import tensorflow as tf
import re
import numpy as np
import math
import mido
import sys
import os
import time
import datetime
import random
from os.path import basename
from collections import Counter

SAVE_MODEL = False

if len(sys.argv) < 3:
	print("Give input file and output destination")
	sys.exit(1)

class Note:
	def __init__(self, m, t, d, v):
		self.mes = m
		self.absTime = t
		self.duration = d or 0
		self.endV = v

	def __hash__(self):
		return abs(hash(self.mes)) * 31 + self.duration * 7 + self.mes.time * 97

	def __str__(self):
		return str(self.mes) + " duration: " + str(self.duration) + " abs time: " + str(self.absTime) + " end velocity: " + str(self.endV)

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			return self.__dict__ == other.__dict__
		else:
			return False

	def __ne__(self, other):
		return not self.__eq__(other)


NONE_TOKEN = 0
START_TOKEN = 1
STOP_TOKEN = 2
SHARED_FEATURES = {'type', 'time'}
FEATURES_BY_TYPE = {
	'note_on': {'channel', 'note', 'velocity', 'duration'} | SHARED_FEATURES,
	'control_change': {'channel', 'control', 'value'} | SHARED_FEATURES,
	'program_change': {'channel', 'program'} | SHARED_FEATURES,
	'pitchwheel': {'channel', 'pitch'} | SHARED_FEATURES,
	'note_off': {'channel', 'note', 'velocity'} | SHARED_FEATURES,
}
FEATURES_SET = reduce(set.union, FEATURES_BY_TYPE.values())
FEATURES = sorted(list(FEATURES_SET))
ID_BY_FEATURE = {feature:i for i, feature in enumerate(FEATURES)}
FEATURE_BY_ID = {i:feature for i, feature in enumerate(FEATURES)}
numFeatures = len(FEATURES)

init_id_lookup = {
	None: NONE_TOKEN,
	START_TOKEN: START_TOKEN,
	STOP_TOKEN: STOP_TOKEN
}
init_value_lookup = {
	NONE_TOKEN: None,
	START_TOKEN: START_TOKEN,
	STOP_TOKEN: STOP_TOKEN
}
id_lookup_by_feature = {feature:init_id_lookup.copy() for feature in FEATURES}
value_lookup_by_feature = {feature:init_value_lookup.copy() for feature in FEATURES}

#mido.Message(msgType, channel, ..., time)
#msgType is note_on to start, note_on to end, time is diff between events

# message->note
def generate_noteseq_from_msgarray(msgArray):
	noteSeq = []
	time = 0
	# for mes in msgArray:
	# 	print mes
	for i in range(len(msgArray)):
		m = msgArray[i]
		if m.type is 'note_on' and m.velocity is 0:
			msgArray[i] = mido.Message('note_off',
						channel=m.channel,
						note=m.note,
						velocity=m.velocity,
						time=m.time)

	for i in range(len(msgArray)):
		curr = msgArray[i]
		time += curr.time
		if curr.type is 'note_on':
			duration = 0
			for j in range(i+1, len(msgArray)):
				duration += msgArray[j].time
				if msgArray[j].type is 'note_off' and curr.note is msgArray[j].note and curr.channel is msgArray[j].channel:
					noteSeq += [Note(curr, time, duration, msgArray[j].velocity)]
					break
		elif curr.type is 'control_change' or curr.type is 'program_change' or curr.type is 'pitchwheel':
			noteSeq += [Note(curr, time, 0, 0)]
	# for note in noteSeq:
	# 	print str(note.type) + ' ' + str(note.duration)
	return noteSeq

def getTime(msg):
	return msg.time

#Input: A sequence of notes of the format [[msgType, channel, time, duration, note, velocity, control, value, program, pitch], ...]
#Output: A sequence of myMessages which correspond to the MIDI stream
def generate_msgarray_from_noteseq(noteSeq):
	msgArray = []
	noteoffs = []
	for i in range(len(noteSeq)):
		curr = noteSeq[i]
		if curr.mes.type is 'note_on':
			msg = [curr.mes, mido.Message('note_off',
						channel=curr.mes.channel,
						note=curr.mes.note,
						velocity=curr.endV,
						time=curr.mes.time+curr.duration)]
		elif curr.mes.type is 'control_change' or curr.mes.type is 'program_change' or curr.mes.type is 'pitchwheel':
			msg = [curr.mes]
		msgArray += msg

	msgArray = sorted(msgArray, key=getTime)

	for i in range(len(msgArray)-1, 0, -1):
		if i < 1:
			msgArray[i].time = 0
		else:
			msgArray[i].time = msgArray[i].time - msgArray[i-1].time

	return msgArray


def generate_message_from_feature_ids(featureIds):
	features = {feature:value_lookup_by_feature[feature][featureIds[i]] for i, feature in FEATURE_BY_ID.items()}
	msgType = features['type']
	endV = 0
	if msgType is STOP_TOKEN:
		return STOP_TOKEN
	if msgType is 'note_on':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			note=features['note'],
			velocity=features['velocity'],
			time=features['time'])
		endV = features['velocity']
	elif msgType is 'control_change':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			control=features['control'],
			value=features['value'],
			time=features['time'])
	elif msgType is 'program_change':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			program=features['program'],
			time=features['time'])
	elif msgType is 'pitchwheel':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			pitch=features['pitch'],
			time=features['time'])
	else:
		msg = mido.Message(msgType)
	return Note(msg, None, features['duration'], endV);

def extract_features(message):
	if isinstance(message, int) and (message == START_TOKEN or message == STOP_TOKEN):
		return np.array([message for _ in range(numFeatures)])
	features = np.zeros(numFeatures)
	for i, feature in FEATURE_BY_ID.items():
		id_lookup = id_lookup_by_feature[feature]
		value_lookup = value_lookup_by_feature[feature]
		value = None
		if hasattr(message, feature):
			value = getattr(message, feature)
		elif hasattr(message.mes, feature):
			value = getattr(message.mes, feature)
		if value not in id_lookup:
			id_lookup[value] = len(id_lookup)
			value_lookup[len(value_lookup)] = value
		feature_id = id_lookup[value]
		features[i] = feature_id
	return features

def next_batch(token_ids, features, i, batch_size, num_steps):
	max_start = len(token_ids) - num_steps
	starts = np.random.randint(0, max_start, size=batch_size)
	ends = starts + num_steps
	indices = np.array([range(start, end) for start, end in zip(starts, ends)])
	features_x = np.take(features, indices, axis=0)
	features_y = np.take(features, indices+1, axis=0)
	return features_x, features_y

def tokenizeFile(path, songIndex):
	tokenized.append(list());
	for msg in mido.MidiFile(path):
		if not isinstance(msg, mido.MetaMessage):
			#if msg.type == 'note_off':
			#	msg = mido.Message('note_on', note=msg.note, velocity=0, time=msg.time)
			if msg.type in FEATURES_BY_TYPE:
				msg.time = int(msg.time * 90 * 4 * 2)
				tokenized[songIndex].append(msg)

def make_embedding(embedSize, featureSize):
	return tf.Variable(
		tf.random_uniform([featureSize, embedSize],
		minval=0,
		maxval=1,
		dtype=tf.float32,
		seed=0))

def make_embeddings(embedSize, featureSizes):
	return [make_embedding(embedSize, featureSize) for featureSize in featureSizes]

def embeddings_lookup(E, x):
	e_list = [tf.nn.embedding_lookup(E_i, x[:, :, i]) for i, E_i in enumerate(E)]
	return tf.concat(2, e_list)

def make_feature_predictor(lstmSize, rnn2D, y, featureSize):
	# Forward pass
	W = tf.Variable(tf.truncated_normal([lstmSize, featureSize], stddev=0.1))
	B = tf.Variable(tf.constant(0.1, shape=[featureSize]))
	logits = tf.matmul(rnn2D, W) + B
	# Compute loss
	y1D = tf.reshape(y, [-1]) # -1 means flatten
	weights = tf.tile(tf.ones(shape=[1]), [tf.shape(y1D)[0]])
	# weights = tf.Variable(tf.constant(1.0, shape=weights_shape))
	loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [y1D], [weights])
	return W, B, logits, loss

def make_feature_predictors(lstmSize, rnn2D, y, featureSizes):
	return zip(*[make_feature_predictor(lstmSize, rnn2D, y[:, :, i], featureSize) for i, featureSize in enumerate(featureSizes)])

def sample(logits):
	if logits.min() is logits.max():
		return np.random.choice(len(logits, 1))[0]
	logits = logits - logits.min()
	sq_logits = np.multiply(logits, logits)
	prob_dist = np.divide(sq_logits, sq_logits.sum())
	feature_id = np.random.choice(len(logits), 1, p=prob_dist)[0]
	return feature_id

def get_next_token_feature_id(feature, msgType, featureLogits):
	if msgType is STOP_TOKEN:
		return STOP_TOKEN
	if feature not in FEATURES_BY_TYPE[msgType]:
		return NONE_TOKEN
	logits = featureLogits[0][2:] # Assumes None is 0 and START is 1
	## Sample from distribution using logits
	next_id = sample(logits) + 2 # Assumes None is 0 and START is 1
	## Pick most likely logit
	# next_id = np.argmax(logits) + 2 # Assumes None is 0 and START is 1
	return next_id

def get_next_token_feature_ids(batchLogits):
	typeFeatureId = ID_BY_FEATURE['type']
	msgTypeId = sample(batchLogits[typeFeatureId][0][2:]) + 2 # Assumes None is 0 and START is 1
	msgType = value_lookup_by_feature['type'][msgTypeId]
	featureIds = [get_next_token_feature_id(feature, msgType, batchLogits[i]) for i, feature in FEATURE_BY_ID.items()]
	featureIds[typeFeatureId] = msgTypeId
	return featureIds

def save_output(out_path, msgs):
	mid = mido.MidiFile()
	track = mido.MidiTrack()
	mid.tracks.append(track)
	track.append(mido.Message('program_change', program=0, time=0))
	mid.ticks_per_beat = 300
	for msg in msgs:
		track.append(msg)
	mid.save(out_path)

tokenized = list()
if os.path.isdir(sys.argv[1]):
	i = 0
	for filename in os.listdir(sys.argv[1]):
		tokenizeFile(sys.argv[1] + "/" + filename, i)
		i += 1
else:
	tokenizeFile(sys.argv[1], 0)

# Convert the notes to ints
counts = Counter([msg for song in tokenized for msg in generate_noteseq_from_msgarray(song)])
vocab = dict()
lookup = dict()
lookup[START_TOKEN] = START_TOKEN
lookup[STOP_TOKEN] = STOP_TOKEN
vocabSize = len(counts)+2 # include START_TOKEN/STOP_TOKEN
index = 2 # account for START_TOKEN (0) and STOP_TOKEN (1)
for word in counts:
	print word
	vocab[word] = index
	lookup[index] = word
	index += 1


# Build train/test int lists
trainInts1 = list()
trainFeatures1 = list()
for song in tokenized:
	print song[0]
	newsong = generate_noteseq_from_msgarray(song)
	# b = True
	# barr = []
	# for i in range(len(newsong)):
	# 	if i >= len(song):
	# 		break
	# 	#print("i: " + str(i))
	# 	#print("song: " + str(song[i]))
	# 	#print("nsng: " + str(newsong[i]))
	# 	if song[i] != newsong[i]:
	# 		barr.append(i)
	# 		b = False
	# print(b)
	# print(barr)
	# save_output(sys.argv[2], newsong)

	trainInts1.append(START_TOKEN)
	trainFeatures1.append(extract_features(START_TOKEN))
	for word in newsong:
		# print word
		# print vocab[word]

		trainInts1.append(vocab[word])
		trainFeatures1.append(extract_features(word))
	trainInts1.append(STOP_TOKEN) # append STOP_TOKEN token
	trainFeatures1.append(extract_features(STOP_TOKEN))
trainInts = np.array(trainInts1)
trainFeatures = np.array(trainFeatures1)

# Inputs and outputs
batchSize = 2
numSteps = 4
x = tf.placeholder(tf.int32, [batchSize, None, numFeatures])
y = tf.placeholder(tf.int32, [batchSize, None, numFeatures])
keepProb = tf.placeholder(tf.float32)

featureSizes = [len(id_lookup_by_feature[feature]) for feature in FEATURES]

# Embedding matrix
embedSize = 128
E = make_embeddings(embedSize, featureSizes)
e = embeddings_lookup(E, x)
eDrop = tf.nn.dropout(e, keepProb)

lstmSize = 256
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstmSize, state_is_tuple=True)
initialState = lstm.zero_state(batchSize, tf.float32)

rnn, outst = dyrnn = tf.nn.dynamic_rnn(lstm, eDrop, initial_state=initialState)
rnn2D = tf.reshape(rnn, [-1, lstmSize])

Ws, Bs, logits, losses = make_feature_predictors(lstmSize, rnn2D, y, featureSizes)
avg_loss = tf.reduce_sum(losses)/(batchSize*numSteps)
perplexity = tf.exp(avg_loss)
abs_W_means = [tf.reduce_mean(tf.abs(W)) for W in Ws]
abs_B_means = [tf.reduce_mean(tf.abs(B)) for B in Bs]
regularize_W = tf.add_n(abs_W_means)/numFeatures
regularize_B = tf.add_n(abs_B_means)/numFeatures
regularization = 50 * regularize_W + 50 * regularize_B

# Setup training
sess = tf.InteractiveSession()
trainStep = tf.train.AdamOptimizer(1e-2).minimize(perplexity + regularization)
sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

if len(sys.argv) > 3:
	saver.restore(sess, sys.argv[3])
else:
	NUM_EPOCHS = 2000
	for e in range(NUM_EPOCHS):
		i = 0
		state = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
		X = 0
		while X + batchSize * numSteps + 1 <= len(trainInts):
			batch_x, batch_y = next_batch(trainInts, trainFeatures, i, batchSize, numSteps)
			state, _, perp, reg = sess.run([outst, trainStep, perplexity, regularization],
			feed_dict={
				x: batch_x,
				y: batch_y,
				keepProb: 0.5,
				initialState: state
			})
			X += batchSize*numSteps
			i += 1
		if e % 20 == 0:
			print(e, perp, reg)
	if SAVE_MODEL:
		ts = time.time()
		timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H_%M_%S')
		input_name = basename(sys.argv[1]).split('.')[0]
		save_path = saver.save(sess, "models/%s_%s_%s.ckpt" % (input_name, NUM_EPOCHS, timestamp))
		print("Model saved to %s" % save_path)

state = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
genNotes = list()
nextToken = START_TOKEN
i = 0
while nextToken != STOP_TOKEN:
	tokenFeatures = extract_features(nextToken)
	currFeatures = np.tile(tokenFeatures, (batchSize, 1, 1))
	state, batchLogits = sess.run([outst, logits],
		feed_dict={x: currFeatures, keepProb: 1.0, initialState: state})
	nextFeatureIds = get_next_token_feature_ids(batchLogits)
	nextToken = generate_message_from_feature_ids(nextFeatureIds)
	if nextToken != STOP_TOKEN:
		genNotes.append(nextToken)
	i += 1

absTime = 0
for note in genNotes:
	note.absTime = absTime
	absTime += note.mes.time

genMsgs = generate_msgarray_from_noteseq(genNotes)

for msg in genMsgs:
	print msg

save_output(sys.argv[2], genMsgs)
