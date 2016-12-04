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

SAMPLE = True
SAVE_MODEL = False

if len(sys.argv) < 3:
	print("Give input file and output destination")
	sys.exit(1)

class Note:
	def __init__(self, m, t, d, v, interval, octave):
		self.mes = m
		self.absTime = t
		self.duration = d or 0
		self.endV = v
		self.interval = interval
		self.octave = octave

	def __hash__(self):
		return abs(hash(self.mes)) * 31 + self.duration * 7 + self.mes.time * 97

	def __str__(self):
		return '%s duration=%s absTime=%s endV=%s interval=%s octave=%s' % (self.mes, self.duration, self.absTime, self.endV, self.interval, self.octave)

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
FEATURES_BY_TYPE = {
	'note_on': {'type', 'channel', 'velocity', 'time', 'duration', 'interval', 'octave'},
	'control_change': {'type', 'channel', 'control', 'value', 'time'},
	'program_change': {'type', 'channel', 'program', 'time'},
	'pitchwheel': {'type', 'channel', 'pitch', 'time'},
	'key_signature': {'type', 'key', 'time'},
	'set_tempo': {'type', 'tempo', 'time'}
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

# returns semitones from key, not the interval roman numeral
semitones_per_octave = 12
def get_interval(note, key):
	interval = (note - key) % semitones_per_octave
	return interval

def get_octave(note):
	return int(note / semitones_per_octave)

def get_note(key, octave, interval):
	return key + octave * semitones_per_octave + interval

#mido.Message(msgType, channel, ..., time)
#msgType is note_on to start, note_on to end, time is diff between events

# message->note
def generate_noteseq_from_msgarray(msgArray):
	noteSeq = []
	time = 0
	key = get_note_id('C')
	# for mes in msgArray:
	# 	print mes
	for i in range(len(msgArray)):
		m = msgArray[i]
		if m.type == 'note_on' and m.velocity == 0:
			msgArray[i] = mido.Message('note_off',
						channel=m.channel,
						note=m.note,
						velocity=m.velocity,
						time=m.time)

	for i in range(len(msgArray)):
		curr = msgArray[i]
		if curr.type == 'key_signature':
			key = get_note_id(curr.key)
		time += curr.time
		if curr.type == 'note_on':
			interval = get_interval(curr.note, key)
			octave = get_octave(curr.note)
			duration = 0
			for j in range(i+1, len(msgArray)):
				duration += msgArray[j].time
				if msgArray[j].type == 'note_off' and curr.note == msgArray[j].note and curr.channel == msgArray[j].channel:
					noteSeq += [Note(curr, time, duration, msgArray[j].velocity, interval, octave)]
					break
		elif curr.type == 'control_change' or curr.type == 'program_change' or curr.type == 'pitchwheel':
			noteSeq += [Note(curr, time, 0, 0, 0, 0)]
	# for note in noteSeq:
	# 	print str(note.type) + ' ' + str(note.duration)
	for i in range(1, len(noteSeq)):
		noteSeq[i].mes.time = noteSeq[i].absTime - noteSeq[i-1].absTime
	return noteSeq

def getTime(msg):
	return msg.time

#Input: A sequence of notes of the format [[msgType, channel, time, duration, note, velocity, control, value, program, pitch], ...]
#Output: A sequence of myMessages which correspond to the MIDI stream
def generate_msgarray_from_noteseq(noteSeq):
	absTime = 0
	for note in noteSeq:
		note.absTime = absTime
		absTime += note.mes.time

	msgArray = []
	noteoffs = []
	key = get_note_id('C')
	for i in range(len(noteSeq)):
		curr = noteSeq[i]
		curr.mes.time = curr.absTime
		if curr.mes.type == 'key_signature':
			key = get_note_id(curr.mes.key)
		if curr.mes.type == 'note_on':
			curr.mes.note = get_note(key, curr.octave, curr.interval)
			# convert from beats to ticks
			noteOnTime = round(curr.mes.time * medianTicksPerBeat)
			noteOffTime = round((curr.mes.time + curr.duration) * medianTicksPerBeat)
			msg = [curr.mes, mido.Message('note_off',
						channel=curr.mes.channel,
						note=curr.mes.note,
						velocity=curr.endV,
						time=noteOffTime)]
		elif curr.mes.type == 'control_change' or curr.mes.type == 'program_change' or curr.mes.type == 'pitchwheel':
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
	if msgType == STOP_TOKEN:
		return STOP_TOKEN
	if msgType == 'note_on':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			note=0, # this gets set later
			velocity=features['velocity'],
			time=features['time'])
		endV = features['velocity']
	elif msgType == 'control_change':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			control=features['control'],
			value=features['value'],
			time=features['time'])
	elif msgType == 'program_change':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			program=features['program'],
			time=features['time'])
	elif msgType == 'pitchwheel':
		msg = mido.Message(
			msgType,
			channel=features['channel'],
			pitch=features['pitch'],
			time=features['time'])
	else:
		msg = mido.Message(msgType)
	return Note(msg, None, features['duration'], endV, features['interval'], features['octave']);

def extract_features(message):
	if isinstance(message, int) and (message == START_TOKEN or message == STOP_TOKEN):
		features = np.array([message for _ in range(numFeatures)])
		return features
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

def next_batch(features, i, batch_size, num_steps):
	max_start = len(features) - num_steps
	starts = np.random.randint(0, max_start, size=batch_size)
	ends = starts + num_steps
	indices = np.array([range(start, end) for start, end in zip(starts, ends)])
	features_x = np.take(features, indices, axis=0)
	features_y = np.take(features, indices+1, axis=0)
	return features_x, features_y

note_id_by_name = {
	'C': 0,
	'C#': 1,
	'Db': 1,
	'D': 2,
	'D#': 3,
	'Eb': 3,
	'E': 4,
	'F': 5,
	'F#': 6,
	'Gb': 6,
	'G': 7,
	'G#': 8,
	'Ab': 8,
	'A': 9,
	'A#': 10,
	'Bb': 10,
	'B': 11,
	'Cb': 11
}

def get_note_id(key):
	minor = key[-1] == 'm'
	if minor:
		note_name = key[:-1]
	else:
		note_name = key
	note_id = note_id_by_name[note_name]
	return note_id

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
	loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [y1D], [weights])
	return W, B, logits, loss

def make_feature_predictors(lstmSize, rnn2D, y, featureSizes):
	return zip(*[make_feature_predictor(lstmSize, rnn2D, y[:, :, i], featureSize) for i, featureSize in enumerate(featureSizes)])

def sample(logits):
	if logits.min() == logits.max():
		return np.random.choice(len(logits), 1)[0]
	logits = logits - logits.min()
	sq_logits = np.multiply(logits, logits)
	prob_dist = np.divide(sq_logits, sq_logits.sum())
	feature_id = np.random.choice(len(logits), 1, p=prob_dist)[0]
	return feature_id

def get_next_token_feature_id(feature, msgType, featureLogits):
	if msgType == STOP_TOKEN:
		return STOP_TOKEN
	if feature not in FEATURES_BY_TYPE[msgType]:
		return NONE_TOKEN
	logits = featureLogits[0]
	if SAMPLE:
		next_id = sample(logits[3:]) + 3 # Assumes 0:3 is NONE START STOP
	else:
		next_id = np.argmax(logits[3:]) + 3 # Assumes 0:3 is NONE START STOP
	return next_id

def get_next_token_feature_ids(batchLogits):
	typeFeatureId = ID_BY_FEATURE['type']
	msgTypeLogits = batchLogits[typeFeatureId][0]
	if SAMPLE:
		msgTypeId = sample(msgTypeLogits[2:]) + 2 # Assumes 0:2 is NONE START
	else:
		msgTypeId = np.argmax(msgTypeLogits[2:]) + 2
	msgType = value_lookup_by_feature['type'][msgTypeId]
	featureIds = [get_next_token_feature_id(feature, msgType, batchLogits[i]) for i, feature in FEATURE_BY_ID.items()]
	featureIds[typeFeatureId] = msgTypeId
	return featureIds

trainBpms = []
medianBpm = 120
trainTicksPerBeat = []
medianTicksPerBeat = 480

def save_output(out_path, msgs):
	mid = mido.MidiFile()
	track = mido.MidiTrack()
	mid.tracks.append(track)
	track.append(mido.Message('program_change', program=0, time=0))
	medianTempo = mido.bpm2tempo(medianBpm)
	track.append(mido.MetaMessage('set_tempo', tempo=medianTempo, time=0))
	mid.ticks_per_beat = medianTicksPerBeat
	for msg in msgs:
		track.append(msg)
	mid.save(out_path)

def tokenizeFile(path, songIndex):
	tokenized.append(list())
	newSong = list()
	bpm = 120
	midoFile = mido.MidiFile(path)
	ticksPerBeat = midoFile.ticks_per_beat
	trainTicksPerBeat.append(ticksPerBeat)
	for msg in midoFile:
		if msg.type in FEATURES_BY_TYPE or msg.type == 'note_off':
			if msg.type == 'set_tempo':
				bpm = int(mido.tempo2bpm(msg.tempo))
				trainBpms.append(bpm)
			if hasattr(msg, 'time'):
				# convert from seconds to beats
				beats = msg.time * (bpm/60)
				beatsRounded = round(beats * 128) / 128
				msg.time = beatsRounded
			if hasattr(msg, 'velocity'):
				msg.velocity = int(msg.velocity / 5) * 5
			print msg
			newSong.append(msg)
	tokenized[songIndex] = generate_noteseq_from_msgarray(newSong)

tokenized = list()
if os.path.isdir(sys.argv[1]):
	i = 0
	for filename in os.listdir(sys.argv[1]):
		tokenizeFile(sys.argv[1] + "/" + filename, i)
		i += 1
else:
	tokenizeFile(sys.argv[1], 0)

if len(trainBpms):
	medianBpm = sorted(trainBpms)[int(len(trainBpms)/2)]
if len(trainTicksPerBeat):
	medianTicksPerBeat = sorted(trainTicksPerBeat)[int(len(trainTicksPerBeat)/2)]

print "tempo=%s ticksPerBeat=%s" % (medianBpm, medianTicksPerBeat)

# Build train/test int lists
trainFeatures1 = list()
for song in tokenized:
	trainFeatures1.append(extract_features(START_TOKEN))
	for msg in song:
		trainFeatures1.append(extract_features(msg))
	trainFeatures1.append(extract_features(STOP_TOKEN))
trainFeatures = np.array(trainFeatures1)

# Inputs and outputs
batchSize = 4
numSteps = 8
x = tf.placeholder(tf.int32, [batchSize, None, numFeatures])
y = tf.placeholder(tf.int32, [batchSize, None, numFeatures])
keepProb = tf.placeholder(tf.float32)

featureSizes = [len(id_lookup_by_feature[feature]) for feature in FEATURES]

# Embedding matrix
embedSize = 64
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
trainStep = tf.train.AdamOptimizer(1e-4).minimize(perplexity + regularization)
sess.run(tf.initialize_all_variables())

for feature, lookup in id_lookup_by_feature.items():
	print feature, len(lookup)

saver = tf.train.Saver()

if len(sys.argv) > 3:
	saver.restore(sess, sys.argv[3])
else:
	NUM_EPOCHS = 800
	for e in range(NUM_EPOCHS):
		i = 0
		state = (np.zeros([batchSize, lstmSize]), np.zeros([batchSize, lstmSize]))
		X = 0
		while X + batchSize * numSteps + 1 <= len(trainFeatures):
			batch_x, batch_y = next_batch(trainFeatures, i, batchSize, numSteps)
			_, _, perp, reg = sess.run([outst, trainStep, perplexity, regularization],
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
while nextToken != STOP_TOKEN:
	tokenFeatures = extract_features(nextToken)
	currFeatures = np.tile(tokenFeatures, (batchSize, 1, 1))
	state, batchLogits = sess.run([outst, logits],
		feed_dict={x: currFeatures, keepProb: 1.0, initialState: state})
	nextFeatureIds = get_next_token_feature_ids(batchLogits)
	nextToken = generate_message_from_feature_ids(nextFeatureIds)
	if nextToken != STOP_TOKEN:
		print nextToken
		genNotes.append(nextToken)

genMsgs = generate_msgarray_from_noteseq(genNotes)

for msg in genMsgs:
	print(msg)

save_output(sys.argv[2], genMsgs)
