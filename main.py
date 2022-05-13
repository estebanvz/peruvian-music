# %%
import os
import pickle
from tabnanny import verbose
from warnings import filters

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, Sequential, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    GRU,
    LSTM,
    Activation,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GaussianNoise,
    GlobalAveragePooling1D,
    Lambda,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    Multiply,
    Permute,
    RepeatVector,
    Reshape,
    Softmax,
)
from keras.utils.vis_utils import plot_model
from music21 import chord, converter, environment, instrument, key, note, stream, tempo
from RNNAttention import (
    create_lookups,
    create_network,
    get_distinct,
    get_music_list,
    prepare_sequences,
    sample_with_temp,
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

us = environment.UserSettings()
us["midiPath"] = "/usr/bin/timidity"
us["musicxmlPath"] = "/usr/bin/mscore"
us["musescoreDirectPNGPath"] = "/usr/bin/mscore"
# %%
TEMPO = None


def extract_notes(composition):
    notes = []
    durations = []
    for element in composition.flat:
        if isinstance(element, tempo.MetronomeMark):
            TEMPO = element
        if isinstance(element, note.Note):
            notes.append(str(element.nameWithOctave))
            durations.append(str(element.duration.quarterLength))
        if isinstance(element, note.Rest):
            notes.append(str(element.name))
            durations.append(str(element.duration.quarterLength))
        if isinstance(element, chord.Chord):
            notes.append(".".join(n.nameWithOctave for n in element.pitches))
            durations.append(str(element.duration.quarterLength))
    return np.array(notes), np.array(durations), TEMPO


original_score = converter.parse("music.mid")
c_original_score = original_score.chordify()
# for c in c_original_score.recurse().getElementsByClass('Chord'):
#     c.closedPosition(forceOctave=4, inPlace=True)
notes = []
duration = []
i_notes = []
i_durations = []
instruments = []
for part in [c_original_score]:
    notes, durations, TEMPO = extract_notes(part)
    i_notes.append(notes)
    i_durations.append(durations)
n_instruments = len(i_notes)

# %%


def get_main_instrument(instruments_list):
    tmp = []
    for e in instruments_list:
        a = np.unique(e)
        tmp.append(len(a))
    return np.argmax(tmp)


def _notes_to_dummy(instrument_list, array=False):
    uniques = np.unique(instrument_list)
    d_notes = {}
    for i, e in enumerate(uniques):
        if array:
            tmp = np.zeros(len(uniques))
            tmp[i] = 1
            d_notes[e] = tmp
        else:
            d_notes[e] = i
    result = []
    for instrument in instrument_list:
        result.append(d_notes[instrument])
    return np.array(result), d_notes


def notes_to_dummy(instruments_list):
    result = []
    dics = []
    for e in instruments_list:
        instrument, dic = _notes_to_dummy(e)
        result.append(instrument)
        dics.append(dic)
    return np.array(result, dtype=object), dics

    # for instrument in instruments_list:


def _lag_music_notes(music_list, n_lags):
    music_list_aux = music_list.copy()
    result = [
        music_list_aux,
    ]
    for i in range(1, n_lags + 1):
        music_list_aux = music_list_aux[:-1].copy()
        result.append(music_list_aux)
    for i in range(n_lags):
        result[i] = result[i][n_lags - i :]
    tmp = np.flip(np.array(result), axis=0)
    return tmp


def lag_music_notes(musics_lists, n_lags):
    result = []
    for e in musics_lists:
        result.append(_lag_music_notes(e, n_lags))
    return result


index_main = get_main_instrument(i_notes)

id_notes, dics_notes = _notes_to_dummy(i_notes[index_main])
id_durations, dics_durations = _notes_to_dummy(i_durations[index_main])
lag_notes = 8
idl_notes = _lag_music_notes(id_notes, lag_notes)
idl_durations = _lag_music_notes(id_durations, lag_notes)
# %%


def extract_label(idl_array, dictionary):
    idli_notes = idl_array[:-1]
    aux = idl_array[-1:]
    aux = np.array(aux).T
    idlo_notes = []
    for e in aux:
        tmp = np.zeros(len(dictionary))
        tmp[int(e[0])] = 1
        idlo_notes.append(tmp)
    return np.array(idli_notes).T, np.array(idlo_notes)


idli_notes, idlo_notes = extract_label(idl_notes, dics_notes)
idli_durations, idlo_durations = extract_label(idl_durations, dics_durations)

input_network = [idli_notes, idli_durations]
output_network = [idlo_notes, idlo_durations]
# %%


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        return config


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        return config


def create_network(
    n_notes,
    n_durations,
    o_notes,
    o_durations,
    embed_size=100,
    rnn_units=256,
    lag_notes=3,
):
    """create the structure of the neural network"""
    num_heads = 2
    ff_dim = 32
    i_notes = Input(shape=(n_notes,))
    i_durations = Input(shape=(n_durations,))
    embedding_layer = TokenAndPositionEmbedding(
        n_notes, n_notes * lag_notes, embed_size
    )
    x = embedding_layer(i_notes)
    transformer_block = TransformerBlock(embed_size, num_heads, ff_dim)
    x = transformer_block(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.1)(x)
    embedding_layer = TokenAndPositionEmbedding(
        n_durations, n_durations * lag_notes, embed_size
    )
    y = embedding_layer(i_durations)
    transformer_block = TransformerBlock(embed_size, num_heads, ff_dim)
    y = transformer_block(y)
    y = GlobalAveragePooling1D()(y)
    y = Dropout(0.1)(y)
    c = Concatenate()([x, y])
    c = Dense(20, activation="relu")(c)
    c = Dropout(0.1)(c)
    n = Dense(o_notes, activation="softmax", name="d_notes")(c)
    d = Dense(
        o_durations, activation="softmax", name="d_duration", kernel_regularizer="l1"
    )(c)
    notes_out = n
    durations_out = d

    model = Model([i_notes, i_durations], [notes_out, durations_out])
    opti = tf.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        optimizer=opti,
        metrics="accuracy",
    )

    return model


compositor = create_network(
    input_network[0].shape[1],
    input_network[1].shape[1],
    output_network[0].shape[1],
    output_network[1].shape[1],
    lag_notes=lag_notes,
)

# %%

nombre_modelo = "peruvian_music"
logdir = "./logs/" + nombre_modelo
tensorboard_callback = callbacks.TensorBoard(logdir, histogram_freq=1)
logs = compositor.fit(
    input_network,
    output_network,
    epochs=500,
    batch_size=32,
    shuffle=True,
    callbacks=[tensorboard_callback],
    verbose=2,
)
# %%
# %load_ext tensorboard
# %tensorboard --logdir ./logs/
# %%


def composition(compositor, initial_notes, initial_durations, n_notes=500):
    tolerance = 2
    index = np.random.randint(len(input_network[0]))
    initial_note = initial_notes[index : index + 1]
    initial_duration = initial_durations[index : index + 1]
    f_notes = initial_note.copy()
    f_durations = initial_duration.copy()
    for _ in range(n_notes):
        tmp1 = f_notes[:, -tolerance:]
        tmp1 = np.sum(tmp1)
        tmp2 = f_notes[:, -tolerance * 2 : -tolerance]
        tmp2 = np.sum(tmp2)
        if np.random.choice(range(100)) > 70 or int(tmp1) == int(tmp2):
            # if(np.random.choice(range(10))>9):
            index = np.random.randint(len(input_network[0]))
            initial_note = initial_notes[index : index + 1]
            initial_duration = initial_durations[index : index + 1]
        p_notes, p_durations = compositor.predict(
            [initial_note, initial_duration], verbose=0
        )
        for i, note in enumerate(p_notes):
            p_notes = np.argmax(note).reshape((1, 1))
        for i, duration in enumerate(p_durations):
            p_durations = np.argmax(duration).reshape((1, 1))
        f_notes = np.hstack([f_notes, p_notes])
        f_durations = np.hstack([f_durations, p_durations])
        initial_note = np.hstack([initial_note[:, 1:], p_notes])
        initial_duration = np.hstack([initial_duration[:, 1:], p_durations])
    return f_notes, f_durations


f_notes, f_durations = composition(compositor, input_network[0], input_network[1])
print(pd.value_counts(f_notes.flatten()))
print(pd.value_counts(input_network[0].flatten()))

# %%


def transform_music_text(f_notes: np, f_durations, d_notes, d_durations):
    d_notes = dict(map(reversed, d_notes.items()))
    d_durations = dict(map(reversed, d_durations.items()))
    f_notes = f_notes.flatten()
    f_durations = f_durations.flatten()
    r_notes = []
    r_durations = []
    for note in f_notes:
        r_notes.append(d_notes[note])
    for duration in f_durations:
        r_durations.append(d_durations[duration])
    return r_notes, r_durations


r_notes, r_durations = transform_music_text(
    f_notes, f_durations, dics_notes, dics_durations
)
# %%
final = []
for a in range(1):
    s = stream.Part()
    s.append(TEMPO)
    s.append(instrument.Guitar())
    for i in range(a, len(r_notes), n_instruments):
        d = eval(str(r_durations[i]))
        if r_notes[i] == "rest":
            s.append(note.Rest(r_notes[i], quarterLength=float(d)))
        elif "." in r_notes[i]:
            s.append(chord.Chord(r_notes[i].split("."), quarterLength=float(d)))
        else:
            s.append(note.Note(r_notes[i], quarterLength=float(d)))
    final.append(s)

song = stream.Score()
for e in final:
    song.insert(0, e)

song.write("midi", "alma.mid")
# %%
compositor.save("model.h5")
# %%
plot_model(
    compositor, to_file="model_plot.png", show_shapes=True, show_layer_names=True
)
