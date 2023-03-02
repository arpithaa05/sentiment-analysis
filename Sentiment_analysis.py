import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


df = pd.read_csv(r"amazon_baby.csv")
df["sentiments"] = df.rating.apply(lambda x: 0 if x in [1,2] else 1)


#Data preprocessing
tokenizer = Tokenizer(oov_token="<OOV>")

#80% Training data and 20% Testing
split = round(len(df)*0.8)

train_revs = df["review"][:split]
train_label = df["sentiments"][:split]
test_revs = df["review"][split:]
test_label = df["sentiments"][split:]

#Converting everything to string
train_sent = []
train_labels = []
test_sent = []
test_labels = []

for row in train_revs:
    train_sent.append(str(row))
for row in train_label:
    train_labels.append(row)
for row in test_revs:
    test_sent.append(str(row))
for row in test_label:
    test_labels.append(row)

vocab_size = 40000
embedding_dim = 16
max_len = 120
trunc_type = "post"
oov_tok = "<OOV>"
padding_type = "post"

#Tokenizing
tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(train_sent)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(train_sent)
padded = pad_sequences(sequences, maxlen=max_len,truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sent)
test_padded = pad_sequences(test_sequences, maxlen=max_len)

#Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
print(df.dtypes)

training_labels_final = np.array(train_labels)
testing_labels_final = np.array(test_labels)

num_epochs = 20
history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(test_padded,testing_labels_final))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs=range(len(acc))
plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Loss')
plt.title('Training and validation loss')
plt.figure()