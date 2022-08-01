#from https://medium.com/deep-learning-with-keras/build-an-efficient-tensorflow-input-pipeline-for-word-level-text-generation-2d224e02ae15
import tensorflow as tf
import numpy
import re
import string
from tensorflow.keras import layers
raw_data_ds = tf.data.TextLineDataset(["nietzsche.txt"])

for elems in raw_data_ds.take(10):
    print(elems.numpy()) #.decode("utf-8")

raw_data_ds = raw_data_ds.map(lambda x: tf.strings.split(x))
for elems in raw_data_ds.take(5):
    print(elems.numpy())

raw_data_ds=raw_data_ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
for elems in raw_data_ds.take(5):
    print(elems.numpy())

input_sequence_size = 4

sequence_data_ds=raw_data_ds.window(input_sequence_size+1, drop_remainder=True)
for window in sequence_data_ds.take(3):
  print(list(window.as_numpy_iterator()))

sequence_data_ds = sequence_data_ds.flat_map(lambda window: window.batch(5))
for elem in sequence_data_ds.take(3):
  print(elem)

sequence_data_ds = sequence_data_ds.map(lambda window: (window[:-1], window[-1:]))
X_train_ds_raw = sequence_data_ds.map(lambda X,y: X)
y_train_ds_raw = sequence_data_ds.map(lambda X,y: y)

print("Input X  (sequence) \t\t    ----->\t Output y (next word)")
for elem1, elem2 in zip(X_train_ds_raw.take(3),y_train_ds_raw.take(3)):
   print(elem1.numpy(),"\t\t----->\t", elem2.numpy())

def convert_string(X: tf.Tensor):
  str1 = ""
  for ele in X:
    str1 += ele.numpy().decode("utf-8")+" "
  str1= tf.convert_to_tensor(str1[:-1])
  return str1

X_train_ds_raw=X_train_ds_raw.map(lambda x: tf.py_function(func=convert_string,
          inp=[x], Tout=tf.string))

print("Input X  (sequence) \t\t    ----->\t Output y (next word)")
for elem1, elem2 in zip(X_train_ds_raw.take(5),y_train_ds_raw.take(5)):
   print(elem1.numpy(),"\t\t----->\t", elem2.numpy())

print(X_train_ds_raw.element_spec, y_train_ds_raw.element_spec)

X_train_ds_raw=X_train_ds_raw.map(lambda x: tf.reshape(x,[1]))

X_train_ds_raw.element_spec, y_train_ds_raw.element_spec

#Preprocessing

def custom_standardization(input_data):
    lowercase     = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    stripped_num  = tf.strings.regex_replace(stripped_html, "[\d-]", " ")
    stripped_punc  =tf.strings.regex_replace(stripped_num,
                             "[%s]" % re.escape(string.punctuation), "")
    return stripped_punc

max_features = 54762    # Number of distinct words in the vocabulary
sequence_length = input_sequence_size    # Input sequence size
batch_size = 128                # Batch size

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    # split --> DEFAULT: split each sample into substrings (usually words)
    output_mode="int",
    output_sequence_length=sequence_length,
)

vectorize_layer.adapt(raw_data_ds.batch(batch_size))

print("The size of the vocabulary (number of distinct words): ", len(vectorize_layer.get_vocabulary()))

print("The first 10 entries: ", vectorize_layer.get_vocabulary()[:10])

vectorize_layer.get_vocabulary()[3]

def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return tf.squeeze(vectorize_layer(text))

for elem in X_train_ds_raw.take(3):
  print("X: ",elem.numpy())

# Vectorize the data.
X_train_ds = X_train_ds_raw.map(vectorize_text)
y_train_ds = y_train_ds_raw.map(vectorize_text)

for elem in X_train_ds.take(3):
  print("X: ",elem.numpy())

X_train_ds_raw.element_spec, y_train_ds_raw.element_spec

for elem in y_train_ds.take(2):
  print("shape: ", elem.shape, "\n next_char: ",elem.numpy())

y_train_ds=y_train_ds.map(lambda x: x[:1])

for elem in y_train_ds.take(2):
  print("shape: ", elem.shape, "\n next_char: ",elem.numpy())

for (X,y) in zip(X_train_ds.take(5), y_train_ds.take(5)):
  print(X.numpy(),"-->",y.numpy())

#Finalize dataset

X_train_ds_raw.element_spec, y_train_ds_raw.element_spec

train_ds =  tf.data.Dataset.zip((X_train_ds,y_train_ds))
train_ds.element_spec

def _fixup_shape(X, y):
    X.set_shape([4])
    y.set_shape([1])
    return X, y

train_ds=train_ds.map(_fixup_shape)
train_ds.element_spec

for el in train_ds.take(5):
  print(el)

#dataset optimizations:

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size, drop_remainder=True).cache().prefetch(buffer_size=AUTOTUNE)

train_ds.element_spec