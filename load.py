#Script used to load models of the rnn in test.py, generates a handful of test outputs.
import tensorflow as tf

import numpy as np
import os
import time
from tensorflow.keras import backend

one_step_reloaded = tf.saved_model.load('test_with_all_mails_drive_20_model_v2')

def get_output(input):
    states = None
    next_char = tf.constant([input])
    result = [next_char]

    for n in range(200):
      next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
      result.append(next_char)

    print(tf.strings.join(result)[0].numpy().decode("utf-8"))
    print('--------------------------------END OF MESSAGE--------------------------------')

get_output('Dear ')
get_output('Dear Jose')
get_output('Hello ')
get_output('Greetings ')
get_output('Warning ')