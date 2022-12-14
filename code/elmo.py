"""Experiment with ELMo Model"""

import tensorflow as tf 
import tensorflow_hub as hub 

elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

def elmo_vectors(x):
    """Creates ELMo embeddings."""
    embeddings = elmo(inputs={
        "tokens": tf.squeeze(tf.cast(x, tf.string)),
        "sequence_len": tf.constant(batch_size*[max_len])
    },
    signature="tokens",
    as_dict=True)["elmo"]
    
    return embeddings

# define text tensor
text_input = tf.Variable(["the cat is on the mat", "dogs are in the fog"], dtype=tf.string) 

embed = elmo_vectors(text_input)

print(embed)