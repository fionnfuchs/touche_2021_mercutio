from pipeline import Pipe
import tensorflow as tf
import numpy as np
from log import get_child_logger
import pickle
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string

logger = get_child_logger(__file__)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


class ClassifierRanking(Pipe):

    model_path = "material/argument_classifier/model_conv1d"

    def __init__(self, config):
        logger.info("Loading model...")
        model = tf.keras.models.load_model(self.model_path)

        from_disk = CustomUnpickler(
            open("material/argument_classifier/text_vectorization_conv1d", "rb")
        ).load()
        vectorize_layer = TextVectorization.from_config(from_disk["config"])
        # You have to call `adapt` with some dummy data (BUG in Keras)
        vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        vectorize_layer.set_weights(from_disk["weights"])

        # A string input
        inputs = tf.keras.Input(shape=(1,), dtype="string")
        # Turn strings into vocab indices
        indices = vectorize_layer(inputs)
        # Turn vocab indices into predictions
        outputs = model(indices)

        # Our end to end model
        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        logger.info("Model loaded.")

    def process(self, topic):
        for po in topic.processing_objects:
            for uuid in po.documents:
                try:
                    d = po.documents[uuid]
                    text = d.chat_noir_result.text
                    value = self.model.predict([text])
                    po.documents[uuid].scores["classifier"] = value
                except:
                    logger.info("Could not predict value for document text. Skipping.")
        return topic


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "custom_standardization":
            return custom_standardization
        return super().find_class(module, name)
