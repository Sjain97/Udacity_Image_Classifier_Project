import self_utils
from PIL import Image
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json
import numpy as np

import warnings
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py Settings")

    parser.add_argument('image_path', help='image path')

    parser.add_argument('model_name', help='model name in same directory')

    parser.add_argument('--top_k',
                        default = 1,
                        type=int,
                        help='Choose top K matches as int.')

    parser.add_argument('--category_names',
                        default = 'label_map.json',
                        type=str,
                        help='Path to a JSON file mapping labels to flower names.')

    args = parser.parse_args()

    return args

def predict(processed_image, model, top_k):
  predictions = model.predict(processed_image)
  top_k_values, top_k_indices = tf.nn.top_k(predictions, top_k)
  top_k_indices = top_k_indices[0] + 1
  return top_k_indices.numpy().astype(str), top_k_values.numpy()[0]

def main():

    # Keyword Args for Prediction
    args = arg_parser()

    saved_keras_model_filepath = './{}'.format(args.model_name)
    model = tf.keras.models.load_model(saved_keras_model_filepath, custom_objects={'KerasLayer': hub.KerasLayer})

    # Process Image
    image = np.asarray(Image.open(args.image_path))
    processed_image = np.expand_dims(self_utils.process_image(image), axis=0)

    # Use `processed_image` to predict the top K most likely classes
    top_flowers, top_probs = predict(processed_image, model, args.top_k)

    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    labels = [class_names[str(top_flowers[i])] for i in range(args.top_k)]

    # Print out probabilities
    print('Top flower names: \n', labels)
    print('\nCorresponding top probabilities:\n', top_probs)


#Run Program
if __name__ == '__main__': main()
