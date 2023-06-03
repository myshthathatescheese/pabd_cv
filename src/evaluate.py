import os

import tensorflow as tf

model = tf.keras.models.load_model('models/my_model')


def resize(img, label):
    img_r = tf.image.resize(img, (180, 180))
    return img_r, label


if __name__ == '__main__':
    data_dir = 'data/raw/pinterest'
    test_dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
    test_dataset = test_dataset.map(resize)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision(),
    ]
    model.compile(metrics=metrics)
    history = model.evaluate(test_dataset)
    loss, accuracy, precision, recall = history
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

