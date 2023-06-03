import tensorflow as tf

data_dir = 'data/raw/pinterest'
report_path = 'output/report-5.txt'
model_path = 'models/my_model'


def evaluate():
    test_df = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(180, 180))
    model = tf.keras.models.load_model(model_path)
    metrics = [
        tf.metrics.BinaryAccuracy(),
        tf.metrics.Precision(),
        tf.metrics.Recall()
    ]
    model.compile(metrics=metrics)
    result = model.evaluate(test_df)
    _, acc, precision, recall = result

    result_lines = [
        f'Test data len: {len(test_df)}\n'
        f'Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}'
    ]
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(result_lines)


if __name__ == '__main__':
    evaluate()
