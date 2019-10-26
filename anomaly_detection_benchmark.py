import pandas as pd
from loglizer.models.PCA import PCA
from loglizer.models.LogClustering import LogClustering
from loglizer.models.InvariantsMiner import InvariantsMiner
from loglizer import dataloader, preprocessing
import timeit

models = ['PCA', 'InvariantsMiner', 'LogClustering']  # anomaly detection models
struct_log = 'log_parsed/HDFS.log_structured.csv'  # dataset
labels = 'anomaly_labels.csv'  # labels used for testing

if __name__ == '__main__':
    # load the dataset
    (x_tr, y_train), (x_te, y_test) = dataloader.load_HDFS(struct_log,
                                                           label_file=labels,
                                                           window='session',
                                                           train_ratio=0.7,
                                                           split_type='uniform')
    benchmark_results = []
    feature_extractor = preprocessing.FeatureExtractor()

    # train each model
    for _model in models:
        print('Evaluating {} on HDFS:'.format(_model))

        if _model == 'PCA':
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf',
                                                      normalization='zero-mean')
            model = PCA()
            model.fit(x_train)

        elif _model == 'InvariantsMiner':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr)
            model = InvariantsMiner(epsilon=0.5)
            model.fit(x_train)

        elif _model == 'LogClustering':
            feature_extractor = preprocessing.FeatureExtractor()
            x_train = feature_extractor.fit_transform(x_tr, term_weighting='tf-idf')
            model = LogClustering(max_dist=0.3, anomaly_threshold=0.3)
            model.fit(x_train[y_train == 0, :])  # Use only normal samples for training"""

        # extract feature matrix
        x_test = feature_extractor.transform(x_te)

        print('Train accuracy:')
        start = timeit.default_timer()
        precision, recall, f1 = model.evaluate(x_train, y_train)
        stop = timeit.default_timer()
        runtime = stop - start
        benchmark_results.append([_model + '-train', precision, recall, f1, runtime])

        print('Test accuracy:')
        start = timeit.default_timer()
        precision, recall, f1 = model.evaluate(x_test, y_test)
        stop = timeit.default_timer()
        runtime = stop - start
        benchmark_results.append([_model + '-test', precision, recall, f1, runtime])

    # store results as csv file
    pd.DataFrame(benchmark_results, columns=['Model', 'Precision', 'Recall', 'F1', 'Runtime']) \
        .to_csv('anomaly_detection_results/benchmark_results.csv', index=False)
