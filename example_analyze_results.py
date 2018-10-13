from mlaut.analyze_results import AnalyseResults
from mlaut.data import Data
from mlaut.analyze_results.scores import ScoreAccuracy
import pickle
from mlaut.estimators.estimators import instantiate_default_estimators




from mlaut.estimators.nn_estimators import Deep_NN_Classifier
hyperparameters = {'epochs': [50,100], 
                    'batch_size': [0, 50, 100]}
def keras_model1(num_classes, input_dim):
    model = OverwrittenSequentialClassifier()
    model.add(Dense(288, input_dim=input_dim, activation='relu'))
    model.add(Dense(144, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])

    return model

deep_nn_4_layer_thin_dropout = Deep_NN_Classifier(keras_model=keras_model1, 
                            properties={'name':'NN-4-layer_thin_dropout'})


def keras_model2(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_no_dropout = Deep_NN_Classifier(hyperparameters=hyperparameters,
                            keras_model=keras_model2,
                            properties={'name':'NN-4-layer_wide_no_dropout'})


def keras_model3(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(2500, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_4_layer_wide_with_dropout = Deep_NN_Classifier(hyperparameters=hyperparameters,
                            keras_model=keras_model3,
                            properties={'name':'NN-4-layer_wide_with_dropout'})


def keras_model4(num_classes, input_dim):
    nn_deep_model = OverwrittenSequentialClassifier()
    nn_deep_model.add(Dense(5000, input_dim=input_dim, activation='relu'))
    nn_deep_model.add(Dense(4500, activation='relu'))
    nn_deep_model.add(Dense(4000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(3500, activation='relu'))
    nn_deep_model.add(Dense(3000, activation='relu'))
    nn_deep_model.add(Dense(2500, activation='relu'))
    nn_deep_model.add(Dropout(0.5))


    nn_deep_model.add(Dense(2000, activation='relu'))
    nn_deep_model.add(Dense(1500, activation='relu'))
    nn_deep_model.add(Dense(1000, activation='relu'))
    nn_deep_model.add(Dropout(0.5))

    nn_deep_model.add(Dense(500, activation='relu'))
    nn_deep_model.add(Dense(250, activation='relu'))
    nn_deep_model.add(Dense(num_classes, activation='softmax'))

    model_optimizer = optimizers.Adam(lr=0.001)
    nn_deep_model.compile(loss='mean_squared_error', optimizer=model_optimizer, metrics=['accuracy'])
    return nn_deep_model

deep_nn_12_layer_wide_with_dropout = Deep_NN_Classifier(hyperparameters=hyperparameters,
                            keras_model=keras_model4,
                            properties={'name':'NN-12-layer_wide_with_dropout'})


estimators = [deep_nn_4_layer_thin_dropout,
            deep_nn_4_layer_wide_no_dropout, 
            deep_nn_4_layer_wide_with_dropout,
            deep_nn_12_layer_wide_with_dropout]

estim = instantiate_default_estimators(['Classification'])
for e in estim:
    if e.properties['name'] is not 'NeuralNetworkDeepClassifier':
        estimators.append(e)






data = Data()
input_io = data.open_hdf5('data/delgado.h5', mode='r')
out_io = data.open_hdf5('data/delgado-classification-deep.h5', mode='a')
analyze = AnalyseResults(hdf5_output_io=out_io, 
                        hdf5_input_io=input_io, 
                        input_h5_original_datasets_group='openml/', 
                        output_h5_predictions_group='experiments/predictions/')
                    
score_accuracy = ScoreAccuracy()
(errors_per_estimator, 
 errors_per_dataset_per_estimator, 
 errors_per_dataset_per_estimator_df) = analyze.prediction_errors(metric=score_accuracy, estimators=estimators)

training_time = analyze.average_training_time(estimators)
print(training_time)
# print(f'Errors per estimator: {errors_per_estimator}')
# print(f'Errors per dataset and per estimator: {errors_per_dataset_per_estimator}')

# t_test, t_test_df = analyze.t_test(observations)
# print('******t-test******')
# print(t_test_df)
# sign_test, sign_test_df = analyze.sign_test(observations)
# print('******sign test******')
# print(sign_test_df)

# t_test_bonferroni, t_test_bonferroni_df = analyze.t_test_with_bonferroni_correction(observations)
# print('******t-test bonferroni correction******')
# print(t_test_bonferroni_df)

# wilcoxon_test, wilcoxon_test_df = analyze.wilcoxon_test(observations)
# print('******Wilcoxon test******')
# print(wilcoxon_test_df)

# friedman_test, friedman_test_df = analyze.friedman_test(observations)
# print('******Friedman test******')
# print(friedman_test_df)

# nemeniy_test = analyze.nemenyi(observations)
# print('******Nemeniy test******')
# print(nemeniy_test)