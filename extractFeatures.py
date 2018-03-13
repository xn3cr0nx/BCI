import numpy as np
from math import floor, ceil
# from pandas import *
from scipy.linalg import fractional_matrix_power
from functools import reduce
from operator import itemgetter
from scipy.stats import hmean

############################################################################################################################################
# Data Importing
############################################################################################################################################

signal = np.array(np.genfromtxt(
    '/home/xn3cr0nx/Desktop/Datasets/csv/CompetitionIV/IV_II/A01T/original_signal.csv', delimiter=','))
assert(signal.shape == (672528, 25)), "wrong data shape from csv"
features = np.delete(signal, np.s_[22:], axis=1)
assert(features.shape == (672528, 22)), "wrong data shape after extraction"

POS = np.array(np.genfromtxt(
    '/home/xn3cr0nx/Desktop/Datasets/csv/CompetitionIV/IV_II/A01T/info/POS.csv', delimiter=','), dtype=np.float64)
TYP = np.array(np.genfromtxt(
    '/home/xn3cr0nx/Desktop/Datasets/csv/CompetitionIV/IV_II/A01T/info/TYP.csv', delimiter=','))
assert(TYP.shape[0] == 603), "wrong events position shape"
assert(POS.shape[0] == 603), "wrong events type shape"

# imports labels
labels = np.array(np.genfromtxt(
    '/home/xn3cr0nx/Desktop/Datasets/csv/CompetitionIV/IV_II/A01T/info/Classlabel.csv', delimiter=','))
assert(labels.shape[0] == 288), "wrong labels shape"

############################################################################################################################################



############################################################################################################################################
# Preprocessing
############################################################################################################################################

def generate_positions(TYP, POS):
  # filtering the position corresponding to a valid type
  classes = [769, 770, 771, 772]
  # for i in TYP:
  #   if(i in classes):
  #     position.append(POS[np.where(TYP==i)[0][0]])
  positions = []
  for i in range(TYP.shape[0]):
    if(TYP[i] in classes):
      if(i < TYP.shape[0] - 1):
        positions.append((POS[i], POS[i + 1] - 1))
      else:
        positions.append((POS[i], features.shape[0]))
  positions = np.array(positions)
  assert(positions.shape[0] == 288), "wrong positions shape"
  return positions

# Generate trials and test their properties
def generate_trials(positions):
  # dividing runs in 288 trials, 48 per run
  trials = []
  for i in positions.astype(np.int64):
    trials.append(features[i[0]:i[1]])
  trials = np.array(trials)
  assert(positions.shape[0] == 288), "wrong trials shape"
  for i in trials:
    i[np.isnan(i)] = 0
    # i = np.nan_to_num(i)
  for i in trials:
    assert(not(np.any(np.isnan(i))))
    assert(not(np.any(np.isinf(i))))
  return trials

# Generates the sample covariances matrices and testing their properties
  def generate_SCMs(trials):
    SCMs = []
    for trial in trials:
      SCM = trial.transpose().dot(trial) / (trial.shape[0] - 1)
      SCMs.append(SCM)
    SCMs = np.array(SCMs)
    assert(SCMs[0].shape[0] == SCMs[0].shape[1]
           ), "the sample covariance matrix isn't squared"
    assert((SCMs[0].transpose() == SCMs[0]).all()), "the matrix isn't symmetric"
    for SCM in SCMs:
      assert(np.all(np.linalg.eigvals(SCM) > 0)
             ), "the matrix isn't postive definite"
    # print(SCMs[229][:, 3], "you have to interpolate nan, in this way is possible to calculate the mean")
    for i in SCMs:
      assert(not(np.any(np.isnan(i))))
      assert(not(np.any(np.isinf(i))))
    return SCMs

  # generate_SCMs(generate_trials(generate_positions(TYP, POS)))




############################################################################################################################################
# Differential Geometry Utils
############################################################################################################################################

def calculate_log(P):
  return np.linalg.multi_dot([np.linalg.eig(P)[1], np.diag(np.log(np.linalg.eig(P)[0])), np.transpose(np.linalg.eig(P)[1])])


def calculate_exp(P):
  eigenvectors = np.linalg.eig(P)[1]
  # assert(eigenvectors.shape == (22, 22))
  eigenvectors_transp = np.transpose(np.linalg.eig(P)[1])
  # assert(eigenvectors_transp.shape == (22, 22))
  exp_eigenvalues = np.diag(np.exp(np.linalg.eig(P)[0]))
  # assert(exp_eigenvalues.shape == (22, 22))
  # return np.linalg.multi_dot([np.linalg.eig(P)[1], np.diag(np.exp(np.linalg.eig(P)[0])), np.transpose(np.linalg.eig(P)[1])])
  return np.linalg.multi_dot([eigenvectors, exp_eigenvalues, eigenvectors_transp])


def calculate_Log(P, Q):
  return np.linalg.multi_dot([fractional_matrix_power(P, 0.5), calculate_log(np.linalg.multi_dot([fractional_matrix_power(P, -0.5), Q, fractional_matrix_power(P, -0.5)])), fractional_matrix_power(P, 0.5)])


def calculate_Exp(P, S):
  return np.linalg.multi_dot([fractional_matrix_power(P, 0.5), calculate_exp(np.linalg.multi_dot([fractional_matrix_power(P, -0.5), S, fractional_matrix_power(P, -0.5)])), fractional_matrix_power(P, 0.5)])


def calculate_retraction_map(P, S):
  return calculate_exp(calculate_log(P) + S)

def calculate_inverse_retraction_map(P, S):
  return calculate_log(S) - calculate_log(P)


######################
# Distances
######################

# Euclidean distance
def euclidean_distance(A, B):
  return np.linalg.norm(A - B)

# Riemannian distance between two SPD matrices P and Q
def riemmanian_distance(A, B):
  # norm = calculate_Log(A, B)
  # return np.trace(np.linalg.multi_dot([norm, np.linalg.inv(A), norm, np.linalg.inv(A)]))**0.5
  return np.linalg.norm(calculate_log(np.linalg.multi_dot([fractional_matrix_power(A, 0.5), B, fractional_matrix_power(A, 0.5)])))

# Log Euclidean distance
def log_euclidean_distance(A, B):
  return np.linalg.norm(calculate_log(A) - calculate_log(B))  

# Harmonic distance
def harmonic_distance(A, B):
  return np.linalg.norm(np.linalg.inv(A) - np.linalg.inv(B))


#####################
# Means
#####################

# Arithmetic Mean
def arithmetic_mean(P):
  return reduce(lambda x, y: x + y, P) / (len(P))

# Riemmanian Geometric Mean
def riemmanian_geometric_mean(P):
  S, K = [], np.identity(P[0].shape[0])
  # for i in range(len(P)):
  for i in range(5):
    S = reduce(lambda x, y: x + y,
               map(lambda x: calculate_Log(K, x), P)) / (len(P))
    K = calculate_Exp(K, S)
  return S

# Log Euclidean Mean
def log_euclidean_mean(P):
  return np.exp(reduce(lambda x, y: x + y, np.log(P)) / (P[0].shape[0]))

# Harmonic Mean
def harmonic_mean(P):
  return np.linalg.inv(reduce(lambda x, y: x + y, map(lambda x: np.linalg.inv(x), P)) / (P[0].shape[0]))


#####################
# Medians
#####################

# Euclidean Geometric Median
# Weiszfeld algorithm
def euclidean_geometric_median(P):
  S, K = [], np.identity(P[0].shape[0])
  for i in range(len(P)):
    # S = np.linalg.multi_dot([reduce(lambda x, y: x + y, map(lambda x: x/np.linalg.norm(K - x), P)), np.linalg.inv(reduce(lambda x, y: x + y, map(lambda x: 1/np.linalg.norm(K - x), P)))])
    # S = reduce(lambda x, y: x + y, map(lambda x: x/np.linalg.norm(K - x), P)) * np.linalg.inv(reduce(lambda x, y: x + y, map(lambda x: 1/np.linalg.norm(K - x), P)))
    S = reduce(lambda x, y: x + y, map(lambda x: x/np.linalg.norm(K - x), P)) * reduce(lambda x, y: x + y, map(lambda x: 1/np.linalg.norm(K - x), P))**(-1)
    K = S
  return S

# Riemmanian Geometric Median
def riemmanian_geometric_median(P):
  V, K = [], np.identity(P[0].shape[0])
  for i in range(3):
    V = reduce(lambda x, y: x + y, map(lambda x: calculate_Log(K, x)/riemmanian_distance(K, x), P)) * reduce(lambda x, y: x + y, map(lambda x: 1/riemmanian_distance(K, x), P))**(-1)
    K = calculate_Exp(K, V)
  return V

# Log-Euclidean Median
def log_euclidean_median(P):
  V, K = [], np.identity(P[0].shape[0])
  for i in range(3):
    V = reduce(lambda x, y: x + y, map(lambda x: calculate_inverse_retraction_map(K, x)/riemmanian_distance(K, x), P)) * reduce(lambda x, y: x + y, map(lambda x: 1/riemmanian_distance(K, x), P))**(-1)
    K = calculate_retraction_map(K, V)
  return V


#####################
# Trimmed
#####################

# Trimmed Mean
# key 'riemmanian_mean' for compute Trimmed Riemmanian Geometric Mean
# key 'log_euclidean_mean' for compute Trimmed Log Euclidean Mean
def trimmed_mean(P, a, key):
  mean = riemmanian_geometric_mean(P) if key=='riemmanian_mean' else log_euclidean_mean(P) if key=='log_euclidean_mean' else None
  ordered = sorted([(riemmanian_distance(mean, x), x) if key=='riemmanian_mean' else (log_euclidean_distance(mean, x), x) if key=='log_euclidean_mean' else None for i,x in enumerate(P)], key=itemgetter(0)) # returns the trimmed array containing tuple with distance and matrix
  trimmed = ordered[floor(len(P)*a):ceil(len(P)-len(P)*a)]
  return riemmanian_geometric_mean([x[1] for x in trimmed]) if key=='riemmanian_mean' else log_euclidean_mean([x[1] for x in trimmed]) if key=='log_euclidean_mean' else None

# Trimmed Median
# key 'riemmanian_median' for compute Trimmed Riemmanian Geometric Median
# key 'log_euclidean_median' for compute Trimmed Log Euclidean Median
def trimmed_median(P, a, key):
  mean = riemmanian_geometric_median(P) if key=='riemmanian_median' else log_euclidean_median(P) if key=='log_euclidean_median' else None
  ordered = sorted([(riemmanian_distance(mean, x), x) if key=='riemmanian_median' else (log_euclidean_distance(mean, x), x) if key=='log_euclidean_median' else None for i,x in enumerate(P)], key=itemgetter(0)) # returns the trimmed array containing tuple with distance and matrix
  trimmed = ordered[floor(len(P)*a):ceil(len(P)-len(P)*a)]
  return riemmanian_geometric_median([x[1] for x in trimmed]) if key=='riemmanian_median' else log_euclidean_median([x[1] for x in trimmed]) if key=='log_euclidean_median' else None





############################################################################################################################################
# Predictions
############################################################################################################################################

# Divides SCMs and labels in train and test data
# def features_and_labels(SCMs, labels):
  x_train, x_test, y_train, y_test = SCMs[:floor(len(SCMs) * 0.7)], SCMs[ceil(
      (len(SCMs) * 0.7)):], labels[:floor(len(labels) * 0.7)], labels[ceil(len(labels) * 0.7):]
  # return (x_train, y_train), (x_test, y_test)


  # Divides the x_train data in 4 classes
class1, class2, class3, class4 = [], [], [], []
for i in range(len(y_train)):
  if(y_train[i] == 1.0):
    class1.append(x_train[i])
  elif(y_train[i] == 2.0):
    class2.append(x_train[i])
  elif(y_train[i] == 3.0):
    class3.append(x_train[i])
  elif(y_train[i] == 4.0):
    class4.append(x_train[i])

classes = [class1, class2, class3, class4]

def calculate_accuracy(classes, key):
  predictions = []
  if(key == 'c'):
    means = [np.mean(c) for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [euclidean_distance(x_test[i], means[j]) for j in range(len(means))]) + 1)
  elif(key == 'riemmanian_geometric_mean'):
    means = [riemmanian_geometric_mean(c) for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [riemmanian_distance(x_test[i], means[j]) for j in range(len(means))]) + 1)
  elif(key == 'log_euclidean_mean'):
    means = [log_euclidean_mean(c) for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [log_euclidean_distance(x_test[i], means[j]) for j in range(len(means))]) + 1)
  elif(key == 'harmonic_mean'):
    means = [hmean(c) for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [harmonic_distance(x_test[i], means[j]) for j in range(len(means))]) + 1)
  elif(key == 'euclidean_geometric_median'):
    means = [euclidean_geometric_median(c) for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [euclidean_distance(x_test[i], means[j]) for j in range(len(means))]) + 1)
  elif(key == 'riemmanian_geometric_median'):
    means = [riemmanian_geometric_median(c) for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [riemmanian_distance(np.linalg.inv(x_test[i]), np.linalg.inv(means[j])) for j in range(len(means))]) + 1)
  elif(key == 'log_euclidean_median'):
    means = [log_euclidean_median(c) for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [log_euclidean_distance(np.linalg.inv(x_test[i]), np.linalg.inv(means[j])) for j in range(len(means))]) + 1)
  elif(key == 'trimmed_riemmanian_mean'):
    means = [trimmed_mean(c, 0.1, 'riemmanian_mean') for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [riemmanian_distance(np.linalg.inv(x_test[i]), np.linalg.inv(means[j])) for j in range(len(means))]) + 1)
  elif(key == 'trimmed_riemmanian_median'):
    means = [trimmed_median(c, 0.1, 'riemmanian_median') for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [riemmanian_distance(np.linalg.inv(x_test[i]), np.linalg.inv(means[j])) for j in range(len(means))]) + 1)
  elif(key == 'log_euclidean_mean'):
    means = [trimmed_mean(c, 0.1, 'log_euclidean_mean') for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [log_euclidean_distance(np.linalg.inv(x_test[i]), np.linalg.inv(means[j])) for j in range(len(means))]) + 1)
  elif(key == 'log_euclidean_median'):
    means = [trimmed_median(c, 0.1, 'log_euclidean_median') for c in classes]
    for i in range(len(x_test)):
      predictions.append(np.argmin(
        [log_euclidean_distance(np.linalg.inv(x_test[i]), np.linalg.inv(means[j])) for j in range(len(means))]) + 1)
  else: None
  accuracy = np.sum(predictions == y_test) / len(y_test)
  return (key+' '+str(accuracy)+'%')


def print_accuracies():
  print("Arithmetic Mean Accuracy", calculate_accuracy(classes, 'arithmetic_mean'), "\n",
        "Riemmanian Geometric Mean Accuracy", calculate_accuracy(classes, 'riemmanian_geometric_mean'), "\n",
        "Log Euclidean Mean Accuracy", calculate_accuracy(classes, 'log_euclidean_mean'), "\n",
        "Harmonic Mean Accuracy", calculate_accuracy(classes, 'harmonic_mean'), "\n",
        "Euclidean Geometric Median Accuracy", calculate_accuracy(classes, 'euclidean_geometric_median'), "\n",
        "Riemmanian Geometric Median Accuracy", calculate_accuracy(classes, 'riemmanian_geometric_median'), "\n",
        "Log Euclidean Median Accuracy", calculate_accuracy(classes, 'log_euclidean_median'), "\n",
        "Trimmed Riemmanian Geometric Mean Accuracy", calculate_accuracy(classes, 'trimmed_riemmanian_mean'), "\n",
        "Trimmed Riemmanian Geometric Median Accuracy", calculate_accuracy(classes, 'trimmed_riemmanian_median'), "\n",
        "Trimmed Log Euclidean Mean Accuracy", calculate_accuracy(classes, 'log_euclidean_mean'), "\n",
        "Trimmed Log Euclidean Median Accuracy", calculate_accuracy(classes, 'log_euclidean_median'))



############################################################################################################################################
# TensorFlow
############################################################################################################################################

# import tensorflow as tf
# import tf_utils

# print("SHOULDN'T DO ANYTHING TILL HERE")
# (train_x, train_y), (test_x, test_y) = features_and_labels(generateSCMs(generateTrials(generatePositions(TYP, POS))), labels)

# # Build 3 hidden layer DNN with 10, 10, 10 units respectively.
# classifier = tf.estimator.DNNClassifier(
#         feature_columns=train_x,
#         # Two hidden layers of 10 nodes each.
#         hidden_units=[10, 10, 10],
#         # The model must choose between 4 classes.
#         n_classes=4)


# # Train the Model.
# classifier.train(
#         input_fn=lambda:tf_utils.train_input_fn(train_x, train_y, 100),
#         steps=1000)


# # Evaluate the model.
# eval_result = classifier.evaluate(
#         input_fn=lambda:tf_utils.eval_input_fn(test_x, test_y, 100))


# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


# predictions = classifier.predict(
#         input_fn=lambda:tf_utils.eval_input_fn(predict_x, labels=None, batch_size=100))

############################################################################################################################################
