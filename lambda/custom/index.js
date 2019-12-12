/* eslint no-use-before-define: 0 */
// sets up dependencies
const Alexa = require('ask-sdk-core');
const i18n = require('i18next');
const sprintf = require('i18next-sprintf-postprocessor');
const arvoice = require('./arvoice');

// core functionality for skill
const GetNewFeedHandler = {
  canHandle(handlerInput) {
    const request = handlerInput.requestEnvelope.request;
    // checks request type
    return request.type === 'LaunchRequest'
      || (request.type === 'IntentRequest'
        && request.intent.name === 'GetNewFeedIntent');
  },
  handle(handlerInput) {
    const requestAttributes = handlerInput.attributesManager.getRequestAttributes();
    // gets a random feed by assigning an array to the variable
    // the random feed from the array will be selected by the i18next library
    // the i18next library is set up in the Request Interceptor
    const randomFeed = requestAttributes.t('FEEDS');
    arvoice.sendData(randomFeed);
    // concatenates a standard message with the random feed
    const speakOutput = requestAttributes.t('GET_FEED_MESSAGE') + randomFeed;

    return handlerInput.responseBuilder
      .speak(speakOutput)
      .withSimpleCard(requestAttributes.t('SKILL_NAME'), randomFeed)
      .getResponse();
  },
};

const HelpHandler = {
  canHandle(handlerInput) {
    const request = handlerInput.requestEnvelope.request;
    return request.type === 'IntentRequest'
      && request.intent.name === 'AMAZON.HelpIntent';
  },
  handle(handlerInput) {
    const requestAttributes = handlerInput.attributesManager.getRequestAttributes();
    return handlerInput.responseBuilder
      .speak(requestAttributes.t('HELP_MESSAGE'))
      .reprompt(requestAttributes.t('HELP_REPROMPT'))
      .getResponse();
  },
};

const FallbackHandler = {
  // 2018-Aug-01: AMAZON.FallbackIntent is only currently available in en-* locales.
  //              This handler will not be triggered except in those locales, so it can be
  //              safely deployed for any locale.
  canHandle(handlerInput) {
    const request = handlerInput.requestEnvelope.request;
    return request.type === 'IntentRequest'
      && request.intent.name === 'AMAZON.FallbackIntent';
  },
  handle(handlerInput) {
    const requestAttributes = handlerInput.attributesManager.getRequestAttributes();
    return handlerInput.responseBuilder
      .speak(requestAttributes.t('FALLBACK_MESSAGE'))
      .reprompt(requestAttributes.t('FALLBACK_REPROMPT'))
      .getResponse();
  },
};

const ExitHandler = {
  canHandle(handlerInput) {
    const request = handlerInput.requestEnvelope.request;
    return request.type === 'IntentRequest'
      && (request.intent.name === 'AMAZON.CancelIntent'
        || request.intent.name === 'AMAZON.StopIntent');
  },
  handle(handlerInput) {
    const requestAttributes = handlerInput.attributesManager.getRequestAttributes();
    return handlerInput.responseBuilder
      .speak(requestAttributes.t('STOP_MESSAGE'))
      .getResponse();
  },
};

const SessionEndedRequestHandler = {
  canHandle(handlerInput) {
    const request = handlerInput.requestEnvelope.request;
    return request.type === 'SessionEndedRequest';
  },
  handle(handlerInput) {
    console.log(`Session ended with reason: ${handlerInput.requestEnvelope.request.reason}`);
    return handlerInput.responseBuilder.getResponse();
  },
};

const ErrorHandler = {
  canHandle() {
    return true;
  },
  handle(handlerInput, error) {
    console.log(`Error handled: ${error.message}`);
    console.log(`Error stack: ${error.stack}`);
    const requestAttributes = handlerInput.attributesManager.getRequestAttributes();
    return handlerInput.responseBuilder
      .speak(requestAttributes.t('ERROR_MESSAGE'))
      .reprompt(requestAttributes.t('ERROR_MESSAGE'))
      .getResponse();
  },
};

const LocalizationInterceptor = {
  process(handlerInput) {
    const localizationClient = i18n.use(sprintf).init({
      lng: handlerInput.requestEnvelope.request.locale,
      resources: languageStrings,
    });
    localizationClient.localize = function localize() {
      const args = arguments;
      const values = [];
      for (let i = 1; i < args.length; i += 1) {
        values.push(args[i]);
      }
      const value = i18n.t(args[0], {
        returnObjects: true,
        postProcess: 'sprintf',
        sprintf: values,
      });
      if (Array.isArray(value)) {
        return value[Math.floor(Math.random() * value.length)];
      }
      return value;
    };
    const attributes = handlerInput.attributesManager.getRequestAttributes();
    attributes.t = function translate(...args) {
      return localizationClient.localize(...args);
    };
  },
};

const skillBuilder = Alexa.SkillBuilders.custom();

exports.handler = skillBuilder
  .addRequestHandlers(
    GetNewFeedHandler,
    HelpHandler,
    ExitHandler,
    FallbackHandler,
    SessionEndedRequestHandler,
  )
  .addRequestInterceptors(LocalizationInterceptor)
  .addErrorHandlers(ErrorHandler)
  .lambda();

// translations
const enData = {
  translation: {
    SKILL_NAME: 'Machine Learning Vocabulary',
    GET_FEED_MESSAGE: 'Here\'s your feed: ',
    HELP_MESSAGE: 'You can say tell me a machine learning feed, or, you can say exit... What can I help you with?',
    HELP_REPROMPT: 'What can I help you with?',
    FALLBACK_MESSAGE: 'The Machine Learning Vocabulary skill can\'t help you with that.  It can help you discover feeds, terms and definitions about machine learning if you say tell me a machine learning feed. What can I help you with?',
    FALLBACK_REPROMPT: 'What can I help you with?',
    ERROR_MESSAGE: 'Sorry, an error occurred.',
    STOP_MESSAGE: 'Goodbye!',
    FEEDS:
      [
        'A\/B testing is a statistical way of comparing two or more techniques, typically an incumbent against a new rival.',
        'Accuracy is the fraction of predictions that a classification model got right.',
        'An activation function is a function that takes in the weighted sum of all of the inputs from the previous layer and then generates and passes an output value to the next layer.',
        'Active learning is a training in which the algorithm chooses some of the data it learns from.',
        'AdaGrad is a sophisticated gradient descent algorithm that rescales the gradients of each parameter, effectively giving each parameter an independent learning rate.',
        'The bag of words is a representation of the words in a phrase or passage, irrespective of order.',
        'Baseline is a simple model or heuristic used as reference point for comparing how well a model is performing.',
        'Batch is the set of examples used in one iteration of model training.',
        'Batch size is the number of examples in a batch.',
        'Bayesian neural network is a probabilistic neural network that accounts for uncertaintly in weights and outputs.',
        'Binary classification is a type of classification task that outputs one of two mutually exclusive classes.',
        'Boosting is a machine learning that iteratively combines a set of simple and not very accurate classifiers into a classifier with high accuracy by upweighting the examples that the model is currently misclassifying.',
        'Bucketing is converting a feature into multiple binary features called buckets or bins, typically based on value range.',
        'Candidate generation is the initial set of recommendations chosen by a recommendation system.',
        'Centroid is the center of a cluster as determined by a k-means or k-median algorithm.',
        'Checkpoint is data that captures the state of the variables of a model at a particular time.',
        'A class is one of a set of enumerated target values for a label.',
        'Classification model is a type of machine learning model for distinguishing among two or more discrete classes.',
        'Classification threshold is a scalar value criterion that is applied to a model\'s predicted score in order to separate the positive class from the negative class.',
        'A class imbalanced data set is a binary classification problem in which the labels for the two classes have significantly different frequencies.',
        'Clipping is a technique for handling outliers.',
        'Clustering is grouping related examples, particularly during unsupervised learning.',
        'Collaborative filtering is making predictions about the interests of one user based on the interests of many other users.',
        'Confirmation bias is the tendency to search for, interpret, favor, and recall information in a way that confirms one\'s preexisting beliefs or hypotheses.',
        'Continuous feature is a floating point feature with an infinite range of possible values.',
        'Convenience sampling is using a data set not gathered scientifically in order to run quick experiments.',
        'Convergence refers to a state reached during training in whcih training loss and validation loss change very little or not at all with each iteration after a certain number of iterations.',
        'A convex function is a function in which the region above the graph of the function is a convex set.',
        'A convex optimization is the process of using mathematical techniques such as gradient descent to find the minimum of a convex function.',
        'Convex set is a subset of Euclidean space that a line drawn between any two points in the subset remains completely within the subset.',
        'Convolution is a mixture of two functions.',
        'Convolutional layer is a layer of a deep neural network in which a convolutional filter passes along an input matrix.',
        'Convolutional neural network is a neural network in which at least one layer is a convolutional layer.',
        'Crash blossom is a sentence or phrase with an ambiguous meaning.',
        'Cross validation is a mechanism for estimating how well a model will generalize to new data by testing the model agains one or more non-overlapping data subsets withheld from the training set.',
        'Data analysis is obtaining an understanding of data by considering samples, measurement, and visualization.',
        'Data augmentation is artificially boosting the range and number of training examples by transforming existing examples to create additional examples.',
        'Dataframe is a popular data type for representing data sets in Pandas.',
        'A data set is a collection of examples.',
        'A decision boundary is the separator between classes learned by a model in a binary class or multi-class classification problems.',
        'A decision tree is a model represented as a sequence of branching statements.',
        'Deep model is a type of neural network containing multiple hidden layers.',
        'A deep neural network is a synonym for deep model.',
        'A dense layer is synonym for fully connected layer.',
        'A device is a category of hardware that can run a TensorFlow session, including CPUs, GPUs, and TPUs.',
        'A discrete feature is a feature with a finite set of possible values.',
        'A discriminative model is a model that predicts labels from a set of one or more features.',
        'A dynamic model is a model that is trained online in a continuously updating fashion.',
        'Eager execution is a TensorFlow programming environment in which operations run immediately.',
        'Embeddings is a categorical feature represented as a continuous valued feature.',
        'Epoch is a full training pass over the entire data set such that each example has been seen once.',
        'Example is one row of a data set.',
        'False negative is an example in which the model mistakenly predicted the negative class.',
        'False positive is an example in which the model mistakenly predicted the positive class.',
        'A feature is an input variable used in making predictions.',
        'A few shot learning is a machine learning approach, often used for object classification, designed to learn effective classifiers from only a small number of training examples.',
        'Generalization refers to your model\'s ability to make correct predictions on new, previously unseen data as opposed to the data used to train the model.',
        'A gradient is the vector of partial derivatives with respect to all of the independent variables.',
        'A graph, in TensorFlow, is a computation specification.',
        'Heuristic is a practical and non optimal solution to a problem, which is sufficient for making progress or for learning from.',
        'Holdout data are examples intentionally not used during training.',
        'Hyperplace is a boundary that separates a space into two subspaces.',
        'Inference often refers to the process of making predictions by applying the trained model to unlabeled examples.',
        'Input function, in TensorFlow, is a function that returns input data to the training, evaluation, or prediction method of an Estimator.',
        'The input layer is the first layer in a neural network.',
        'An instance is a synonym for example.',
        'An iteration is a single update of a model\'s weights during training.',
        'K-means is a popular clustering algorithm that groups examples in unsupervised learning.',
        'K-median is a clustering algorithm closely related to k-means.',
        'A label in supervised learning, is the answer or result portion of an example.',
        'A lambda is a synonym for regularization rate.',
        'A layer is a set of neurons in a neural network that process a set of input features, or the output of those neurons.',
        'A learning rate is a scalar used to train a model via gradient descent.',
        'A linear regression is a type of regression model that outputs a continuous value from a linear combination of input features.',
        'A logistic regression is a model that generates a probability for each possible discrete label value in classification problems by applying a sigmoid function to a linear prediction.',
        'Logits is the vector of raw predictions that a classification model generates, which is ordinarily then passed to a normalization function.',
        'Log loss is the loss function used in binary logistic regression.',
        'A loss is a measure of how far a model\'s predictions are from its label.',
        'A loss curve is a graph of loss as a function of training iterations.',
        'Majority class is the more common label in a class imbalanced data set.',
        'Matrix factorization is a mechanism for finding the matrices whose dot product approximates a target matrix.',
        'Metric is a number that you care about.',
        'A mini batch is a small, randomly selected subset of the entire batch of examples run together in a single iteration of training or inference.',
        'A minority class is the less common label in a class imbalanced data set.',
        'A model is the representation of what a machine learning system has learned from the training data.',
        'Model function is the function within an Estimator that implements machine learning training, evaluation, and inference.',
        'A model training is the process of determining the best model.',
        'Natural language understanding is determining a user\'s intentions based on what the user typed or said.',
        'A negative class is one class is termed positive and the other is termed negative.',
        'Neuron is a node in a neural network.',
        'Noise is anything that obscures the signal in a data set.',
        'Numerical data are features represented as integers or real-valued numbers.',
        'Numpy is an open source math library that provides efficient array operations in Python.',
        'Objective is a metric that your algorithm is trying to optimize.',
        'Optimizer is a specific implementation of the gradient descent algorithm.',
        'Outliers are values distant from most other values.',
        'Overfitting is creating a model that matches the training data so closely that the model fails to make correct predictions on new data.',
        'Parameter is a variable of a model that the machne learning system trains on its own.',
        'Partial derivative is a derivative in which all but one of the variables is considered a constant.',
        'Perplexity is one measure of how well a model is accomplishing its task.',
        'Pipeline is the infrastructure surrounding a machine learning algorithm.',
        'A precision is a metric for classification models.',
        'A prediction is a model\'s output when provided with an input example.',
        'Prediction bias is a value indicating how far apart the average of predictions is from the average of labels in the data set.',
        'Prior belief is what you believe about the data before you being training on it.',
        'Quantile is each bucket in quantile bucketing.',
        'Queue is a TensorFlow operation that implements a queue data structure.',
        'A rater is a human who provides labels in examples.',
        'Recommendation system is a system that selects for each user a relatively small set of desirable items from a large corpus.',
        'Recurrent neural network is a neura network that is intentionally run multiples times, where parts of each run feed into the next run.',
        'Regression model is a type of model that outputs continuous values.',
        'Regularization is the penalty on a model\'s complexity.',
        'Regularization rate is a scalar value, represented as lambda, specifying the relative importance of the regularization function.',
        'Representation is the process of mapping data to useful features.',
        'Saved model is the recommended format for saving and recovering TensorFlow models.',
        'Scikit-learn is a popular open-source machine learning platform.',
        'Scoring is the part of a recommendation system that provides a value or ranking for each item produced by the candidate generation phase.',
        'A sequence model is a model whose inputs have a sequential dependence.',
        'Softmax is a function that provides probabilities for each possible class in a multi-class classification model.',
        'Sparse feature is a feature vector whose values are predominantly zero or empty.',
        'Sparse representation is a representation of a tensor that only stores nonzero elements.',
        'Squared loss is the loss function used in linear regression.',
        'Static model is a model that is trained offline.',
        'A step is a forward and backward evaluation of one batch.',
        'Supervised machine learning is training a model from input data and its corresponding labels.',
        'Temporal data is data recorded at different points in time.',
        'Tensor is the primary data structure in TensorFlow programs.',
        'TensorFlow is a large-scale, distributed, machine learning platform.',
        'Test set is the subset of the data set that you use to test your model after the odel has gone through initial vetting by the validation set.',
        'Time series analysis is a subfield of machine learning and statistics that analyzes temporal data.',
        'Training is the process of determining the ideal parameters comprising a model.',
        'Training set is the subset of the data set used to train a model.',
        'Transfer learning is transferring information from one machine learning task to another.',
        'Unsupervised machine learning is training a model to find patterns in a data set, typically an unlabeled data set.',
        'Weight is a coefficient for a feature in a linear model, or an edge in a deep network.',
      ],
  },
};

const enauData = {
  translation: {
    SKILL_NAME: 'Machine Learning Vocabulary',
  },
};

const encaData = {
  translation: {
    SKILL_NAME: 'Machine Learning Vocabulary',
  },
};

const engbData = {
  translation: {
    SKILL_NAME: 'Machine Learning Vocabulary',
  },
};

const eninData = {
  translation: {
    SKILL_NAME: 'Machine Learning Vocabulary',
  },
};

const enusData = {
  translation: {
    SKILL_NAME: 'Machine Learning Vocabulary',
  },
};

// constructs i18n and l10n data structure
// translations for this sample can be found at the end of this file
const languageStrings = {
  'en': enData,
  'en-AU': enauData,
  'en-CA': encaData,
  'en-GB': engbData,
  'en-IN': eninData,
  'en-US': enusData,
};