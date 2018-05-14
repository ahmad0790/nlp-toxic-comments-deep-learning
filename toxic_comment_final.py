import keras
import numpy as np
import pandas as pd
import matplotlib as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, History
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

PROJECT_HOME = ''
#PROJECT_HOME = 'NLP-Project/'

def split_data_train_validate_test(data, train_percent=.8, validate_percent=.2, seed=None):
    np.random.seed(seed)
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    #shuffled_data = np.random.permutation(data)
    train = shuffled_data.iloc[0:int(train_percent*(data.shape[0])),:]
    validate = shuffled_data.iloc[int(train_percent*(data.shape[0])):int((train_percent+validate_percent)*(data.shape[0])),:]
    return train, validate

def read_data(valid_prob):
    train = pd.read_csv(PROJECT_HOME+'data/train.csv')
    test = pd.read_csv(PROJECT_HOME+'data/test.csv')
    
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] 
    train, validate = split_data_train_validate_test(train, valid_prob, 1-valid_prob)
    y_train = train[list_classes].values
    y_validate = validate[list_classes].values
    
    list_sentences_train = train["comment_text"]
    list_sentences_valid = validate["comment_text"]
    list_sentences_test = test["comment_text"]
    
    return list_sentences_train, list_sentences_test, list_sentences_valid, y_train, y_validate

def tokenizeComments(list_sentences_train, list_sentences_test, list_sentences_valid, max_words):
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts((list(list_sentences_train)+list(list_sentences_valid)))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_valid = tokenizer.texts_to_sequences(list_sentences_valid)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    
    return list_tokenized_train, list_tokenized_test, list_tokenized_valid

def getCommentLengthsDistribution(comments):
    commentsList = []
    for i in range(0, len(comments)):
        commentsList.append(len(comments[i]))
    
    plt.hist(commentsList,bins = np.arange(0,500,10))
    plt.show()

def padComments(list_tokenized_train, list_tokenized_test,list_tokenized_valid, max_comment_length):
    X_train = pad_sequences(list_tokenized_train, maxlen=max_comment_length)
    X_test = pad_sequences(list_tokenized_test, maxlen=max_comment_length)
    X_valid = pad_sequences(list_tokenized_valid, maxlen=max_comment_length)
    return X_train, X_test, X_valid


def getCommentLengthsDistribution(comments):
    commentsList = []
    for i in range(0, len(comments)):
        commentsList.append(len(comments[i]))
    
    #fig, ax = plt.subplots()
    plt.hist(commentsList,bins = np.arange(0,500,10))
    plt.xlabel('Number of Words in Comment')
    plt.ylabel('Comment Counts')
    plt.title('Histogram of Word Counts in Comments')
    plt.axvline(x=200, color='r', linestyle='dashed', linewidth=2)
    plt.show()


def buildFeedForwardModel(X_train, y, input_len, e, batch, l, opt, valid_split):
    # define the architecture for the simple feed forward network
    model = Sequential()
    model.add(Embedding(250000, 128, input_length=input_len))
    model.add(Dense(1024, input_dim=input_len, kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dropout(0.25))
    model.add(Dense(512, kernel_initializer="uniform", activation="sigmoid"))
    model.add(Dropout(0.25))
    model.add(Dense(6))
    model.add(Activation("softmax"))
    model.compile(loss=l, optimizer=opt, metrics=["accuracy"])
    model.fit(X_train, y, epochs=e, batch_size=batch, verbose=1, validation_split=valid_split)
    return model
 
def buildCNNModel(x_train, y_train, x_valid, y_valid
                  , max_features, filters, stride, kernel_size
                  , embedding_dim, bs, hidden_dim, convolved_layers
                  , act_function, loss_function, epoch_num, maxlen):

    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    
    model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
    model.add(Dropout(0.2))
    
    model.add(Conv1D(filters,
                 kernel_size,
                 activation=act_function,
                 strides=stride))
    
    if convolved_layers != 1:
        model.add(MaxPooling1D(pool_size=2))
    else:
        model.add(GlobalMaxPool1D())


    if convolved_layers == 2:
        model.add(Conv1D(filters,
                     kernel_size,
                     activation=act_function,
                     strides=stride))
        model.add(GlobalMaxPool1D())

    if convolved_layers == 3:
        model.add(Conv1D(filters,
                     kernel_size,
                     activation=act_function,
                     strides=stride))
        model.add(MaxPooling1D(pool_size=2))

        model.add(Conv1D(filters,
                     kernel_size,
                     activation=act_function,
                     strides=stride))
        model.add(GlobalMaxPool1D())

    model.add(Dense(hidden_dim))
    model.add(Dropout(0.2))
    model.add(Activation(act_function))

    model.add(Dense(6))
    model.add(Activation('sigmoid'))

    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = History()

    nn_model = model.fit(x_train, y_train,
              batch_size=bs,
              epochs=epoch_num,
              #validation_split = 0.3,
              validation_data = (x_valid, y_valid),
              callbacks=[early_stopping, history],
              shuffle = True,
              verbose = 1)
    
    #val_accuracy = model.evaluate(x_valid, y_valid, verbose=0)
    
    return model, nn_model.history

def build_LSTM_Model(x_train, y_train, x_valid, y_valid, max_features, embedding_dim, bs, 
    act_function, loss_function, epoch_num, maxlen, bidirectional):

    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
    if bidirectional == True:
        model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    else:
        model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Activation(act_function))
    model.add(Dense(6))
    model.add(Activation('sigmoid'))

    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=bs,
              epochs=epoch_num,
              validation_split = 0.2)
              #validation_data=(x_test, y_test))

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = History()

    lstm_model = model.fit(x_train, y_train,
              batch_size=bs,
              epochs=epoch_num,
              validation_data = (x_valid, y_valid),
              callbacks=[early_stopping, history],
              shuffle = True,
              verbose = 1)
    
    
    return model, lstm_model.history


def cnn_paramater_grid_search(x_train, x_valid
                            , y, y_valid
                            , MAX_WORDS
                            , max_comment_length
                            , filter_list
                            , kernel_size_list
                            , embedding_dim_list
                            , hidden_dim_list
                            , activation_list
                            , batch_size_list
                            ,seed):
    
    np.random.seed(seed)
    grid_search_results = pd.DataFrame(columns=['Model_Number','Filters','Kernel_Size','Embedding_Dims','Hidden_Dims'
                    ,'Train_Loss','Train_Acc','Valid_Loss','Valid_Acc'])
    
    i = 0
    for filter_num in filter_list:
        for window_size in kernel_size_list:
            for embedding_dim in embedding_dim_list:
                for hidden_dim in hidden_dim_list: 

                    print ('Now Training Model with Parameters (Filters, window_size, embedding dimension, hidden dimensions): ')
                    print(filter_num)
                    print(window_size)
                    print(embedding_dim)
                    print(hidden_dim)

                    model, model_hist = buildCNNModel(
                        x_train
                        , y
                        , x_valid
                        , y_valid
                        , max_features = MAX_WORDS
                        , filters = filter_num
                        , stride = 1
                        , kernel_size = window_size
                        , embedding_dim = embedding_dim
                        , bs = 128
                        , hidden_dim = hidden_dim
                        , convolved_layers = 1
                        , act_function = 'relu'
                        , loss_function = 'binary_crossentropy'
                        , epoch_num = 5
                        , maxlen = max_comment_length)

                    val_loss = model_hist["val_loss"][-1]
                    val_acc = model_hist["val_acc"][-1]
                    loss = model_hist["loss"][-1]
                    acc = model_hist["acc"][-1]
                        
                    grid_search_results.loc[i] = 0
                    grid_search_results.iloc[i,0] = i
                    grid_search_results.iloc[i,1] = filter_num
                    grid_search_results.iloc[i,2] = window_size
                    grid_search_results.iloc[i,3] = embedding_dim
                    grid_search_results.iloc[i,4] = hidden_dim
                    grid_search_results.iloc[i,5] = loss
                    grid_search_results.iloc[i,6] = acc
                    grid_search_results.iloc[i,7] = val_loss
                    grid_search_results.iloc[i,8] = val_acc

                    print(grid_search_results)
                    grid_search_results.to_csv(PROJECT_HOME+'results/grid_search_results_backup_2.csv', index = False)
                    
                    i = i + 1
    
    return grid_search_results

def predictModel(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_predictions(y_true, y_pred, threshold, score_type):
    #print (y_pred[0])
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred <= threshold] = 0
    #print (y_pred[0])
    precision = precision_score(y_true, y_pred, average=score_type)
    recall = recall_score(y_true, y_pred, average=score_type)
    accuracy = accuracy_score(y_true, y_pred)
   #conf_matrix = confusion_matrix(y_true, y_pred)
    return precision, recall, accuracy

def baseline_predictions(y_train):
    y_train_copy = y_train.copy()
    y_train_copy[:,:] = 0
    y_train_copy[y_train == 0] = 1
    results = y_train_copy.sum(axis=0)
    total_correct = results.sum()
    total_preds = len(y_train)*6
    accuracy = total_correct/total_preds
    perc_correct_each_class = results = results/len(y_train)
    return accuracy, perc_correct_each_class 


if __name__ == "__main__":

    #TEST RUN - CREATE WORD VECTORS
    MAX_WORDS = 200000
    max_comment_length = 200
    submission = pd.read_csv(PROJECT_HOME+'data/sample_submission.csv')
    list_sentences_train, list_sentences_test, list_sentences_valid, y_train, y_validate = read_data(0.8)
    list_tokenized_train, list_tokenized_test, list_tokenized_valid,  = tokenizeComments(list_sentences_train, list_sentences_test, list_sentences_valid, MAX_WORDS)
    x_train, x_test, x_valid = padComments(list_tokenized_train, list_tokenized_test, list_tokenized_valid, max_comment_length)
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    ##UNCOMMENT TO GET DISTRIBUTION OF COMMENT WORD LENGTHS
    #getCommentLengthsDistribution(list_tokenized_train)

    #MAKE BASELINE PREDICTIONS
    baseline_accuracy, baseline_perc_correct_each_class = baseline_predictions(y_train)

    y_train_base = y_train.copy()
    y_train_base[:,:] = 0
    precision_base, recall_base, accuracy_base = evaluate_predictions(y_train, y_train_base, 0.5,'micro')

    print ('The BASELINE is just predicting every comment to be non toxic')
    print ('The Precision is: ' + str(precision_base))
    print ('The Recall is: ' + str(recall_base))
    print ('The Subset Accuracy is: ' + str(accuracy_base))
    print('The Unweighted Accuracy is: ' + str(baseline_accuracy))
    print('The baseline Unweighted Accuracy for each class is:' + str(baseline_perc_correct_each_class))
    submission[list_classes] = 0
    submission.to_csv(PROJECT_HOME + '/data/final_model_cnn.csv', index = False)
    print('')

    ##BUILDING THE FEEDFORWARD
    '''
    print('LET US TRY DEEP LEARNING USING A SIMPLE FEED FORWARD NEURAL NET MODEL')
    ff_model = buildFeedForwardModel(x_train, y_train, 200, 10, 128, 'binary_crossentropy', 'sgd', 0.2)
    y_pred_ff = predictModel(lstm_model, x_test)
    precision, recall, accuracy = evaluate_predictions(y_validate, y_pred_ff, threshold_cutoff,'micro')
    print ('The Precision Overall is: ' + str(precision))
    print ('The Recall Overall is: ' + str(recall))
    print ('The Weighted Accuracy Overall is: ' + str(accuracy))
    precision, recall, accuracy = evaluate_predictions(y_validate, y_pred_validate, threshold_cutoff,None)
    print ('The Precision for each class is: ' + str(precision))
    print ('The Recall for each class is: ' + str(recall))

    y_pred_ff = predictModel(ff_model, x_test)
    submission[list_classes] = y_pred_ff
    submission.to_csv('NLP-Project/data/ff_model.csv', index = False)
    '''

    ##BUILDING CNN FINAL MODEL   
    print('LET US TRY DEEP LEARNING USING THE FINAL TUNED CNN MODEL')
    print("TRAINING A SINGLE CONV LAYER CNN WITH FILTER = 64, Kernel = 5, Embedding Dim =128, Hidden Layer Dim = 128, Batch Size = 128, Activation = Relu")
    cnn_model_1layer, cnn_model_1layer_hist = buildCNNModel(x_train, y_train
        , x_valid, y_validate
        , max_features = MAX_WORDS
        , filters = 64
        , stride = 1
        , kernel_size = 5
        , embedding_dim = 128
        , bs =128
        , hidden_dim = 128
        , convolved_layers = 1
        , act_function = 'relu'
        , loss_function = 'binary_crossentropy'
        , epoch_num = 10
        , maxlen = max_comment_length)

    val_loss = cnn_model_1layer_hist["val_loss"][-1]
    val_acc = cnn_model_1layer_hist["val_acc"][-1]

    ##evaluate model 
    print('The Validation Loss overall is: ' + str(val_loss))
    print('The Validation Unweighted Accuracy overall is: ' + str(val_acc))

    y_pred_validate = predictModel(cnn_model_1layer, x_valid)
    threshold_cutoff = 0.5
    precision, recall, accuracy = evaluate_predictions(y_validate, y_pred_validate, threshold_cutoff,'micro')
    print ('The Precision Overall is: ' + str(precision))
    print ('The Recall Overall is: ' + str(recall))
    print ('The Weighted Accuracy Overall is: ' + str(accuracy))
    precision, recall, accuracy = evaluate_predictions(y_validate, y_pred_validate, threshold_cutoff,None)
    print ('The Precision for each class is: ' + str(precision))
    print ('The Recall for each class is: ' + str(recall))

    ##make predictions on kaggle test set
    y_pred = predictModel(cnn_model_1layer, x_test)
    submission[list_classes] = y_pred
    submission.to_csv('NLP-Project/data/final_model_cnn.csv', index = False)


    ##BUILDING THE LSTM MODEL
    print("RUNNING THE LSTM MODEL")
    lstm_model, lstm_model_hist = build_LSTM_Model(x_train, y_train, x_valid, y_validate, MAX_WORDS, 128, 128, 'relu', 'binary_crossentropy',5,max_comment_length,True)
    val_loss = lstm_model_hist["val_loss"][-1]
    val_acc = lstm_model_hist["val_acc"][-1]

    ##evaluate model 
    print('THE PERFORMANCE METRICS FOR THE LSTM MODEL')
    print('The Validation Loss overall is: ' + str(val_loss))
    print('The Validation Unweighted Accuracy overall is: ' + str(val_acc))

    y_pred_validate = predictModel(lstm_model, x_valid)
    threshold_cutoff = 0.5
    precision, recall, accuracy = evaluate_predictions(y_validate, y_pred_validate, threshold_cutoff,'micro')
    print ('The Precision Overall is: ' + str(precision))
    print ('The Recall Overall is: ' + str(recall))
    print ('The Weighted Accuracy Overall is: ' + str(accuracy))
    precision, recall, accuracy = evaluate_predictions(y_validate, y_pred_validate, threshold_cutoff,None)
    print ('The Precision for each class is: ' + str(precision))
    print ('The Recall for each class is: ' + str(recall))
    y_pred_lstm = predictModel(lstm_model, x_test)
    submission[list_classes] = y_pred_lstm
    submission.to_csv('NLP-Project/results/lstm_model_v2.csv', index = False)

    #THE END - FOR EXPERIMENTS DONE PLEASE REFER TO experiments.py