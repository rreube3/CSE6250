#
# # Paper: "Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention"
#
# # 5-Label classification
#
import csv
import string
from nltk import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from keras.utils.np_utils import to_categorical
import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPool2D
from keras.layers import Conv2D
from keras.layers import Concatenate
# from keras.optimizers import Adam
# https://programmerah.com/keras-nightly-import-package-error-cannot-import-name-adam-from-keras-optimizers-29815/
from keras.optimizer_v2 import adam as adam_v2
# https://stackoverflow.com/questions/62707558/importerror-cannot-import-name-adam-from-keras-optimizers
# from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
import time

###Data files
punctuations = list(string.punctuation)

ip_txt_file = "data/500_Reddit_users_posts_labels.csv"  # CSV file: "User", "Post", "Label"


ip_feat_file = "data/External_Features.csv"            # CSV file: "User", "Features"

w2v_file = {'file': 'data/numberbatch-en.txt', 'is_binary': False}

###file used to save the model outputs with 314 features
op_file = "data/Result_5-Label_Classification.tsv"

###file used to save the model outputs with 300 features
# op_file = "data/Result_5-Label_Classification_300features.tsv"

severity_classes = {'Supportive': 0, 'Indicator': 1, 'Ideation': 2, 'Behavior': 3, 'Attempt': 4}
##Hyper-parameters
sys_params = {'emb_dim': 300,
              'max_sent_len': 1500,
              'str_padd': '@PADD',
              'cross_val': 5}

cnn_params = {'no_filters': 100,
              'kernels': [3, 4, 5],
              'channel': 1,
              'c_stride': (1, sys_params['emb_dim']),
              'pad': 'same',
              'ip_shape': (sys_params['max_sent_len'], sys_params['emb_dim'], 1),
              'c_activ': 'relu',
              'drop_rate': 0.3,
              'dense_1_unit': 128,
              'dense_2_unit': 128,
              'dense_activ': 'relu',
              'op_unit': 5,             # 5-Label classification
              'op_activ': 'softmax',
              'l_rate': 0.001,
              'loss': 'categorical_crossentropy',
              'batch': 4,
              'epoch': 50,
              'verbose': 1}

intermediate_layer = 'flat_drop'    # for extracting features from CNN

print ('\nSystem Parameters: ', sys_params)
print ('\nCNN Parameters: ', cnn_params)

# Read the input CSV file
def read_ip_file(ip_file):

    padd = sys_params['str_padd']
    max_len = sys_params['max_sent_len']

    x_data, y_data = [], []

    if ip_file:
        with open(ip_file,"r") as csv_file:

            # Exclude the first line (header)
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)

            # Loop through each line
            for row in csv_reader:

                sent = row[1]

                # Remove non-ascii characters
                printable = set(string.printable)

#               # Remove punctuation
                # Remove punctuation and non-ascii characters
                lst_tokens = [item.lower().strip("".join(punctuations)) for item in word_tokenize(sent) if
                              item not in punctuations and item not in printable]

                # Strip the sentence if it exceeds the max length
                if len(lst_tokens) > max_len:
                    lst_tokens = lst_tokens[:max_len]

                # Padd the sentence if the length is less than max length
                elif len(lst_tokens) < max_len:
                    for j in range(len(lst_tokens), max_len):
                        lst_tokens.append(padd)

                y_data.append(severity_classes[row[2].strip()])
                x_data.append(lst_tokens)

    return x_data, y_data

# Vectorize the input data using pretrained word2vec embedding lookup
def vectorize_data(lst_input):

    padd = sys_params['str_padd']
    wv_size = sys_params['emb_dim']

    # Load the pre-trained word2vec model
    w2v_model = KeyedVectors.load_word2vec_format(w2v_file['file'], binary=w2v_file['is_binary'])

    # Get the word2vec vocabulary
    vocab =w2v_model.key_to_index
    #
    padding_zeros = np.zeros(wv_size, dtype=np.float32)

    x_data = []

    # Loop through each sentence
    for sent in lst_input:
        emb = []
        for tok in sent:

            # Zero-padding for padded tokens
            if tok.lower() == padd:
                emb.append(list(padding_zeros))

            # Get the token embedding from the word2vec model
            elif tok.lower() in vocab.keys():
                emb.append(w2v_model[tok.lower()].astype(float).tolist())
            # Zero-padding for out-of-vocab tokens
            else:
                emb.append(list(padding_zeros))

        x_data.append(emb)

    del w2v_model, vocab

    return np.array(x_data)

# Prepare the input data
def read_data(ip_file):

    # Read the input file
    x_data, y_data = read_ip_file(ip_file)

    # Vectorize the data
    x_data = vectorize_data(x_data)

    # # Reshape the data for CNN
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)  # last argument 1 indicates #channel

    # Convert into numpy array
    x_data, y_data = np.array(x_data), np.array(y_data)

    return x_data, y_data

# # Read additional external features
def read_external_features(raw_data, raw_features):

    user_ids, features = [], []

    # Read the user ids from raw_data csv and append them to "user_ids" list
    with open(raw_data) as file:
        for line in file:
            split = line.strip().split(',')
            user_ids.append(split[0])

    with open(raw_features) as csv_file:
        # Read the feature file
        csvreader = csv.reader(csv_file, delimiter=',')
        # Skip the header row
        header = next(csvreader)
        # Loop through each user feature row
        for row in csvreader:
            # Convert the feature score into float
            scores = [float(value) for value in row[1:]]
            # Append feature score list for each user to final features list
            features.append(scores)

    # Return numpy array of features
    return np.array(features)

def get_cnn_model():
    seq_len = sys_params['max_sent_len']
    emb_dim = sys_params['emb_dim']

    l_ip = Input(shape=(seq_len, emb_dim, 1), dtype='float32')
    lst_convfeat = []
    for filter in cnn_params['kernels']:
        l_conv = Conv2D(filters=cnn_params['no_filters'], kernel_size=(filter, emb_dim), strides=cnn_params['c_stride'],
                        padding=cnn_params['pad'], data_format='channels_last', input_shape=cnn_params['ip_shape'],
                        activation=cnn_params['c_activ'])(l_ip)
        l_pool = MaxPool2D(pool_size=(seq_len, 1))(l_conv)
        lst_convfeat.append(l_pool)

    l_concat = Concatenate(axis=1)(lst_convfeat)
    l_flat = Flatten()(l_concat)
    l_drop = Dropout(rate=cnn_params['drop_rate'], name='flat_drop')(l_flat)

    l_op = Dense(units=cnn_params['op_unit'], activation=cnn_params['op_activ'], name='cnn_op')(l_drop)

    final_model = Model(l_ip, l_op)
    final_model.compile(optimizer=adam_v2.Adam(learning_rate=cnn_params['l_rate']), loss=cnn_params['loss'], metrics=['accuracy'])    # 'categorical_crossentropy'

    return final_model

# Returns a MLP model for final classification
def get_mlp_model(ip_dim):

    mlp_model = Sequential()

    mlp_model.add(Dense(units=cnn_params['op_unit'], activation=cnn_params['op_activ'], name='classif_op',
                            input_dim=ip_dim))

    mlp_model.compile(optimizer=adam_v2.Adam(learning_rate=cnn_params['l_rate']), loss=cnn_params['loss'],
                          metrics=['accuracy'])
    return mlp_model

# Compute Precision, Recall, and F1-score
def get_prf1_score(y_true, y_pred):
    tp, fp, fn = 0.0, 0.0, 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            tp += 1
        elif y_pred[i] > y_true[i]:
            fp += 1
        else:
            fn += 1
    if tp == 0:
        tp = 1.0
    if fp == 0:
        fp = 1.0
    if fn == 0:
        fn  = 1.0
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = 2 * P * R / (P + R)
    print ('\nPrecision: {0}\t Recall: {1}\t F1-Score: {2}'.format(P, R, F))
    return {'P': P, 'R': R, 'F': F}
###new added file
def oe_score(y_true, y_pred):
    oe_no=0
    nt=len(y_pred)
    for i in range(nt):
        if abs(y_pred[i]-y_true[i])>1:
            oe_no+=1
    OE= oe_no/nt
    print('OE:{}'.format(OE))
    return {'OE': OE}  
def scores(ypred,ytest):
    ypred = np.argmax(ypred, axis=-1)
    ytest = np.argmax(ytest, axis=-1)
    score = get_prf1_score(ytest, ypred)
    oe=oe_score(ytest, ypred)
    return(score,oe)

if __name__ == '__main__':
    ###begin time:
    start_time=time.time()

    with open(op_file, 'w') as of:

        x_data, y_data = read_data(ip_txt_file)

        ext_feature = read_external_features(ip_txt_file, ip_feat_file)

        cv_count = 0
        k_score = []
        oescore = []
       
        kscore_svmln=[]
        oescore_svmln=[]
        
        # Stratified cross-validation
        skf = StratifiedKFold(n_splits=sys_params['cross_val'])
        skf.get_n_splits(x_data, y_data)

        # Run the model for each splits
        for train_index, test_index in skf.split(x_data, y_data):
            cv_count += 1
            print ('\nRunning Stratified Cross Validation: {0}/{1}...'.format(cv_count, sys_params['cross_val']))

            x_train, x_test = x_data[train_index], x_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]

            # Convert the class labels into categorical
            y_train, y_test = to_categorical(y_train), to_categorical(y_test)

            # Reshape the data for CNN
            x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

            # External features for this particular split
            train_ext_feat, test_ext_feat = ext_feature[train_index], ext_feature[test_index]

            # CNN model for training on the embedded text input
            cnn_model = get_cnn_model()
            print (cnn_model.summary())

            # Train the model
            cnn_model.fit(x=x_train, y=y_train, batch_size=cnn_params['batch'], epochs=cnn_params['epoch'], verbose=cnn_params['verbose'])
            
            # Trained model for extracting features from intermediate layer
            model_feat_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(intermediate_layer).output)

            # Get CNN gerated features
            train_cnn_feat = model_feat_extractor.predict(x_train)
            test_cnn_feat = model_feat_extractor.predict(x_test)

            # Merge the CNN generated features with the external features
            x_train_features = []
            for index, cnn_feature in enumerate(train_cnn_feat):
                tmp_feat = list(cnn_feature)
                ###When run model without these 14 features, comment the following lines. So these 14 features won't be added to the input data
                tmp_feat.extend(list(train_ext_feat[index]))
                x_train_features.append(np.array(tmp_feat))

            x_test_features = []
            for index, cnn_feature in enumerate(test_cnn_feat):
                tmp_feat = list(cnn_feature)
                ###When run model without these 14 features, comment the following lines. So these 14 features won't be added to the input data
                tmp_feat.extend(list(test_ext_feat[index]))
                x_test_features.append(np.array(tmp_feat))

            # Convert the list into numpy array
            x_train_features = np.array(x_train_features)
            x_test_features = np.array(x_test_features)

            del train_cnn_feat, test_cnn_feat

            # Get the MLP model for final classification
            mlp_model = get_mlp_model(ip_dim = len(x_train_features[0]))
            print (mlp_model.summary())

            tc = time.time()

            # Train the MLP model
            mlp_model.fit(x=x_train_features, y=y_train, batch_size=cnn_params['batch'], epochs=cnn_params['epoch'], verbose=cnn_params['verbose'])
                        
            print ('\nTime elapsed in training CNN: ', str(datetime.timedelta(seconds=time.time() - tc)))

            print ('\nEvaluating on Test data...\n')
            # # Print Loss and Accuracy
            model_metrics = mlp_model.evaluate(x_test_features, y_test)

            for i in range(len(model_metrics)):
                print (mlp_model.metrics_names[i], ': ', model_metrics[i])

            y_pred = mlp_model.predict(x_test_features)
            
            score,oe=scores(y_pred,y_test)
            k_score.append(score)
            oescore.append(oe)
            y_pred = np.argmax(y_pred, axis=-1)
            y_test_mlp = np.argmax(y_test, axis=-1)


            # Scikit-learn classification report (P, R, F1, Support)
            report = classification_report(y_test_mlp, y_pred)
            print (report)

            of.write('Cross_Val:\n')
            for i in range(len(y_pred)):
                of.write('\t'.join([str(y_test_mlp[i]), str(y_pred[i])]) + '\n')
           
            ###SVM model--linear
            ytrain_new=[key for ss in y_train for key,val in enumerate(ss) if val==1]
            ytest_new=[key for ss in y_test for key,val in enumerate(ss) if val==1]

            from sklearn.svm import SVC
            model_svm_ln = SVC(kernel='linear', probability=True)
            model_svm_ln.fit(x_train_features, np.array(ytrain_new))
            y_pred_svmln=model_svm_ln.predict(x_test_features)
            score_svmln = get_prf1_score(ytest_new, y_pred_svmln.tolist())
            oe_svmln=oe_score(ytest_new, y_pred_svmln.tolist())
            kscore_svmln.append(score_svmln)
            oescore_svmln.append(oe_svmln)       
            del x_train, y_train


        print('=============5 Labels MLP model: 300 Features from CNN model + 14 External Features\n')
#         print('=============CNN model: 300 Features\n')
        print (k_score)        
        print("\n",oescore)

        avgP = np.average([score['P'] for score in k_score])
        avgR = np.average([score['R'] for score in k_score])
        avgF = np.average([score['F'] for score in k_score])
        avgOE= np.average([score['OE'] for score in oescore])

        print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP, avgR, avgF,avgOE))
        
        print('=============5 Labels SVM Linear: 300 Features from CNN model + 14 External Features\n')

        print (k_score)        
        print("\n",oescore)

        avgP_svmln = np.average([score['P'] for score in kscore_svmln])
        avgR_svmln = np.average([score['R'] for score in kscore_svmln])
        avgF_svmln = np.average([score['F'] for score in kscore_svmln])
        avgOEs_svmln = np.average([score['OE'] for score in oescore_svmln])

        print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP_svmln,avgR_svmln,avgF_svmln,avgOEs_svmln ))
        end_time=time.time()
        print("Time Completing the tast:{} hours".format((end_time-start_time)/3600))