import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras import layers, Input, regularizers
from tensorflow.keras.models import Model, Sequential
import matplotlib.pyplot as plt
from datetime import datetime

def _get_major_full_name(x):
    # x has to be a Series
    major_names = {
        "afres":"African American Studies",
        "arcre":"Agriculture",
        "asies":"Asian Studies",
        "biogy":"Biology",
        "chery":"Chemistry",
        "comce":"Computer Science",
        "comns":"Communications",
        "crigy":"Criminology",
        "dance":"Dance",
        "eduon":"Education",
        "engsh":"English",
        "envce":"Environmental Science",
        "g_art":"General Art",
        "g_bns":"General Business",
        "g_eng":"General Engieering",
        "g_hst":"General History",
        "g_hth":"General Health",
        "g_lng":"General Language",
        "genes":"Gender Studies",
        "geogy":"Geology",
        "geohy":"Geography",
        "intes":"International Studies",
        "lincs":"Linguistics",
        "matcs":"Mathematics",
        "music":"Music",
        "neuce":"Neuroscience",
        "perrt":"Performance Art",
        "phihy":"Philosophy",
        "phycs":"Physics",
        "polcs":"Politics",
        "psygy":"Psychology",
        "socgy":"Sociology",
        "stacs":"Statistics",
        "digia":"Digital Art",
        "visrt":"Visual Art",
        'accng':"Accounting",
        'busss':"Business",
        'ecocs':"Economics",
        'marng':"Marketing",
        'cheng':"Chemical Engineering",
        'civng':"Civic Engineering",
        'eleng':"Electronic Engineering",
        'engng':"Engieering",
        'matce':"Materials Science",
        'mecng':"Mechanical Engineering",
        'hisry':"History",
        'arcgy':"Archeology",
        'antgy':"Anthropology",
        'clacs':"Classics",
        'relon':"Religion",
        'heace':"Health Science",
        'kingy':"Kinesiology",
        'nurng':"Nursing",
        'frech':"French",
        'geran':"German",
        'itaan':"Italian",
        'spash':"Spanish"
    }


    return [major_names.get(e) for e in x]

def _clean_major_all(major_all, threshold):
    # input
        # major_all = pd.DataFrame(MAJOR_ALL_PATH)
        # threshold is largest proportion of missing cases tolerated
    # process
        # remove rows that have too much missing (Likert only)
        # remove rows that have demographics missing
    # output
        # cleaned major_all_df
    likert_items = []
    pattern = re.compile(r"l\d[^\r\n]+\d")
    for col in major_all.columns:
        if re.match(pattern, col) is not None:
            likert_items.append(col)
    # if missing data exceeds a proportion, the row is deleted
    ncol = major_all[likert_items].shape[1]
    passed = major_all[likert_items].isna().sum(axis=1) <= ncol*threshold
    # also has to not miss any demographics
    # somehow missing cases have been coded as "null_unknonw" (string, not NA)
    passed = passed & (major_all[['age', 'certificate', 'education', 'school', 'gender']] == 'null_unknown').sum(axis=1) == 0
    cleaned_df = major_all.loc[passed, :].fillna(0.).reset_index(drop=True)
    return cleaned_df

def _get_major_is_top_n_df(
    TOP_N,
    MAJOR_ALL_PATH='data/major_all.csv',
    threshold=.3,
    generic_only=False,
    use_max=True # use Max when lumping 50 majors to 33, else uses mean
):
    '''
        hi!
    '''
    d = _clean_major_all(
        pd.read_csv(MAJOR_ALL_PATH, dtype={
                'age': str,
                'certificate': str,
                'education': str,
                'school': str,
                'gender': str
            }),
        threshold=threshold
    )
    # get a list of unique majors
    major_set = set()
    # only counting forced-choice sections, after oct 09 discussions
    pattern = re.compile(r"f[4-7]([^\r\n]+)\d")
    for col in d.columns:
        m = re.match(pattern, col)
        if m is not None and m.group(1) != "engng":
            major_set.add(m.group(1))
    all_majors = list(major_set)
    # getting a sum of scores for each major
    major_scores_df = pd.DataFrame()
    def ipsative_check(major, col):
        m = re.match("f[4-7]"+major+"\d", col)
        return m is not None

    for major in all_majors:
        # oops forgot to enforce forced choice
        cols_for_the_major = [col for col in d.columns if ipsative_check(major, col)]
        major_frame = d.loc[:,cols_for_the_major]
        major_scores_df[major] = major_frame.sum(axis=1)

    # returning only 33 more generic majors
    if generic_only:
        # mapping specific majors to general majors
        major_mapping = {
            'g_art': ['digia', 'visrt'],
            'g_bns': ['accng', 'busss', 'ecocs', 'marng'],
            'g_eng': ['cheng', 'civng', 'eleng', 'matce', 'mecng'],
            'g_hst': ['hisry', 'arcgy', 'antgy', 'clacs', 'relon'],
            'g_hth': ['heace', 'kingy', 'nurng'],
            'g_lng': ['frech', 'geran', 'itaan', 'spash']
        }

        for general_major, specific_majors_list in major_mapping.items():
            if use_max:
                major_scores_df[general_major] = major_scores_df[specific_majors_list].max(axis=1) # using max?
            else:
                major_scores_df[general_major] = major_scores_df[specific_majors_list].mean(axis=1)
            major_scores_df = major_scores_df.drop(specific_majors_list ,axis=1)
    # is 1.0 if is top n for the person, and is 0 if it isn't.
    major_is_top_n_df = major_scores_df.apply(lambda x: x.nlargest(TOP_N, keep='all'), axis = 1).apply(lambda x: x > 0.).astype(np.float32)
    majors = major_is_top_n_df.columns
    return major_is_top_n_df, majors

def _shuffle(train_data_full, train_labels_full):
    nrow = train_data_full.shape[0]
    shuffled_indices = np.arange(nrow)
    np.random.shuffle(shuffled_indices)
    train_data_full_shuffled = train_data_full[shuffled_indices]
    train_labels_full_shuffled = train_labels_full[shuffled_indices]
    return train_data_full_shuffled, train_labels_full_shuffled

def _likert_selector(seq):
    pattern = re.compile(r'l\d[^\r\n]+')
    result = [re.match(pattern, text) is not None for text in seq]
    return result

def _standardize_x(train_data_full, mode='default'):
    nrow, ncol = train_data_full.shape[0], train_data_full.shape[1]
    if mode == "by_column":
        train_data_full -= train_data_full.mean(axis=0).repeat(nrow).reshape((ncol, nrow)).transpose()
        train_data_full /= train_data_full.std(axis=0).repeat(nrow).reshape((ncol, nrow)).transpose()
    elif mode == 'by_row':
        train_data_full -= train_data_full.mean(axis=1).repeat(ncol).reshape((nrow, ncol))
        train_data_full /= train_data_full.std(axis=1).repeat(ncol).reshape((nrow, ncol))
    elif mode == 'none':
        pass
    else:
        train_data_full = (train_data_full - train_data_full.mean())/train_data_full.std()
    return train_data_full

def _subset_x(train_data_full, subset=40):
    ncol = train_data_full.shape[1]
    selected_items = np.random.choice(range(ncol), subset, replace=False)
    selected = [i in selected_items for i in range(ncol)]
    return selected, train_data_full[:, selected]

def _get_full_train(major_all_path, top_n, threshold=.3, generic_only=False, use_max=True):

    raw_data = pd.read_csv(
        major_all_path,
        dtype={
            'age': str,
            'certificate': str,
            'education': str,
            'school': str,
            'gender': str
        }
    )
    major_all = _clean_major_all(raw_data, threshold)
    top_n_df, _ = _get_major_is_top_n_df(top_n, major_all_path, threshold, generic_only=generic_only, use_max=use_max)

    likert_items = major_all.loc[:,_likert_selector(major_all.columns)].astype(np.float32)
    # getting full data and labels
    train_data_full = likert_items.to_numpy() #pretty much X
    train_labels_full = top_n_df.to_numpy()
    return train_data_full, train_labels_full

def _check_gpu_status():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("GPU device name --", tf.config.list_physical_devices('GPU'))
    #return tf.config.list_physical_devices('GPU')[0]

def _build_model_default(majors, loss_weights, input_shape=(99, )):
    # this is the default model
    input_layer = Input(input_shape)
    dense_1 = layers.Dense(72, activation='relu')(input_layer)
    norm_1 = layers.BatchNormalization()(dense_1)
    dense_2 = layers.Dense(64, activation='relu')(norm_1)
    norm_2 = layers.BatchNormalization()(dense_2)
    prediction_layers = [layers.Dense(1, activation='sigmoid', name=major)(norm_2) for major in majors]
    model = Model(input_layer, prediction_layers)
    model.compile(
        optimizer='rmsprop',
        loss = ['binary_crossentropy' for l in range(len(majors))],
        metrics=['accuracy'],
        loss_weights=loss_weights.to_list()
    )
    #print(model.summary())
    return model


def _fit_model(model, train_data, train_labels, val_data, val_labels, early_stopping, epochs, majors):
    history = model.fit(
        train_data,
        [train_labels[:,j] for j in range(len(majors))],
        batch_size=10000,
        epochs=epochs,
        validation_data = (val_data, [val_labels[:,k] for k in range(len(majors))]),
        callbacks = [early_stopping],
        verbose=0
    )
    history_df = pd.DataFrame(history.history)
    return history_df

def _evaluate_model(model, test_data, test_labels, majors):
    return model.evaluate(
                test_data,
                [test_labels[:,i] for i in range(len(majors))],
                verbose=0
            )

def run_multilabel_kfold(
    major_all_path='./data/major_all.csv',
    top_n=3,
    k_fold=5,
    epochs=3,
    subset=30,
    build_model=None,
    threshold=.3,
    generic_only=False,
    use_max=True,
    *args,
    **kwargs
):
    # just checking if GPU is used
    _check_gpu_status()
    #with tf.device('/GPU:0'):  # uses gpu by default
    t_start = datetime.now()
    train_data_full, train_labels_full = _get_full_train(major_all_path, top_n, threshold, generic_only, use_max)
    train_data_full = _standardize_x(train_data_full, mode='by_column')
    if subset is not None:
        selected, train_data_full = _subset_x(train_data_full, subset)
    else:
        selected = None
    train_data_full_shuffled, train_labels_full_shuffled = _shuffle(train_data_full, train_labels_full)

    top_n_df, majors = _get_major_is_top_n_df(top_n, major_all_path, threshold, generic_only, use_max)


    # common settings
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, restore_best_weights=False)
    loss_weights = top_n_df.std()

    # doing k_fold_validation
    nrow, ncol = train_data_full.shape[0], train_data_full.shape[1]
    one_fold = nrow // (k_fold + 1) # reserving one fold for test data
    validation = pd.Series(dtype=np.float32)
    evaluation = pd.Series(dtype=np.float32)
    if build_model is not None:
        model = build_model(majors, loss_weights, ncol, *args, **kwargs)
    else:
        model = _build_model_default(majors, loss_weights, ncol)
    if k_fold > 1:
        for i in range(k_fold):
            print("Fold " + str(i + 1) + " of " + str(k_fold))
            # split X, y
            val_split = list(range(one_fold * i, one_fold * (i + 1)))
            train_split = [i for i in range(one_fold*k_fold) if i not in val_split]

            train_data = train_data_full_shuffled[train_split]
            val_data = train_data_full_shuffled[val_split]
            train_labels = train_labels_full_shuffled[train_split]
            val_labels = train_labels_full_shuffled[val_split]

            # test data is always the last batch
            test_data = train_data_full_shuffled[one_fold*k_fold:]
            test_labels = train_labels_full_shuffled[one_fold*k_fold:]

            fit_params = (model, train_data, train_labels, val_data, val_labels, early_stopping, epochs, majors)
            eval_params = (model, test_data, test_labels, majors)

            if i == 0:
                validation = _fit_model(*fit_params).iloc[-1,:] # this is a pd.Series
                evaluation = pd.Series(_evaluate_model(*eval_params))
            else:
                validation += _fit_model(*fit_params).iloc[-1,:]
                evaluation += pd.Series(_evaluate_model(*eval_params))
        validation /= k_fold
        evaluation /= k_fold #unsupported operation None, int???
    else:
        print("K fold validation disabled. Using default validation split 7000:1000:1300")
        train_data = train_data_full_shuffled[:7000]
        val_data = train_data_full_shuffled[7000:8000]
        train_labels = train_labels_full_shuffled[:7000]
        val_labels = train_labels_full_shuffled[7000:8000]
        test_data = train_data_full_shuffled[8000:]
        test_labels = train_labels_full_shuffled[8000:]

        fit_params = (model, train_data, train_labels, val_data, val_labels, early_stopping, epochs, majors)
        eval_params = (model, test_data, test_labels, majors)

        validation = _fit_model(*fit_params).iloc[-1,:]
        evaluation = pd.Series(_evaluate_model(*eval_params))
    duration = datetime.now() - t_start
    print("k-fold validation done. Execution time -- " + str(duration.total_seconds()))

    if selected is not None:
        return selected, validation, evaluation
    else:
        return validation, evaluation
