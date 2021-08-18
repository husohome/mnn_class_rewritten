
class DATA:
    '''
        This class is meant to function like an enum.
        The keys are important. They represent the slots for different types/ different stages of data.
        The values are not important.
        An example use case is, in the Step class:
            self.data[DATA.FULL] = someFullData(pandas)
    '''
    # the keys are not important
    FULL="full-data"
    TEMP="maybe important to keep a temp data portion, I guess...? maybe use deepcopy?"
    FULL_X="with only the X"
    FULL_Y="full labels with y"
    FULL_TRAIN="x and y train"
    FULL_VALIDATAION="x and y validation"
    FULL_TEST="x and y test portions"
    X_TRAIN="train data, x portion"
    Y_TRAIN="train data, y portion"
    X_VALIDATION="validation data, x portion"
    Y_VALIDATION="validation data, y portion"
    X_TEST="test data, x portion"
    Y_TEST="test data, y portion"
    # maybe write a service class to deal with it?


class MODEL:
    '''
        This class is meant to function like an enum.
        The keys are important. They represent the slots for different types/ different stages of model.
        The values are not important.
        An example use case is, in the Step class:
            self.model[MODEL.PRELIM] = someKerasModel
    '''
    MODEL="where keras model definition is stored (both unfitted or fitted)."
    HYPER_N_LAYERS="Hyperparameter - number of hidden layers"
    HYPER_N_NODES="hyperparameter - number of nodes for each layer"
    HYPER_N_REGULARIZERS="hyperparamter - regularizers used for each layer"



class LOG:
    '''
        This class is meant to function like an enum.
        The keys are important. They represent the slots for different types/ different stages of log.
        The values are not important.
        An example use case is, in the Step class:
            self.log[LOG.PREPROCESSING] = log for all arguments used in the preprocessing step
    '''
    PREPROCESSING="preprocessing"
    MODEL_BUILD="model building"
    MODEL_FIT=""
    CROSS_VALIDATION="crosss validation"


class SETTINGS:
    GENERIC_ONLY=False
    USE_MAX=False
    TOP_N=3
    CLEAN_THRESHOLD=.3
    RAW_DATA_PATH="E:/career_ml/data/major_all_fake.csv"
    INPUT_SHAPE=(99,)
    METRICS=['accuracy']
    SPECIFIC_MAJORS = {
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

    GENERAL_MAJORS= {
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
        "stacs":"Statistics"
    }

    ALL_MAJORS= {
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
    MAJOR_MAPPING = {
            'g_art': ['digia', 'visrt'],
            'g_bns': ['accng', 'busss', 'ecocs', 'marng'],
            'g_eng': ['cheng', 'civng', 'eleng', 'matce', 'mecng'],
            'g_hst': ['hisry', 'arcgy', 'antgy', 'clacs', 'relon'],
            'g_hth': ['heace', 'kingy', 'nurng'],
            'g_lng': ['frech', 'geran', 'itaan', 'spash']
    }