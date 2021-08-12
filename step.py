from abc import ABC, abstractmethod, abstractstaticmethod
from constant import DATA, LOG, SETTINGS
import pandas as pd
import numpy as np
import re
import sys


class Step(ABC):

    def __init__(self, data={}, model={}, results={}, log={}):
        self.data = data
        self.model = model
        self.results = results
        self.log = log

    @abstractmethod
    def _log(self):
        pass

    @abstractmethod
    def run(self):
        # define what to do when run
        pass
        # doesn't always need to return self
        # remember to return self?

class Preprocessing(Step):

    def _get_raw_data(self, raw_data_path=SETTINGS.RAW_DATA_PATH):

        self.raw_data_path = raw_data_path
        self.data[DATA.TEMP] = pd.read_csv(
            self.raw_data_path, 
            dtype={
                'age': str,
                'certificate': str,
                'education': str,
                'school': str,
                'gender': str
            })

        return self

    def _get_likert_items(self):
        likert_items = []
        pattern = re.compile(r"l\d[^\r\n]+\d")
        for col in self.data[DATA.TEMP].columns:
            if re.match(pattern, col) is not None:
                likert_items.append(col)
        return likert_items

        
    def _clean_missing(self, clean_threshold=SETTINGS.CLEAN_THRESHOLD):
        # input
            # major_FULL = pd.DataFrame(major_FULL_path)
            # clean_threshold is largest proportion of missing cases tolerated
        # process
            # remove rows that have too much missing (Likert only)
            # remove rows that have demographics missing
        # output
            # cleaned major_FULL_df
        self.__clean_threshold = clean_threshold
        likert_items = self._get_likert_items()
        # if missing data exceeds a proportion, the row is deleted
        ncol = self.data[DATA.TEMP][likert_items].shape[1]
        passed = self.data[DATA.TEMP][likert_items].isna().sum(axis=1) <= ncol*clean_threshold
        # also has to not miss any demographics
        # somehow missing cases have been coded as "null_unknonw" (string, not NA)
        passed = passed & (self.data[DATA.TEMP][['age', 'certificate', 'education', 'school', 'gender']] == 'null_unknown').sum(axis=1) == 0
        cleaned_df = self.data[DATA.TEMP].loc[passed, :].fillna(0.).reset_index(drop=True)
        self.data[DATA.FULL_X] = cleaned_df.loc[:, likert_items]
        self.data[DATA.TEMP] = cleaned_df
        if not hasattr(self, "__detail"):
            self.__detail = ""
        self.__detail += "cleaned likertype items stored in data." + DATA.FULL_X
        return self

    def _get_major_is_top_n_df(
            self,
            top_n=SETTINGS.TOP_N,
            generic_only=SETTINGS.GENERIC_ONLY,
            use_max=SETTINGS.USE_MAX # use Max when lumping 50 majors to 33, else uses mean
        ):

        def ipsative_check(major, col):
            m = re.match("f[4-7]"+major+"\d", col)
            return m is not None

        self.__top_n = top_n
        self.__generic_only = generic_only
        self.__use_max = use_max

        d = self.data[DATA.TEMP]
        major_scores_df = pd.DataFrame()
       
        for major in SETTINGS.SPECIFIC_MAJORS.keys():
            # oops forgot to enforce forced choice
            cols_for_the_major = [col for col in d.columns if ipsative_check(major, col)]
            major_frame = d.loc[:,cols_for_the_major]
            major_scores_df[major] = major_frame.sum(axis=1)

        # returning only 33 more generic majors
        if generic_only:
            # mapping specific majors to general majors
            for general_major, specific_majors_list in SETTINGS.MAJOR_MAPPING.items():
                if use_max:
                    major_scores_df[general_major] = major_scores_df[specific_majors_list].max(axis=1) # using max?
                else:
                    major_scores_df[general_major] = major_scores_df[specific_majors_list].mean(axis=1)
                major_scores_df = major_scores_df.drop(specific_majors_list ,axis=1)
        # is 1.0 if is top n for the person, and is 0 if it isn't.
        major_is_top_n_df = major_scores_df.apply(lambda x: x.nlargest(top_n, keep='all'), axis = 1).apply(lambda x: x > 0.).astype(np.float32)
        self.data[DATA.FULL_Y] = major_is_top_n_df
        if not hasattr(self, "__detail"):
            self.__detail = ""
        self.__detail += "major is top n df stored in data." + DATA.FULL_Y
        return self

    def _log(self):
        if not hasattr(self.log, LOG.PREPROCESSING):
            self.log[LOG.PREPROCESSING] = []

        self.log[LOG.PREPROCESSING].append(
            {
                "raw_data_path": self.raw_data_path,
                "clean_threshold": self.__clean_threshold,
                "top_n": self.__top_n,
                "generic_only": self.__generic_only,
                "use_max": self.__use_max,
                "other": self.__detail
            }
        )

        return self

    def run(
            self,
            raw_data_path=SETTINGS.RAW_DATA_PATH,
            clean_threshold=SETTINGS.CLEAN_THRESHOLD,
            top_n=SETTINGS.TOP_N,
            generic_only=SETTINGS.GENERIC_ONLY,
            use_max=SETTINGS.USE_MAX
        ):
        self._get_raw_data(raw_data_path)._clean_missing(clean_threshold)._get_major_is_top_n_df(top_n, generic_only, use_max)._log()


class BuildModelDefault(Step):

    def __init__(self):
        pass


class Analysis:

    def __init__(self, steps, data, model, results, log):

        self.data = data
        self.model = model
        self.results = results
        self.steps = steps # preferablly Step objects.
        self.log = log
        return None

    def add(self, step):
        if not isinstance(step, Step):
            raise TypeError("steo must be of type Step") 
        if hasattr(self, 'steps'):
            self.steps.append(step(self.data, self.model, self.results))
        else:
            self.steps = step(self.data, self.model, self.results)

    def run(self):
        for s in self.steps:
            s.run()

        # step, step, step, ... step... final results.
        # every step: 
        # uses some data, might use models, might or might not output data to pass to the next step
        # data may have different shapes and formats
        # maybe predefine the shapes and format?
        # requires what?
        # might not need to resuse data so...
        # start
        # process
        # end_pass # passing whatever to the next step. 
        # actuFULLy don't need. You just alter the reference
        # log?


if __name__ == "__main__":
    sys.path.append("E:/career_ml/")

    preprocessing = Preprocessing({},{},{},{})
    preprocessing.run()
    print(preprocessing.log)
    print(preprocessing.data)






    