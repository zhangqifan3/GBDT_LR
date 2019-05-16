import sys
import os
from os.path import dirname, abspath
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import CountVectorizer as multivalue
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler as standard
from sklearn.preprocessing import MinMaxScaler as min_max
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer as one_hot
from sklearn.externals import joblib
import numpy as np

MODEL_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'model')
from lib.read_conf import Config
from lib.dataset import DataSet


class GBDT_spr(object):
    '''
    GBDT_spr class
    GBDT模型训练，生成离散特征
    '''
    def __init__(self, data_file):
        self._data_file = data_file
        self._DataSet = DataSet(self._data_file)
        self._conf = Config()
        self.dataset = self._DataSet.input_fn()
        self.batch_dataset = self._DataSet.iter_minibatches()
        self._feature_colums = self._feature_colums()
        self.gbdt_conf = self._conf.read_model_conf()['gbdt_conf']
        self.model_conf = self._conf.read_model_conf()['model_conf']

    def _feature_colums(self):
        '''
        特征列处理
        :return:
            gbdt_colums， type: list
        '''
        gbdt_colums = []
        feature_conf_dic = self._conf.read_feature_conf()[0]
        for feature, conf in feature_conf_dic.items():
            f_type, f_tran = conf["type"], conf["transform"]
            if f_type == 'category':
                if f_tran == 'multivalue':
                    opt = (feature, multivalue())
                    gbdt_colums.append(opt)
                if f_tran == 'one_hot':
                    opt = (feature, one_hot())
                    gbdt_colums.append(opt)

            else:
                opt = ([feature], min_max())
                gbdt_colums.append(opt)
        return gbdt_colums

    def gbdt_model(self, mode):
        '''
        gbdt模型训练，生成离散特征
        :param
            mode: ‘train’ or  ‘pred’
        :return:
            lr_feat：gbdt生成的离散特征
            y：对应数据的label
        '''
        mapper = DataFrameMapper(self._feature_colums, sparse=True)
        if mode == 'train':
            X = mapper.fit_transform(self.dataset)
            y = list(self.dataset['label'])
            grd = GradientBoostingClassifier(n_estimators=int(self.gbdt_conf['n_estimators']),
                                         #    random_state=int(self.gbdt_conf['random_state']),
                                             learning_rate = float(self.gbdt_conf['learning_rate']),
                                         #    subsample=float(self.gbdt_conf['subsample']),
                                             min_samples_leaf = int(self.gbdt_conf['min_samples_leaf']),
                                             max_depth=int(self.gbdt_conf['max_depth']),
                                             max_leaf_nodes = int(self.gbdt_conf['max_leaf_nodes']),
                                             min_samples_split=int(self.gbdt_conf['min_samples_split']))
            if self.model_conf['batch_size'] == '0':
                grd.fit(X, y)
                joblib.dump(grd, os.path.join(MODEL_DIR, "gbdt_model.m"))
                new_feature = grd.apply(X)
                new_feature = new_feature.reshape(-1, int(self.gbdt_conf['n_estimators']))
                enc = OneHotEncoder()
                enc.fit(new_feature)
                lr_feat = np.array(enc.transform(new_feature).toarray())
            else:
                for i, dataset in enumerate(self.batch_dataset):
                #    print(dataset)
                    batch_X = mapper.fit_transform(dataset)
                    batch_y = list(dataset['label'])
                    grd.fit(batch_X, batch_y)
                    new_feature = grd.apply(batch_X)
                    new_feature = new_feature.reshape(-1, int(self.gbdt_conf['n_estimators']))
                    enc = OneHotEncoder()
                    enc.fit(new_feature)
                    new_feature2 = np.array(enc.transform(new_feature).toarray())
                    print(new_feature2)
                    if i == 0:
                        lr_feat = new_feature2
                    else:
                        lr_feat = np.concatenate([lr_feat,new_feature2], axis = 0)
                joblib.dump(grd, os.path.join(MODEL_DIR, "gbdt_model.m"))

        else:
            X = mapper.fit_transform(self.dataset)
            y = list(self.dataset['label'])
            grd = joblib.load(os.path.join(MODEL_DIR, "gbdt_model.m"))
            new_feature = grd.apply(X)
            new_feature = new_feature.reshape(-1, int(self.gbdt_conf['n_estimators']))
            enc = OneHotEncoder()
            enc.fit(new_feature)
            lr_feat = np.array(enc.transform(new_feature).toarray())
        return lr_feat, y

if __name__ == '__main__':
    train_new_feature2  = GBDT_spr('D:\\code\\GBDT_LR\\data\\test.csv').gbdt_model(mode = 'train')
    print(train_new_feature2)