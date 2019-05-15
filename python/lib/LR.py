from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import sys
import os
from os.path import dirname, abspath
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PACKAGE_DIR)
sys.path.insert(0, PACKAGE_DIR)
from sklearn.externals import joblib
from lib.read_conf import Config
from lib.GBDT import GBDT_spr
MODEL_DIR = os.path.join(dirname(dirname(dirname(abspath(__file__)))), 'model')

class LR(object):
    '''
    LR class
    LR模型训练，预测
    '''
    def __init__(self, data_file, mode):
        self._conf = Config()
        self.lr_conf = self._conf.read_model_conf()['lr_conf']
        self._data_file = data_file
        self._mode = mode
        self._gbdt_spr = GBDT_spr(self._data_file)

    def lr_model(self):
        '''
        lr模型训练及预测
        :return: AUC
        '''
        if self._mode == 'train':
            gbdt_features, y_label = self._gbdt_spr.gbdt_model(self._mode)
            grd_lm = LogisticRegression(penalty = self.lr_conf['penalty'],
                                        C = float(self.lr_conf['C']))
            grd_lm.fit(gbdt_features, y_label)
            joblib.dump(grd_lm, os.path.join(MODEL_DIR, "lr_model.m"))

        else:
            gbdt_features, y_label = self._gbdt_spr.gbdt_model(self._mode)
            grd_lm = joblib.load(os.path.join(MODEL_DIR, "lr_model.m"))

            y_pred_grd_lm = grd_lm.predict_proba(gbdt_features)[:, 1]

            fpr_grd_lm, tpr_grd_lm, _ = metrics.roc_curve(y_label, y_pred_grd_lm)
            roc_auc = metrics.auc(fpr_grd_lm, tpr_grd_lm)
            AUC_Score = metrics.roc_auc_score(y_label, y_pred_grd_lm)
            return roc_auc
if __name__ == '__main__':
    train1 = LR('D:\\code\\GBDT_LR\\data\\test.csv',mode = 'pred').lr_model()
    print(train1)


