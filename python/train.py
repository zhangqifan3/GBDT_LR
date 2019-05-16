from lib.LR import LR
from lib.read_conf import Config
def main():
    CONFIG = Config()
    model_conf = CONFIG.read_model_conf()['model_conf']
    if model_conf['mode'] == 'train':
        train1 = LR(model_conf['data_dir_train'],mode = 'train').lr_model()
    else:
        Accuracy, AUC = LR(model_conf['data_dir_pred'],mode = 'pred').lr_model()
        print("LR_Accuracy: %f" % Accuracy)
        print("LR_AUC: %f" % AUC)
if __name__ == '__main__':
    main()