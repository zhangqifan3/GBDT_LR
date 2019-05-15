from lib.LR import LR
from lib.read_conf import Config
def main():
    CONFIG = Config()
    model_conf = CONFIG.read_model_conf()['model_conf']
    if model_conf['mode'] == 'train':
        train1 = LR(model_conf['data_dir_train'],mode = 'train').lr_model()
    else:
        pred1 = LR(model_conf['data_dir_pred'],mode = 'pred').lr_model()

        print(pred1)
if __name__ == '__main__':
    main()