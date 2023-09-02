# import torch
from fs.encode1 import ENAC2, binary, NCP, EIIP, CKSNAP, ENAC, Kmer, NAC, PseEIIP, ANF, CKSNAP8, Kmer4, TNC, RCKmer5, \
    DNC
from fs.load_acc import TAC
from fs.load_pse import  SCPseTNC, PCPseTNC

class Config(object):
    def __init__(self):
        self.model_name = ""
        self.is_feature_selection = True # use feature selection or not
        self.load_global_pretrain_model = True # use transfer learning or not

        self.output_save_path = "output/"
        self.global_model_save_path = None
        self.model_save_path = "output/"
        self.stacking_model_save_path = self.output_save_path+"stacking_model/"
        self.base_models_name = None
        self.base_models_name = ["RF", "AB", "LD", "ET", "GB", "XGB", "LGBM", "LR"]
        self.encoding = {"ENAC": ENAC2, "binary": binary, "NCP": NCP, "EIIP": EIIP, "Kmer4": Kmer4, "CKSNAP": CKSNAP8,
                         "PseEIIP": PseEIIP, "TNC": TNC, "RCKmer5": RCKmer5, "SCPseTNC": SCPseTNC, "PCPseTNC": PCPseTNC,
                         "ANF": ANF, "NAC": NAC, "TAC": TAC}
        self.dropout = 0
        self.require_improvement = 1000  # early stopping

        self.k = 1
        self.n_vocab = 0  # size of vocab
        self.num_epochs = 500  # epoch
        self.patience = 50
        self.batch_size = 256  # mini-batch
        self.learning_rate = 0.01
        self.features_save_path = "data/features/"  # the number and type of selected feature
        #-------------------------------------
        self.input_length = 41
        self.num_filters = 80
        self.n_folds = 5


    def __copy__(self):
        tmp = Config("")
        tmp.__dict__.update(self.__dict__)
        return tmp
