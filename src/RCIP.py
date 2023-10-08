import numpy as np
import pandas as pd
import datetime
import time
import itertools
import os
import random
# PyTorch 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Sckit-Learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split, KFold
# My Modules
from src.utils import *
from src.vis import *
from src.bio import *
from src.chem import *
from src.models import *

# 
def fix_seed(seed):
    """ Fix random seed
    input
    --------------------
    seed (int)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# 
def organize_input(dataset_in, fp_size, batch_size, flag=None):
    """ Create data loader from raw input dataframe
    input
    --------------------
    dataset_in (dataframe) : the columns named as following are necessary.
        - id : to identify each interaction pair.
        - RNA_id (optional) : to obtain information about RNA from RNAInformation.csv
        - Sequence : RNA sequence are stored with AUGC string.
        - MOL_id (optional) : to obtain information about compound from MolInformation.csv
        - SMILES : Compound structure are stored with canonical SMILES string.
        - Label : 1 means binding, and 0 non-binding. 
        - Ext_id (optional) : to obtain information about interaction from outer database (e.g. RNAInter)
    fp_size (int) : the number of ECFP bits
    batch_size (int)
    flag (int) : if it has any number k, the k-th bit value of ECFP will shuffle (for permutation importance).

    output
    --------------------
    Dataloader (tensor)
    """
    dataset = dataset_in.copy()

    # RNA input
    RNA_seqs = dataset['Sequence'].values.tolist()
    RNA_input = [seq2onehot(seq) for seq in RNA_seqs]
    X1_array = np.array([[RNA] for RNA in RNA_input])
    X1 = torch.from_numpy(X1_array).float()

    # Commpound input
    SMILES = dataset['SMILES'].values.tolist()
    mols = [Chem.MolFromSmiles(smi) for smi in SMILES]
    Chem_input = [AllChem.GetMorganFingerprintAsBitVect(mol,2,fp_size) for mol in mols]
    # Shuffle k-th bit of ECFP ========================================================
    if flag is not None:
        Chem_input_T = list(map(list, (zip(*Chem_input))))
        Chem_input_T[flag] = random.sample(Chem_input_T[flag], len(Chem_input))
        Chem_input = list(map(list, (zip(*Chem_input_T))))
    # =================================================================================
    X2_array = np.array([[chem] for chem in Chem_input])
    X2 = torch.from_numpy(X2_array).float()

    # Label
    y = dataset['Label'].values
    y = torch.from_numpy(y).long()

    # Index
    ids = dataset['id'].values.tolist()
    index = [int(id[1:]) for id in ids]
    index = torch.from_numpy(np.array(index)).long()

    # Data loader
    Dataset = torch.utils.data.TensorDataset(X1, X2, y, index)
    Dataloader = torch.utils.data.DataLoader(dataset=Dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    return Dataloader

# 
def draw_learning_curve(data_path, result_save_path='train/train_result.txt', figsize=(8,9)):
    """ Draw learning curve from train & valid results 
    input
    --------------------
    data_path (str)
    result_save_path (str)
    fig_size (turple)
    """
    os.mkdir(data_path+'train/images')
    train_result = pd.read_csv(data_path+'train/train_result.txt')

    X = train_result['epoch'].astype('int').values.tolist()
    Index = ['Loss','Acc','Rec','Pre','F1','AUROC','AUPRC']
    Color = ['firebrick']+['darkolivegreen']*6
    Title = ['Loss','Accuracy','Recall','Precision','F1-score','AUROC','AUPRC']
    Ylim = [None]+[[0,1.01]]*6

    for index,color,title,ylim in zip(Index,Color,Title,Ylim):
        Y = train_result.loc[:,index].astype('float').values.tolist()
        Y_std = train_result[index+'[std]'].astype('float').values.tolist()

        fig,ax = plt.subplots(figsize=figsize)
        my_plot(X, Y, Y_std, ax, color=color, label=None, title=title, xlabel='epoch', ylabel=title, show_label=False, ylim=ylim)
        fig.savefig(data_path+'train/images/'+title+'.png', dpi=300)

# 
class RCIP():
    """ Model of RNA-Compound Interaction Predictor
    input
    --------------------
    model_name (str) : if you construct models in src/models.py, you can choose model using for analysis.
    result_save_path (str)
    device (str) : {cuda, cpu}
    c_input (int) : the number of ECFP bits
    r_input (int) : the size of One-hot (default value is 4)
    batch_size (int)
    num_epoch_cv (int)
    num_epoch (int)
    """

    #
    def __init__(self, model_name, result_save_path, device, c_input, r_input, batch_size, num_epoch_cv, num_epoch):
        self.model_name = model_name
        self.result_save_path = result_save_path
        self.device = device
        self.c_input = c_input
        self.r_input = r_input
        self.batch_size = batch_size
        self.num_epoch_cv = num_epoch_cv
        self.num_epoch = num_epoch
        print('>> New model was successfully generated.')

    #
    def load_model(self, dropout_rate, l1_alpha, kernel_size, num_kernel, batch_size=None, trained_weight_path=None):
        """ 
        input
        --------------------
        dropout_rate (float)
        l1_alpha (float) : lambda of L1 regularization
        kernel_size (int)
        num_kernel (int)
        batch_size (int)
        trained_weight_path (str)
        """
        if batch_size is not None: self.batch_size = batch_size
        self.model = eval(self.model_name)(self.c_input, self.batch_size, dropout_rate, (self.r_input, kernel_size), num_kernel).to(self.device)
        if trained_weight_path is not None: self.set_trained_weight(trained_weight_path)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.set_parameters(dropout_rate, l1_alpha, kernel_size, num_kernel)

    # 
    def set_parameters(self, dropout_rate, l1_alpha, kernel_size, num_kernel):
        """ 
        input
        --------------------
        dropout_rate (float)
        l1_alpha (float) : lambda of L1 regularization
        kernel_size (int)
        num_kernel (int)
        """
        self.dropout_rate = dropout_rate
        self.l1_alpha = l1_alpha
        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.params_dict = {'dropout_rate':dropout_rate, 'l1_alpha':l1_alpha, 'kernel_size':(self.r_input, kernel_size), 'num_kernel':num_kernel}

    #
    def set_trained_weight(self, trained_weight_path):
        """ Set trained weight values to model
        input
        --------------------
        trained_weight_path (str)
        """
        self.model.load_state_dict(torch.load(trained_weight_path))
        print('>> Trained weight was loaded to model.')

    #
    def save_trained_weight(self, model_save_filename):
        """ Save trained weight values
        input
        --------------------
        model_save_filename (str)
        """
        torch.save(self.model.state_dict(), self.result_save_path+model_save_filename)
        print('>> Trained weight was saved.')

    #
    def load_dataset(self, dataset_in, testset_in=None, method='random', test_size=0.1, column=None, order_list=None):
        """ Load dataset to model
        input
        --------------------
        dataset_in (dataframe) : the columns named as following are necessary.
            - id : to identify each interaction pair.
            - RNA_id (optional) : to obtain information about RNA from RNAInformation.csv
            - Sequence : RNA sequence are stored with AUGC string.
            - MOL_id (optional) : to obtain information about compound from MolInformation.csv
            - SMILES : Compound structure are stored with canonical SMILES string.
            - Label : 1 means binding, and 0 non-binding. 
            - Ext_id (optional) : to obtain information about interaction from outer database (e.g. RNAInter)
        testset_in (dataframe) : set it if you would like to use your own test dataset.
        method (str) : you can select train-test split method from {random, order}.
        test_size (float) : for "random" method
        column (str), order_list (list) : for "order" method 
        """
        if testset_in is not None:
            dataset_train = dataset_in.copy()
            dataset_test = testset_in.copy()
            self.dataset_train = dataset_train
            self.dataset_test = dataset_test
            print('>> User-supplied trainset & testset were loaded.')
        
        else:
            dataset = dataset_in.copy()
            if method=='random':
                self.dataset_train, self.dataset_test = train_test_split(dataset, test_size=test_size, stratify=dataset['Label'], random_state=1234)
                print('>> Whole dataset was split randomly.')
            elif method=='order':
                self.dataset_train = dataset[~dataset[column].isin(order_list)]
                self.dataset_test = dataset[dataset[column].isin(order_list)]
                print('>> Whole dataset was split following to user-supplied test data list.')
            
        print('    Train Dataset Size:   {:>5} (p-{:>5}, n-{:>5})'.format(len(self.dataset_train), len(self.dataset_train[self.dataset_train['Label']==1]), len(self.dataset_train[self.dataset_train['Label']==0])))
        print('    Test  Dataset Size:   {:>5} (p-{:>5}, n-{:>5})'.format(len(self.dataset_test), len(self.dataset_test[self.dataset_test['Label']==1]), len(self.dataset_test[self.dataset_test['Label']==0])))

    # 
    def cross_validation(self, params_dict, cv_fold=5, log_filename='cv/cv_log.txt', result_filename1='cv/cv_train_result.txt', result_filename2='cv/cv_valid_result.txt'):
        """ Run cross validation
        input
        --------------------
        params_dict (dictionary) : the dictionary of parameters, which has hyper parameter name (str) as "key" and its value (list) as "value".
        cv_fold (int)
        log_filename (str) : the basic information of performance will be saved in log file.
        result_filename1 (str) : the evaluation values of train will be saved in result file.
        result_filename2 (str) : the evaluation values of validation will be saved in result file.
        """
        with open(self.result_save_path+log_filename, 'w') as f1:
            print('>> Cross-validation started.')
            f1.write('{} : cross validation started.\n'.format(datetime.datetime.now()))
            f1.write('\n')
            f1.write('   ...processing...   \n')
            f1.write('\n')
            start_clock = time.time()

            # create a dictionary of all combination of parameter values
            param_names = params_dict.keys()
            param_vals = params_dict.values()
            param_comb = list(itertools.product(*param_vals))

            with open(self.result_save_path+result_filename1, 'w') as f2, open(self.result_save_path+result_filename2, 'w') as f3:

                best_score = 0
                best_scores = []

                for i,params in enumerate(param_comb):
                    print('   *Parameter Combination {}/{}'.format(i+1, len(param_comb)))
                    f2.write('> Parameter Combination {}/{}: '.format(i+1, len(param_comb)))
                    f3.write('> Parameter Combination {}/{}: '.format(i+1, len(param_comb)))
                    for k,v in dict(zip(param_names, params)).items(): print('    [{}={}]'.format(k,v), end=''); f2.write('[{}={}]'.format(k,v)); f3.write('[{}={}]'.format(k,v))
                    print(''); f2.write('\n'); f3.write('\n')
                    f2.write('epoch,Loss,Loss[std],Acc,Acc[std],Rec,Rec[std],Pre,Pre[std],F1,F1[std],AUROC,AUROC[std],AUPRC,AUPRC[std]\n')
                    f3.write('epoch,Loss,Loss[std],Acc,Acc[std],Rec,Rec[std],Pre,Pre[std],F1,F1[std],AUROC,AUROC[std],AUPRC,AUPRC[std]\n')

                    # list for save results
                    loss_train_list, acc_train_list, rec_train_list, pre_train_list, f1_train_list, auroc_train_list, auprc_train_list = [[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)]
                    loss_valid_list, acc_valid_list, rec_valid_list, pre_valid_list, f1_valid_list, auroc_valid_list, auprc_valid_list = [[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)],[[] for i in range(self.num_epoch_cv)]

                    # split data into k-fold
                    kf = KFold(n_splits=cv_fold, shuffle=True)

                    for i,(train_index,valid_index) in enumerate(kf.split(self.dataset_train)):
                        # create data loader
                        # train data ------------------------------------------------------------
                        dataset_train = self.dataset_train.iloc[train_index].reset_index()
                        self.train_loader = organize_input(dataset_train, fp_size=self.c_input, batch_size=self.batch_size)
                        # valid data ------------------------------------------------------------
                        dataset_valid = self.dataset_train.iloc[valid_index].reset_index()
                        self.valid_loader = organize_input(dataset_valid, fp_size=self.c_input, batch_size=self.batch_size)
                        # training & validation
                        start_time  = time.time()
                        print('       CV{}[%] ['.format(i+1), end='')
                        bar_interval = self.num_epoch_cv // 10

                        self.load_model(**dict(zip(param_names, params)))

                        for epoch in range(self.num_epoch_cv):
                            train_loss, _, train_acc, _  = self._train()
                            valid_loss, _, valid_acc, _  = self._valid()
                            # save -------------------------------------------------------
                            loss_train_list[epoch].append(train_loss);   loss_valid_list[epoch].append(valid_loss)
                            acc_train_list[epoch].append(train_acc[0]);  acc_valid_list[epoch].append(valid_acc[0])
                            rec_train_list[epoch].append(train_acc[1]);  rec_valid_list[epoch].append(valid_acc[1])
                            pre_train_list[epoch].append(train_acc[2]);  pre_valid_list[epoch].append(valid_acc[2])
                            f1_train_list[epoch].append(train_acc[3]);   f1_valid_list[epoch].append(valid_acc[3])
                            auroc_train_list[epoch].append(train_acc[4]);auroc_valid_list[epoch].append(valid_acc[4])
                            auprc_train_list[epoch].append(train_acc[5]);auprc_valid_list[epoch].append(valid_acc[5])
                            if (epoch+1)%bar_interval==0: print('|', end='')

                        process_time = time.time() - start_time
                        print('] {} ({:.3f} s/epoch)'.format(second2date(process_time), process_time/self.num_epoch_cv))

                    print('   -------------------------------------------------------------------')

                    for epoch in range(self.num_epoch_cv):
                        f2.write('{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(epoch+1, np.mean(loss_train_list[epoch]), np.std(loss_train_list[epoch]), np.mean(acc_train_list[epoch]), np.std(acc_train_list[epoch]),
                                                                                                        np.mean(rec_train_list[epoch]), np.std(rec_train_list[epoch]), np.mean(pre_train_list[epoch]), np.std(pre_train_list[epoch]),
                                                                                                        np.mean(f1_train_list[epoch]), np.std(f1_train_list[epoch]), np.mean(auroc_train_list[epoch]), np.std(auroc_train_list[epoch]),
                                                                                                        np.mean(auprc_train_list[epoch]), np.std(auprc_train_list[epoch])))
                        f3.write('{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(epoch+1, np.mean(loss_valid_list[epoch]), np.std(loss_valid_list[epoch]), np.mean(acc_valid_list[epoch]), np.std(acc_valid_list[epoch]),
                                                                                                        np.mean(rec_valid_list[epoch]), np.std(rec_valid_list[epoch]), np.mean(pre_valid_list[epoch]), np.std(pre_valid_list[epoch]),
                                                                                                        np.mean(f1_valid_list[epoch]), np.std(f1_valid_list[epoch]), np.mean(auroc_valid_list[epoch]), np.std(auroc_valid_list[epoch]),
                                                                                                        np.mean(auprc_valid_list[epoch]), np.std(auprc_valid_list[epoch])))
                    # store the best model parameters according to AUPRC.
                    score = np.mean(auprc_valid_list[-1])
                    if score > best_score:
                        best_score = score
                        best_scores = [np.mean(loss_valid_list[-1]), np.std(loss_valid_list[-1]), np.mean(acc_valid_list[-1]), np.std(acc_valid_list[-1]),
                                       np.mean(rec_valid_list[-1]), np.std(rec_valid_list[-1]), np.mean(pre_valid_list[-1]), np.std(pre_valid_list[-1]),
                                       np.mean(f1_valid_list[-1]), np.std(f1_valid_list[-1]), np.mean(auroc_valid_list[-1]), np.std(auroc_valid_list[-1]),
                                       np.mean(auprc_valid_list[-1]), np.std(auprc_valid_list[-1])]
                        best_params = dict(zip(param_names, params))

            print('   **best parameters are: ', end='')
            for k,v in best_params.items(): print('   [ {}: {} ] '.format(k,v), end='')
            f1.write('{} : cross validation ended.\n'.format(datetime.datetime.now()))
            f1.write('\n')
            f1.write('=== Cross Validation Report ===\n')
            f1.write('  This cross validation processed with parameters range: \n')
            for k,v in self.params_dict.items(): f1.write('    {}: {}\n'.format(k,v))
            f1.write('\n')
            f1.write('  * According to {}-fold cross validation, the best parameters are: \n'.format(cv_fold))
            for k,v in best_params.items(): f1.write('    {}: {}\n'.format(k,v))
            f1.write('\n')
            f1.write('  * With these parameters, model achieved performance below:\n')
            f1.write('   | Loss     : {:.6f}+/-{:.6f}\n'.format(best_scores[0], best_scores[1]))
            f1.write('   | Accuracy : {:.6f}+/-{:.6f}\n'.format(best_scores[2], best_scores[3]))
            f1.write('   | Recall   : {:.6f}+/-{:.6f}\n'.format(best_scores[4], best_scores[5]))
            f1.write('   | Precision: {:.6f}+/-{:.6f}\n'.format(best_scores[6], best_scores[7]))
            f1.write('   | F1-score : {:.6f}+/-{:.6f}\n'.format(best_scores[8], best_scores[9]))
            f1.write('   | AUROC    : {:.6f}+/-{:.6f}\n'.format(best_scores[10], best_scores[11]))
            f1.write('   | AUPRC    : {:.6f}+/-{:.6f}\n'.format(best_scores[12], best_scores[13]))
            f1.write('\n')
            f1.write('  * Model train performances are saved at [{}]\n'.format(self.result_save_path+result_filename1))
            f1.write('  * Model validation performances are saved at [{}]\n'.format(self.result_save_path+result_filename2))
            f1.write('  Totally [{}] has passed on this process.\n'.format(second2date(time.time()-start_clock)))
            self.set_parameters(**best_params)


    # 
    def train(self, log_filename='train/train_log.txt', result_filename='train/train_result.txt', param_filename='train/model_params.pth'):
        """
        input
        --------------------
        log_filename (str) : the basic information of performance will be saved in log file.
        result_filename (str) : the evaluation values will be saved in result file.
        param_filename (str) : the trained model weights will be saved.
        """
        with open(self.result_save_path+log_filename, 'w') as f1:
            f1.write('{} : model train started.\n'.format(datetime.datetime.now()))
            f1.write('\n')
            f1.write('   ...processing...   \n')
            f1.write('\n')
            start_clock = time.time()

            self.load_model(self.dropout_rate, self.l1_alpha,self.kernel_size, self.num_kernel)

            # create data loader
            self.train_loader = organize_input(self.dataset_train, fp_size=self.c_input, batch_size=self.batch_size)

            with open(self.result_save_path+result_filename, 'w') as f2:
                f2.write('epoch,Loss,Loss[std],Acc,Acc[std],Rec,Rec[std],Pre,Pre[std],F1,F1[std],AUROC,AUROC[std],AUPRC,AUPRC[std]\n')

                start_time  = time.time()
                print('>> Train started correctly.')
                print('   Progress Bar[%]: [', end='')
                bar_interval = self.num_epoch // 10

                for epoch in range(self.num_epoch):
                    train_loss, train_loss_std, train_acc, train_acc_std  = self._train()
                    f2.write('{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(epoch+1, train_loss, train_loss_std, train_acc[0], train_acc_std[0],
                                                                                                   train_acc[1], train_acc_std[1], train_acc[2], train_acc_std[2],
                                                                                                   train_acc[3], train_acc_std[3], train_acc[4], train_acc_std[4],
                                                                                                   train_acc[5], train_acc_std[5]))

                    if (epoch+1)%bar_interval==0: print('|', end='')

            process_time = time.time() - start_time
            print('] {} ({:.3f} s/epoch)'.format(second2date(process_time), process_time/self.num_epoch))

            if param_filename is not None: self.save_trained_weight(param_filename)

            f1.write('{} : model train ended.\n'.format(datetime.datetime.now()))
            f1.write('\n')
            f1.write('=== Train Report ===\n')
            f1.write('  This train processed with parameters: ')
            for k,v in self.params_dict.items(): f1.write('[{}: {}]'.format(k,v))
            f1.write('\n')
            f1.write('   | Loss     : {:.6f}+/-{:.6f}\n'.format(train_loss, train_loss_std))
            f1.write('   | Accuracy : {:.6f}+/-{:.6f}\n'.format(train_acc[0], train_acc_std[0]))
            f1.write('   | Recall   : {:.6f}+/-{:.6f}\n'.format(train_acc[1], train_acc_std[1]))
            f1.write('   | Precision: {:.6f}+/-{:.6f}\n'.format(train_acc[2], train_acc_std[2]))
            f1.write('   | F1-score : {:.6f}+/-{:.6f}\n'.format(train_acc[3], train_acc_std[3]))
            f1.write('   | AUROC    : {:.6f}+/-{:.6f}\n'.format(train_acc[4], train_acc_std[4]))
            f1.write('   | AUPRC    : {:.6f}+/-{:.6f}\n'.format(train_acc[5], train_acc_std[5]))
            f1.write('  * Model performances are saved at [{}]\n'.format(self.result_save_path+result_filename))
            f1.write('  * Trained Model parameters are saved at [{}]\n'.format(self.result_save_path+param_filename))
            f1.write('  Totally [{}] has passed on this process.\n'.format(second2date(time.time()-start_clock)))


    # 
    def test(self, param_filename='train/model_params.pth', log_filename='test/test_log.txt', result_filename='test/test_result.txt'):
        """
        input
        --------------------
        param_filename (str) : the trained model weights
        log_filename (str) : the basic information of performance will be saved in log file.
        result_filename (str) : the evaluation values will be saved in result file.
        """
        with open(self.result_save_path+log_filename, 'w') as f1:
            print('>> Test started.')
            f1.write('{} : model test performed.\n'.format(datetime.datetime.now()))
            f1.write('\n')

            self.load_model(self.dropout_rate, self.l1_alpha, self.kernel_size, self.num_kernel, batch_size=len(self.dataset_test), trained_weight_path=self.result_save_path+param_filename)

            # create data loader
            self.test_loader = organize_input(self.dataset_test, fp_size=self.c_input, batch_size=self.batch_size)

            with open(self.result_save_path+result_filename, 'w') as f2:
                f2.write('id,predict,label\n')

                index, pred_list, true_label  = self._test()
                for id, pred, true in zip(index, pred_list, true_label): f2.write('{},{:.6f},{}\n'.format(id, pred, true))

            # 
            f1.write('{} : model test ended.\n'.format(datetime.datetime.now()))
            f1.write('\n')
            f1.write('=== Test Report ===\n')
            f1.write('  This test processed with parameters: ')
            for k,v in self.params_dict.items(): f1.write('[{}: {}]'.format(k,v))
            f1.write('\n')
            # confusion matrix
            pred_label = [1 if pred > .5 else 0 for pred in pred_list]
            TN, FP, FN, TP = confusion_matrix(true_label, pred_label).flatten()
            f1.write('  * Confusion Matrix\n')
            f1.write('                    | Pred Positive | Pred Negative |\n')
            f1.write('    | True Positive |         {:>5} |         {:>5} |\n'.format(TP, FN))
            f1.write('    | True Negative |         {:>5} |         {:>5} |\n'.format(FP, TN))
            f1.write('\n')
            # evaluation
            acc = (TP+TN)/(TP+FP+FN+TN)
            rec = TP/(TP+FN)
            pre = TP/(TP+FP)
            f1s = 2*rec*pre/(rec+pre)
            fpr, tpr, th = roc_curve(true_label, pred_list)
            pre_, rec_, th = precision_recall_curve(true_label, pred_list)
            AUROC = auc(fpr, tpr)
            AUPRC = auc(rec_, pre_)
            MCC = (TP*TN - FP*FN) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
            f1.write('  * Model Performance\n')
            f1.write('   | Accuracy : {:.4f}\n'.format(acc))
            f1.write('   | Recall   : {:.4f}\n'.format(rec))
            f1.write('   | Precision: {:.4f}\n'.format(pre))
            f1.write('   | F1-score : {:.4f}\n'.format(f1s))
            f1.write('   | AUROC    : {:.4f}\n'.format(AUROC))
            f1.write('   | AUPRC    : {:.4f}\n'.format(AUPRC))
            f1.write('   | MCC      : {:.4f}\n'.format(MCC))
            f1.write('  * Model test performances are saved at [{}]\n'.format(self.result_save_path+result_filename))


    # 
    def permutation_importance(self, param_filename='train/model_params.pth', log_filename='permutation_importance/top_bits.txt', result_filename='permutation_importance/importances.txt'):
        """
        input
        --------------------
        param_filename (str) :  : the trained model weights
        log_filename (str) : the information of top 20 important bits will be saved in log file.
        result_filename (str) : the information of all bits will be saved in result file.
        """
        with open(self.result_save_path+log_filename, 'w') as f1:
            print('>> Permutation imoortance started.')
            f1.write('{} : model permutation importance started.\n'.format(datetime.datetime.now()))
            f1.write('\n')
            f1.write('   ...processing...   \n')
            f1.write('\n')
            start_clock = time.time()

            self.load_model(self.dropout_rate, self.l1_alpha, self.kernel_size, self.num_kernel, trained_weight_path=self.result_save_path+param_filename)

            importances = [[0]*self.c_input for _ in range(6)]

            # calculate base accuracy
            base_loader = organize_input(self.dataset_test, fp_size=self.c_input, batch_size=self.batch_size)
            _, _, base_acc, _  = self._valid(base_loader)

            # shuffle data iteration
            with open(self.result_save_path+result_filename, 'w') as f2:
                f2.write('Accuracy,Recall,Precision,F1,AUROC,AUPRC\n')

                for j in range(self.c_input):
                    valid_loader_pi = organize_input(self.dataset_test, fp_size=self.c_input, batch_size=self.batch_size, flag=j)
                    _, _, valid_acc_pi, _  = self._valid(valid_loader_pi)
                    importances[0][j] = (base_acc[0] - valid_acc_pi[0])
                    importances[1][j] = (base_acc[1] - valid_acc_pi[1])
                    importances[2][j] = (base_acc[2] - valid_acc_pi[2])
                    importances[3][j] = (base_acc[3] - valid_acc_pi[3])
                    importances[4][j] = (base_acc[4] - valid_acc_pi[4])
                    importances[5][j] = (base_acc[5] - valid_acc_pi[5])
                    f2.write('{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(importances[0][j], importances[1][j], importances[2][j], importances[3][j], importances[4][j], importances[5][j]))
                    if j%20==0: print(f'   {j} shuffle pattern done.')

            f1.write('{} : permutation importance ended.\n'.format(datetime.datetime.now()))
            f1.write('\n')
            f1.write('=== PI Report [ top 10 bit position ] ===\n')
            f1.write('\n')
            f1.write('   |    Accuracy    |     Recall     |   Precision    |    F1-score    |     AUROC      |     AUPRC      |\n')
            for j in range(10):
                bit_acc = np.max(importances[0]); bit_acc_idx = np.argmax(importances[0])
                bit_rec = np.max(importances[1]); bit_rec_idx = np.argmax(importances[1])
                bit_pre = np.max(importances[2]); bit_pre_idx = np.argmax(importances[2])
                bit_f1  = np.max(importances[3]); bit_f1_idx  = np.argmax(importances[3])
                bit_roc = np.max(importances[4]); bit_roc_idx = np.argmax(importances[4])
                bit_prc = np.max(importances[5]); bit_prc_idx = np.argmax(importances[5])
                f1.write(' {:>2}| {:>4}({:.6f}) | {:>4}({:.6f}) | {:>4}({:.6f}) | {:>4}({:.6f}) | {:>4}({:.6f}) | {:>4}({:.6f}) |\n'.format(j+1, bit_acc_idx, bit_acc, bit_rec_idx, bit_rec, bit_pre_idx, bit_pre,
                                                                                                                                 bit_f1_idx, bit_f1, bit_roc_idx, bit_roc, bit_prc_idx, bit_prc))
                importances[0][bit_acc_idx] = np.min(importances[0])
                importances[1][bit_rec_idx] = np.min(importances[1])
                importances[2][bit_pre_idx] = np.min(importances[2])
                importances[3][bit_f1_idx]  = np.min(importances[3])
                importances[4][bit_roc_idx] = np.min(importances[4])
                importances[5][bit_prc_idx] = np.min(importances[5])
            f1.write('\n')
            f1.write('  Totally [{}] has passed on this process.\n'.format(second2date(time.time()-start_clock)))


    #
    def _train(self):
        """
        output
        --------------------
        train_loss (float) : the mean of loss
        train_loss_std (float) : the standard deviation of loss
        train_acc (float) : the mean of accuracy
        train_acc_std (float) : the standard deviation of accuracy
        """
        train_loss, train_loss_std = 0.0, 0.0
        train_acc, train_acc_std   = [], []
        temp_loss_list = []
        temp_acc_list, temp_rec_list, temp_pre_list, temp_F1_list = [], [], [], []
        temp_roc_list, temp_prc_list = [], []

        self.model.train()
        for i,(X1, X2, y, index) in enumerate(self.train_loader):
            X1, X2, y = X1.to(self.device), X2.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X1, X2)
            loss = self.criterion(outputs, y)
            # L1 regulation ------------------------------
            l1 = torch.tensor(0., requires_grad=True)
            for w in self.model.parameters(): l1 = l1 + torch.norm(w, 1)
            loss = loss + self.l1_alpha * l1
            #---------------------------------------------
            loss.backward()
            self.optimizer.step()

            pred = torch.transpose(outputs, 1, 0)
            pred_list = pred[1].detach().cpu().numpy().tolist()
            pred_binary = [1 if p>0.5 else 0 for p in pred_list]
            true_list = y.detach().cpu().numpy().tolist()
            if np.sum(true_list)>0:
                # ---------------------------------------------------------------
                temp_acc_list.append(accuracy_score(true_list, pred_binary))  # Accuracy
                temp_rec_list.append(recall_score(true_list, pred_binary))    # Recall
                temp_pre_list.append(precision_score(true_list, pred_binary, zero_division=0)) # Precision
                temp_F1_list.append(f1_score(true_list, pred_binary))         # F1-score
                fpr, tpr, th = roc_curve(true_list, pred_list)
                pre, rec, th = precision_recall_curve(true_list, pred_list)
                temp_roc_list.append(auc(fpr, tpr))                         # AUROC
                temp_prc_list.append(auc(rec, pre))                         # AUPRC
                # ---------------------------------------------------------------
            temp_loss_list.append(loss.item())

        # 
        if len(temp_roc_list)>0:
            train_acc = [np.mean(temp_acc_list), np.mean(temp_rec_list), np.mean(temp_pre_list),
                         np.mean(temp_F1_list),  np.mean(temp_roc_list), np.mean(temp_prc_list)]
            train_acc_std = [np.std(temp_acc_list), np.std(temp_rec_list), np.std(temp_pre_list),
                             np.std(temp_F1_list),  np.std(temp_roc_list), np.std(temp_prc_list)]
        else: train_acc, train_acc_std = [None, None, None, None, None, None], [None, None, None, None, None, None]
        train_loss, train_loss_std  = np.mean(temp_loss_list), np.std(temp_loss_list)
        return train_loss, train_loss_std, train_acc, train_acc_std


    # 
    def _valid(self, loader=None):
        """
        input
        --------------------
        loader (dataloader) : you can select data loader (for permutation importance)

        output
        --------------------
        valid_loss (float) : the mean of loss
        valid_loss_std (float) : the standard deviation of loss
        valid_acc (float) : the mean of accuracy
        valid_acc_std (float) : the standard deviation of accuracy
        """
        valid_loss, valid_loss_std = 0.0, 0.0
        valid_acc, valid_acc_std   = [], []
        temp_loss_list = []
        temp_acc_list, temp_rec_list, temp_pre_list, temp_F1_list = [], [], [], []
        temp_roc_list, temp_prc_list = [], []

        if loader is None: use_loader = self.valid_loader
        else             : use_loader = loader

        self.model.eval()
        with torch.no_grad():
            for i,(X1, X2, y, index) in enumerate(use_loader):
                X1, X2, y = X1.to(self.device), X2.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X1, X2)
                loss = self.criterion(outputs, y)

                pred = torch.transpose(outputs, 1, 0)
                pred_list = pred[1].detach().cpu().numpy().tolist()
                pred_binary = [1 if p>0.5 else 0 for p in pred_list]
                true_list = y.detach().cpu().numpy().tolist()
                if np.sum(true_list)>0:
                    # ---------------------------------------------------------------
                    temp_acc_list.append(accuracy_score(true_list, pred_binary))  # Accuracy
                    temp_rec_list.append(recall_score(true_list, pred_binary))    # Recall
                    temp_pre_list.append(precision_score(true_list, pred_binary, zero_division=0)) # Precision
                    temp_F1_list.append(f1_score(true_list, pred_binary))         # F1-score
                    fpr, tpr, th = roc_curve(true_list, pred_list)
                    pre, rec, th = precision_recall_curve(true_list, pred_list)
                    temp_roc_list.append(auc(fpr, tpr))                         # AUROC
                    temp_prc_list.append(auc(rec, pre))                         # AUPRC
                    # ---------------------------------------------------------------
                temp_loss_list.append(loss.item())

        # 
        if len(temp_roc_list)>0:
            valid_acc = [np.mean(temp_acc_list), np.mean(temp_rec_list), np.mean(temp_pre_list),
                         np.mean(temp_F1_list),  np.mean(temp_roc_list), np.mean(temp_prc_list)]
            valid_acc_std = [np.std(temp_acc_list), np.std(temp_rec_list), np.std(temp_pre_list),
                             np.std(temp_F1_list),  np.std(temp_roc_list), np.std(temp_prc_list)]
        else: valid_acc, valid_acc_std = [None, None, None, None, None, None], [None, None, None, None, None, None]
        valid_loss, valid_loss_std  = np.mean(temp_loss_list), np.std(temp_loss_list)
        return valid_loss, valid_loss_std, valid_acc, valid_acc_std


    # 
    def _test(self):
        """
        output
        --------------------
        index (list)
        pred_list (list) : the prediction values
        true_list (list) : the ground truth values
        """
        sm = nn.Softmax(dim=0)
        self.model.eval()
        with torch.no_grad():
            for i,(X1, X2, y, index) in enumerate(self.test_loader):
                X1, X2, y = X1.to(self.device), X2.to(self.device), y.to(self.device)
                outputs = self.model(X1, X2)
                pred = torch.transpose(outputs, 1, 0)
                pred = sm(pred)
                pred_list = pred[1].detach().cpu().numpy().tolist()
                true_list = y.detach().cpu().numpy().tolist()
                index = index.detach().cpu().numpy().tolist()
                index = ['I{:0=6}'.format(id) for id in index]
        return index, pred_list, true_list


