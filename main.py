if __name__=='__main__':

    import yaml
    from src.RCIP import *

    with open('config_main.yaml','r') as f: config = yaml.safe_load(f)

    # fix random seed
    SEED = 1234
    fix_seed(SEED)

    model_path = config['model_path'] + config['folder_name']

    # Creating folders for result saving
    if config['init']: make_folders(model_path)

    else: 
        # load dataset
        dataset = pd.read_csv(config['model_path']+'input/'+config['use_dataset']+'.csv').drop(columns=['Unnamed: 0'])
        if config['use_testset'] is not None: testset = pd.read_csv(config['model_path']+'input/'+config['use_testset']+'.csv').drop(columns=['Unnamed: 0'])
        else:                       testset = None
 
        # select device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # instanciation model
        rcip = RCIP(model_name=config['model_name'], result_save_path=model_path, 
                    device=device, c_input=config['ecfp_input_size'], r_input=config['rna_input_size'],
                    batch_size=config['batch_size'], num_epoch_cv=config['num_epoch_cv'], num_epoch=config['num_epoch'])

        # load dataset
        rcip.load_dataset(dataset, testset, method=config['split_type'], test_size=config['test_size'], column=config['test_column'], order_list=config['test_order'])

        if config['cv']:
            # cross-validation
            rcip.cross_validation(params_dict=config['params_dict'], cv_fold=config['cv_fold'])

        if config['train']:
            # set parameters
            rcip.set_parameters(config['dropout_rate'], config['l1_alpha'], config['kernel_size'], config['num_kernel'])
            # train
            rcip.train()
            # draw learning-curve
            draw_learning_curve(model_path)

        if config['test']:
            # set parameters
            rcip.set_parameters(config['dropout_rate'], config['l1_alpha'], config['kernel_size'], config['num_kernel'])
            # test
            rcip.test()

        if config['pi']:
            # set parameters
            rcip.set_parameters(config['dropout_rate'], config['l1_alpha'], config['kernel_size'], config['num_kernel'])
            # permutation importance
            os.mkdir(model_path+'permutation_importance')
            rcip.permutation_importance()
