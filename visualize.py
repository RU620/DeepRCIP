if __name__=='__main__':

    import yaml
    import pandas as pd
    from src.chem import *

    with open('config_visualize.yaml','r') as f: config = yaml.safe_load(f)

    model_path = config['model_path'] + config['folder_name']

    # Important bit substructure visualization
    # obtain importance of each ECFP bit
    df_importances = pd.read_csv(model_path+'permutation_importance/importances.txt')
    importances = [df_importances[col].values for col in df_importances.columns]
    # importances [0]: Accuracy, [1]: Recall, [2]: Precision, [3]: F1, [4]: AUROC, [5]: AUPRC

    svg = vis_attentionedBit(config['smiles'], importances[0], name=config['name'], t_w=config['threshold'])

    # save
    with open(model_path+config['save_filename']+'.svg', 'w') as f: f.write(svg)