
import pickle
import numpy as np
import scanpy as sc, anndata as ad
import scipy.sparse as sp
import pandas as pd
import sys
import matplotlib.pyplot as plt
import re, seaborn as sns
from collections import OrderedDict
import os

from matplotlib import rcParams
from statannotations.Annotator import Annotator


class alignment_STs_analysis():
    def __init__(self, 
                 save_path, 
                 adata_mouse_embedding, 
                 adata_human_embedding, 
                 homo_region_file_path, 
                 mouse_labels_path, 
                 human_labels_path, 
                 species_list,
                 fig_format= 'jpg', 
                 fig_dpi=500):

        self.save_path = save_path
        self.adata_mouse_embedding = adata_mouse_embedding
        self.adata_human_embedding = adata_human_embedding
        self.homo_region_file_path = homo_region_file_path
        self.mouse_labels_path = mouse_labels_path
        self.human_labels_path = human_labels_path
        self.fig_format = fig_format
        self.fig_dpi = fig_dpi
        self.mouse_color = '#5D8AEF'
        self.human_color = '#FE1613'
        self.species1 = species_list[0]
        self.species2 = species_list[1]

    def experiment_homo_random(self):
        self.homo_corr()
        self.random_corr()
        self.plot_homo_random()
        self.ttest_homo_random()

    def homo_corr(self):
        # No label order version
        '''
        Step 1: Compute average embedding of every region in two species, use two dict to store;
        Step 2: Compute similarity matrix, use np array to store;
        Step 3: Heatmap.
        '''

        adata_mouse_embedding = self.adata_mouse_embedding
        adata_human_embedding = self.adata_human_embedding
        human_mouse_homo_region = pd.read_csv(self.homo_region_file_path)
        print(human_mouse_homo_region)
        home_region_dict = OrderedDict()
        for x, y in zip(human_mouse_homo_region[self.species2], human_mouse_homo_region[self.species1].values):
            home_region_dict[x] = y
        k = 0
        human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
        mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
        human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}
        distance_type = 'correlation'

        save_path_root = self.save_path + 'experiment_homo_random/homo_Hiercluster_correlation/' #.format(distance_type)

        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)

        with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
            pickle.dump(home_region_dict, f)

        for human_region, mouse_region in home_region_dict.items():
            save_path = save_path_root + 'human_{}_mouse_{}/'.format(
                human_region,
                mouse_region)

            save_path = save_path.replace(' ', '_')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            adata_human_embedding_region = adata_human_embedding[
                adata_human_embedding.obs['region_name'].isin([human_region])]
            adata_mouse_embedding_region = adata_mouse_embedding[
                adata_mouse_embedding.obs['region_name'] == mouse_region]
            if adata_mouse_embedding_region.X.shape[0] <= 0:
                continue
         
            # ---------human corr---------------------
            human_df = pd.DataFrame(adata_human_embedding_region.X).T
            human_corr = human_df.corr()
            mean, std = human_corr.mean().mean(), human_corr.stack().std()
            human_correlation_dict['human_region_list'].append(human_region)
            human_correlation_dict['mean'].append(mean)
            human_correlation_dict['std'].append(std)
            # ---------mouse corr---------------------
            mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
            mouse_corr = mouse_df.corr()
            mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
            mouse_correlation_dict['mouse_region_list'].append(mouse_region)
            mouse_correlation_dict['mean'].append(mean)
            mouse_correlation_dict['std'].append(std)
            # ---------------------------------------------------------------------
            ## Cross clustering of human and mouse
            result = pd.concat([human_df, mouse_df], axis=1).corr()
            Var_Corr = result[mouse_df.columns].loc[human_df.columns]
            mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()

            human_mouse_correlation_dict['human_region_list'].append(human_region)
            human_mouse_correlation_dict['mouse_region_list'].append(mouse_region)
            human_mouse_correlation_dict['mean'].append(mean)
            human_mouse_correlation_dict['std'].append(std)
            k += 1
            print('{}-th region finished!'.format(k))

        with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_mouse_correlation_dict, f)
        with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_correlation_dict, f)
        with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(mouse_correlation_dict, f)

    def random_corr(self):
        '''
        Step 1: Compute average embedding of every region in two species, use two dict to store;
        Step 2: Compute similarity matrix, use np array to store;
        Step 3: Heatmap.
        '''
        # Read ordered labels
        human_88_labels = pd.read_csv(self.human_labels_path)
        mouse_67_labels = pd.read_csv(self.mouse_labels_path)

        adata_mouse_embedding = self.adata_mouse_embedding
        adata_human_embedding = self.adata_human_embedding
        #
        human_mouse_homo_region = pd.read_csv(self.homo_region_file_path)

        home_region_dict = OrderedDict()
        mouse_region_list = human_mouse_homo_region[self.species1].values
        for x, y in zip(human_mouse_homo_region[self.species2].values, mouse_region_list):
            home_region_dict[x] = y

        human_88_labels_list = list(human_88_labels['region_name'])
        mouse_67_labels_list = list(mouse_67_labels['region_name'])

        k = 0

        human_correlation_dict = {'human_region_list': [], 'mean': [], 'std': []}
        mouse_correlation_dict = {'mouse_region_list': [], 'mean': [], 'std': []}
        human_mouse_correlation_dict = {'human_region_list': [], 'mouse_region_list': [], 'mean': [], 'std': []}

        distance_type = 'correlation'

        save_path_root = self.save_path + 'experiment_homo_random/random_Hiercluster_{}/'.format(
            distance_type)

        if not os.path.exists(save_path_root):
            os.makedirs(save_path_root)
        with open(save_path_root + 'homo_region_dict.pkl', 'wb') as f:
            pickle.dump(home_region_dict, f)

        for human_region in human_88_labels_list:
            if k >= 500:
                break
            for mouse_region in mouse_67_labels_list:
                if human_region in home_region_dict.keys() and home_region_dict[human_region] == mouse_region:
                    continue
                else:
                   
                    adata_human_embedding_region = adata_human_embedding[
                        adata_human_embedding.obs['region_name'] == human_region]
                    if min(adata_human_embedding_region.X.shape) <= 1:
                        continue

                    save_path = save_path_root + '/human_{}_mouse_{}/'.format(distance_type, human_region,
                                                                              mouse_region)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                   
                    adata_mouse_embedding_region = adata_mouse_embedding[
                        adata_mouse_embedding.obs['region_name'] == mouse_region]
                    if min(adata_mouse_embedding_region.X.shape) <= 1:
                        continue
                 
                    # ---------human corr---------------------
                    human_df = pd.DataFrame(adata_human_embedding_region.X).T
                    human_corr = human_df.corr()
                    mean, std = human_corr.mean().mean(), human_corr.stack().std()
                    human_correlation_dict['human_region_list'].append(human_region)
                    human_correlation_dict['mean'].append(mean)
                    human_correlation_dict['std'].append(std)
                    # ---------mouse corr---------------------
                    mouse_df = pd.DataFrame(adata_mouse_embedding_region.X).T
                    mouse_corr = mouse_df.corr()
                    mean, std = mouse_corr.mean().mean(), mouse_corr.stack().std()
                 
                    mouse_correlation_dict['mouse_region_list'].append(mouse_region)
                    mouse_correlation_dict['mean'].append(mean)
                    mouse_correlation_dict['std'].append(std)
                    # ---------------------------------------------------------------------
                    ## Cross clustering of human and mouse
                    result = pd.concat([human_df, mouse_df], axis=1).corr()
                    Var_Corr = result[mouse_df.columns].loc[human_df.columns]
                    mean, std = Var_Corr.mean().mean(), Var_Corr.stack().std()

                    human_mouse_correlation_dict['human_region_list'].append(human_region)
                    human_mouse_correlation_dict['mouse_region_list'].append(mouse_region)
                    human_mouse_correlation_dict['mean'].append(mean)
                    human_mouse_correlation_dict['std'].append(std)

                    k += 1
                    print('{}-th region finished!'.format(k))
                    if k >= 200:
                        break
            

        with open(save_path_root + 'human_mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_mouse_correlation_dict, f)
        with open(save_path_root + 'human_correlation_dict.pkl', 'wb') as f:
            pickle.dump(human_correlation_dict, f)
        with open(save_path_root + 'mouse_correlation_dict.pkl', 'wb') as f:
            pickle.dump(mouse_correlation_dict, f)

    def plot_homo_random(self):
        '''
        Step 1:load human and mouse cross expression data of homologous regions, and random regions
        Step 2: plot bar, mean and std
        '''
        sns.set(style='white')
        TINY_SIZE = 10  # 39
        SMALL_SIZE = 10  # 42
        MEDIUM_SIZE = 12  # 46
        BIGGER_SIZE = 12 # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']

        fig_format = self.fig_format

        homo_region_data_path = self.save_path + 'experiment_homo_random/homo_Hiercluster_correlation/'
        with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            human_mouse_correlation_dict = pickle.load(f)

        home_len = len(human_mouse_correlation_dict['mean'])
        home_random_type = ['Homologous'] * home_len
        human_mouse_correlation_dict['type'] = home_random_type

        random_region_data_path = self.save_path + 'experiment_homo_random/random_Hiercluster_correlation/'

        with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            random_human_mouse_correlation_dict = pickle.load(f)

        random_len = len(random_human_mouse_correlation_dict['mean'])
        home_random_type = ['Random'] * random_len
        random_human_mouse_correlation_dict['type'] = home_random_type
        concat_dict = {}
        for k, v in random_human_mouse_correlation_dict.items():
            concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
        data_df = pd.DataFrame.from_dict(concat_dict)

        save_path = self.save_path + 'experiment_homo_random/homo_random/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        my_pal = {"Homologous": '#F59E1D', "Random": '#28AF60'}

        rcParams["figure.subplot.right"] = 0.98
        rcParams["figure.subplot.left"] = 0.38
        rcParams["figure.subplot.bottom"] = 0.25

        plt.figure(figsize=(2, 2.5), dpi=self.fig_dpi)
        ax = sns.boxplot(x="type", y="mean", data=data_df, order=["Homologous", "Random"], palette=my_pal, width = 0.68, linewidth=0.8) #

        pairs = [("Homologous", "Random")]

        annotator = Annotator(
            ax, 
            pairs, 
            data=data_df, 
            x="type", 
            y="mean", 
            order=["Homologous", "Random"]
        )
        
        annotator.configure(
            test='t-test_welch',
            text_format='simple',        
            comparisons_correction='bonferroni',
            loc='inside',               
            verbose=2,
            show_test_name=False
        )

        annotator.apply_and_annotate()
        for item in ax.get_xticklabels():
            item.set_rotation(20)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right', rotation_mode="anchor")

        plt.title('')
        plt.ylabel('Correlation')
        plt.xlabel('')
        plt.savefig(save_path + 'mean.'+ fig_format, format=fig_format, dpi=self.fig_dpi)
        plt.show()

        data_df.to_csv(save_path + 'mean.csv')

        plt.figure(figsize=(2,2.5), dpi=self.fig_dpi)
        ax = sns.boxplot(x="type", y="std", data=data_df, order=["Homologous", "Random"], palette=my_pal)
        plt.title('')
        plt.savefig(save_path + 'std.'+ fig_format, format=fig_format, dpi=self.fig_dpi)

        homo_region_data_path = self.save_path + 'experiment_homo_random/homo_Hiercluster_correlation/'
        with open(homo_region_data_path + 'human_correlation_dict.pkl', 'rb') as f:
            human_correlation_dict = pickle.load(f)

        with open(homo_region_data_path + 'mouse_correlation_dict.pkl', 'rb') as f:
            mouse_correlation_dict = pickle.load(f)

        human_mouse_dict_mean = {self.species2: [], self.species1: []}
        human_mouse_dict_std = {self.species2: [], self.species1: []}

        human_mouse_dict_mean[self.species2] = human_correlation_dict['mean']
        human_mouse_dict_mean[self.species1] = mouse_correlation_dict['mean']

        human_mouse_dict_std[self.species2] = human_correlation_dict['std']
        human_mouse_dict_std[self.species1] = mouse_correlation_dict['std']

        rcParams["figure.subplot.right"] = 0.9
        rcParams["figure.subplot.left"] = 0.15
        rcParams["figure.subplot.bottom"] = 0.15

        sns.set(style='white')
        TINY_SIZE = 16  # 39
        SMALL_SIZE = 16  # 42
        MEDIUM_SIZE = 18  # 46
        BIGGER_SIZE = 18  # 46

        plt.rc('font', size=TINY_SIZE)  # 35 controls default text sizes
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=TINY_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Arial']


        ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        g = sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_mean), x=self.species2, y=self.species1, kind="reg", height=4, color='black', ax=ax)
        plt.title('')
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(human_mouse_dict_mean[self.species2]), np.array(human_mouse_dict_mean[self.species1]))
        plt.text(0.35, 0.30, f'R = 0.506, P < 0.05')
        print(f'R = {r_value}, P = {p_value}')
        plt.setp(g.ax_marg_y.patches, color=self.mouse_color)
        plt.setp(g.ax_marg_x.patches, color=self.human_color)
        plt.xlabel(f'{self.species2} mean correlation')
        plt.ylabel(f'{self.species1} mean correlation')
        plt.savefig(save_path + f'mean_{self.species2}_{self.species1}.'+ fig_format, format=fig_format, dpi=self.fig_dpi)

        ax = plt.figure(figsize=(4, 4), dpi=self.fig_dpi)
        sns.jointplot(data=pd.DataFrame.from_dict(human_mouse_dict_std), x=self.species2, y=self.species1, kind="reg", height=4, ax=ax)
        plt.title('')
        plt.savefig(save_path + f'std_{self.species2}_{self.species1}.'+ fig_format, format=fig_format, dpi=self.fig_dpi)
        rcParams["figure.subplot.left"] = 0.1

    def ttest_homo_random(self):
        '''
        Step 1:load human and mouse cross expression data of homologous regions, and random regions
        Step 2: plot bar, mean and std
        '''
        # Read ordered labels
        human_88_labels = pd.read_csv(self.human_labels_path)
        mouse_67_labels = pd.read_csv(self.mouse_labels_path)

        homo_region_data_path = self.save_path + 'experiment_homo_random/homo_Hiercluster_correlation/'
        with open(homo_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            human_mouse_correlation_dict = pickle.load(f)

        home_len = len(human_mouse_correlation_dict['mean'])
        home_random_type = ['homologous'] * home_len
        human_mouse_correlation_dict['type'] = home_random_type

        random_region_data_path = self.save_path + 'experiment_homo_random/random_Hiercluster_correlation/'

        with open(random_region_data_path + 'human_mouse_correlation_dict.pkl', 'rb') as f:
            random_human_mouse_correlation_dict = pickle.load(f)

        random_len = len(random_human_mouse_correlation_dict['mean'])
        home_random_type = ['random'] * random_len
        random_human_mouse_correlation_dict['type'] = home_random_type
        concat_dict = {}
        for k, v in random_human_mouse_correlation_dict.items():
            concat_dict[k] = human_mouse_correlation_dict[k] + random_human_mouse_correlation_dict[k]
        data_df = pd.DataFrame.from_dict(concat_dict)

        save_path = self.save_path + 'experiment_homo_random/homo_random/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        my_pal = {"homologous": (0 / 255, 149 / 255, 182 / 255), "random": (178 / 255, 0 / 255, 32 / 255)}

        random_df = data_df[data_df['type'] == 'random']

        mean_random_list = random_df['mean'].values
        mean_r = np.mean(mean_random_list)
        std_r = np.std(mean_random_list)

        homologous_df = data_df[data_df['type'] == 'homologous']
        mean_homo_list = homologous_df['mean'].values
        mean_h = np.mean(mean_homo_list)
        std_h = np.std(mean_homo_list)

        from scipy import stats

        print(stats.ttest_ind(
            mean_homo_list,
            mean_random_list,
            equal_var=False
        ))

        original_stdout = sys.stdout  # Save a reference to the original standard output

        with open(save_path + 't_test_result.txt', 'w') as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print(stats.ttest_ind(
                mean_homo_list,
                mean_random_list,
                equal_var=False
            ))
            sys.stdout = original_stdout  # Reset the standard output to its original value