import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from cuml.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm.notebook import tqdm

DRUGS = pd.read_csv('train_drug.csv')


def get_folds(y, n_folds: int, random_state: int):
    # y has 'sig_id' as the first column
    targets = y.columns[1:]
    y_drugs = y.merge(DRUGS, on='sig_id', how='left')

    vc = y_drugs['drug_id'].value_counts()
    vc1 = vc.loc[vc <= 18].index.sort_values()  # drugs with n <= 18
    vc2 = vc.loc[vc > 18].index.sort_values()  # drugs with n > 18

    drug_count_1, drug_count_2 = {}, {}

    # Stratify n <= 18
    ml_skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    temp = y_drugs.groupby('drug_id')[targets].mean().loc[vc1]

    for fold, (idxT, idxV) in enumerate(ml_skf.split(temp, temp[targets])):
        dd = {k: fold for k in temp.index[idxV].values}
        drug_count_1.update(dd)

    # Stratify n > 18
    ml_skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    temp = y_drugs.loc[y_drugs['drug_id'].isin(vc2)].reset_index(drop=True)

    for fold, (idxT, idxV) in enumerate(ml_skf.split(temp, temp[targets])):
        dd = {k: fold for k in temp['sig_id'][idxV].values}
        drug_count_2.update(dd)

    y_drugs['fold'] = np.nan
    y_drugs['fold'] = y_drugs['drug_id'].map(drug_count_1)
    y_drugs.loc[y_drugs['fold'].isna(), 'fold'] = y_drugs.loc[
        y_drugs['fold'].isna(), 'sig_id'].map(drug_count_2)

    return y_drugs[['sig_id', 'fold']]


def log_loss_score(X, y, v, n_folds=5, random_state=42, n_neighbors=1000):
    # First scale inputs
    X_scaled = StandardScaler().fit_transform(X.iloc(axis=1)[3:].values)
    X_scaled = np.hstack([X.iloc(axis=1)[:3].values, X_scaled])

    # Get Multi-Label Stratified K-Folds
    df_fold = get_folds(y, n_folds=n_folds, random_state=random_state)

    # Initialize array to store out-of-fold probabilities
    oof = np.zeros((X.shape[0], 206))
    for fold in range(n_folds):
        fold_idx = df_fold[df_fold['fold'] != fold].index
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(
            X_scaled[fold_idx][:, v],
            y.iloc[fold_idx].values[:, 1:])
        pp = model.predict_proba(X_scaled[fold_idx][:, v])
        pp = np.stack([(1 - pp[x][:, 0]) for x in range(len(pp))]).T
        oof[fold_idx, ] = pp

    # Return the accumulated log loss score across all folds
    return log_loss(y.iloc[:, 1:].values.flatten(), oof.flatten())


def mutation(x):
    if np.random.rand(1) < x:
        return True
    else:
        return False


class GeneticAlgorithm:
    def __init__(self, n_folds=5, random_state=42, n_neighbors=1000):
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit(self, X, y, population, generations=20, n_parents=102, cross_over=0.5, mutation_rate=0.1):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        print('Fitting {} generations...'.format(str(generations)))
        for generation in range(generations):
            print('GENERATION:', generation+1)
            fitness_scores = []
            print('Computing log loss scores...')
            for sample in tqdm(population):
                fitness_scores.append(
                    log_loss_score(X, y, sample,
                                   n_folds=self.n_folds,
                                   random_state=self.random_state,
                                   n_neighbors=self.n_neighbors))

            # Smallest log loss values to select parents
            smallest_idx = np.argsort(fitness_scores)[:n_parents]
            parents = np.array([population[i] for i in smallest_idx])
            lonely_parents = smallest_idx[n_parents:n_parents + 2]

            # Create offspring
            print('Generating offspring...')
            offspring_shape = (population.shape[0] - n_parents, population.shape[1])
            offspring = np.ndarray(offspring_shape)
            cross_over_point = int(1//cross_over)
            for i in range(offspring.shape[0]):
                parent1_idx = i % parents.shape[0]
                parent2_idx = (i + 1) % parents.shape[0]
                offspring[i, :cross_over_point] = parents[parent1_idx, :cross_over_point]
                offspring[i, cross_over_point:] = parents[parent2_idx, cross_over_point:]

                for j in range(len(offspring)):
                    if mutation(mutation_rate):
                        if offspring[j]:
                            offspring[j] = False
                        else:
                            offspring[j] = True

            population[:parents.shape[0], :] = parents
            population[parents.shape[0]:parents.shape[0]+2, :] = lonely_parents
            population[parents.shape[0]+2:, :] = offspring
            print('\n')
        return population


































