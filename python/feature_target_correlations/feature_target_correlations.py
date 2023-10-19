"""
This is a feature importance tutorial for the CCEM group meeting.
"""
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
from sklearn.datasets import load_diabetes
from pathlib import Path
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')


def get_data(path: str = None, features: list = None, target: str = None):
    if path is None:
        data_bunch = load_diabetes(as_frame=True, scaled=True)
        df = data_bunch.frame
        features = data_bunch.feature_names
        target = data_bunch.target.name
    else:
        df = pd.read_csv(path)

    return df, features, target

def get_df_with_all_correlations_of_features_with_target(df, features, target):
    # Fix random seed for reproducibility
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # Definitions
    lin = 'linear'
    nonlin = 'nonlinear'
    mono = 'monotonic'
    corr_types = {}

    #%% Preprocess data
    df = df.select_dtypes(include='number')  # Select only numerical columns
    features = [col for col in df.columns if col in features]
    if not target in df.columns:
        raise ValueError(f'Could not find a numerical target "{target}" in dataframe!')

    # Scale numerical data to have mean 0 and variance 1
    scaler = StandardScaler()
    scaler.fit(df[df.columns])
    df[df.columns] = scaler.transform(df)



    # %% ==============    Correlations of features with target    ==================
    correlations = []
    for method in ['pearson',
                   'spearman']:  # 'pearson' catches linear correlations, 'spearman' catches monotonic correlations
        corrs = df.corr(method=method)[target][features]
        corrs = corrs.rename(method)
        correlations.append(corrs)
        corr_types[method] = lin if method == 'pearson' else mono
    correlations = pd.concat(correlations, axis=1)

    # %% ==============    Non-linear feature importances with Random Forests    ==================
    from sklearn.ensemble import RandomForestRegressor
    RF_model = RandomForestRegressor(n_estimators=20, random_state=0)
    RF_model.fit(df[features], df[target])
    RF_importances = RF_model.feature_importances_
    RF_importances = pd.Series(RF_importances, index=features)
    RF_importances = RF_importances.rename('Random Forest')
    corr_types['Random Forest'] = nonlin

    # %% ==============    Non-linear feature importances with permutation importance    ==================
    from sklearn.inspection import permutation_importance
    result = permutation_importance(RF_model, df[features], df[target], n_repeats=10,
                                    random_state=0)  # uses the Random Forest model from above
    perm_sorted_idx = result.importances_mean.argsort()
    perm_importances = pd.Series(result.importances_mean[perm_sorted_idx],
                                 index=np.array(features)[perm_sorted_idx].tolist())
    perm_importances = perm_importances.rename('Permutation')
    corr_types['Permutation'] = nonlin

    # %% ==============    Linear feature importances with PCA    ==================
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(df[features])
    PCA_importances = pca.components_[0]
    PCA_importances = pd.Series(PCA_importances, index=features)
    PCA_importances = PCA_importances.rename('PCA')
    corr_types['PCA'] = lin

    # %% ==============    Linear feature importances with Lasso    ==================
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.1)
    lasso.fit(df[features], df[target])
    Lasso_importances = lasso.coef_ / np.abs(lasso.coef_).sum()
    Lasso_importances = pd.Series(Lasso_importances, index=features)
    Lasso_importances = Lasso_importances.rename('Lasso')
    corr_types['Lasso'] = lin

    # %% ==============    Linear feature importances with Linear Regression    ==================
    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    linreg.fit(df[features], df[target])
    linreg_importances = linreg.coef_ / np.abs(linreg.coef_).sum()
    linreg_importances = pd.Series(linreg_importances, index=features)
    linreg_importances = linreg_importances.rename('Lin. Reg.')
    corr_types['Lin. Reg.'] = lin

    # %% ==============    Non-linear feature importances with XGBoost    ==================
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=0)
    xgb_model.fit(df[features], df[target])
    xgb_importances = xgb_model.feature_importances_
    xgb_importances = pd.Series(xgb_importances, index=features)
    xgb_importances = xgb_importances.rename('XGBoost')
    corr_types['XGBoost'] = nonlin

    # %% ==============    Combine and save all feature importances    ==================
    df_importances = pd.concat(
        [RF_importances, perm_importances, xgb_importances, Lasso_importances, linreg_importances, PCA_importances,
         correlations], axis=1)
    # Sort dataframe columns by linear, non-linear, and monotonic correlations and add this as a multiindex
    df_importances.columns = pd.MultiIndex.from_tuples([(corr_types[col], col) for col in df_importances.columns])
    df_importances = df_importances.sort_index(axis=1, level=0, ascending=False)
    # Sort dataframe by XGBoost feature importances
    df_importances = df_importances.sort_values(by=('nonlinear', 'XGBoost'), ascending=False)

    return df_importances



if __name__ == '__main__':

    output_dir = 'output'

    #%% ==============    Load data    ==================
    df, features, target = get_data()

    #%% ==============    Calculate feature importances    ==================
    df_importances = get_df_with_all_correlations_of_features_with_target(df, features, target)

    #% ==============    Print feature importances    ==================
    precision = 2
    pd.set_option('display.precision', precision)
    print(df_importances.round(precision))
    print('Other methods to look into for feature importances: SHAP, permutation importance, partial dependence plots, LIME, ELI5, ...')

    #%% ==============    Save and print results    ==================
    print(f'Saving output...')
    if not Path(output_dir).exists():
        raise FileNotFoundError(f'Could not find specified output directory "{output_dir}"!')
    df_importances.to_csv(f'{output_dir}/feature_importances.csv')
    print(f'- Feature importances saved to "{output_dir}/feature_importances.csv".')

    # Plot scatter plots of features vs target
    plotdir = Path(output_dir, 'correlation_plots')
    plotdir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        plt.figure()
        sns.regplot(x=feature, y=target, data=df, scatter_kws={'alpha': 0.5})
        plt.savefig(f'{plotdir}/{feature}_vs_{target}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    print(f'- Correlation plot of each feature & target saved to "{plotdir}".')

    print('Done!')



