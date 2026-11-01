import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ArrondirModalites(BaseEstimator, TransformerMixin):
    def __init__(self, bornes=None):
        self.bornes = bornes

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.bornes.items():
            X[c] = X[c].round().clip(lo, hi)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


class ColumnFilter(BaseEstimator, TransformerMixin):
    def __init__(self, corr_feat=None, const_share=None, corr_y=None, prioritaires=None):
        self.corr_feat, self.const_share, self.corr_y = corr_feat, const_share, corr_y
        self.prioritaires = prioritaires

    def fit(self, X, y=None):
        self.n_in_ = X.shape[1]
        drop = set()
        if self.const_share is not None:
            for c in X.columns:
                if X[c].round(6).value_counts(normalize=True).iloc[0] >= self.const_share:
                    drop.add(c)
        if self.corr_feat is not None:
            cols = [c for c in X.columns if c not in drop]
            if self.prioritaires is not None:                      # colonnes à garder en premier
                prio = set(self.prioritaires)
                cols = [c for c in cols if c in prio] + [c for c in cols if c not in prio]
            C = X[cols].corr().abs()
            up = C.where(np.triu(np.ones(C.shape), k=1).astype(bool))
            drop |= {c for c in up.columns if (up[c] > self.corr_feat).any()}
        if self.corr_y is not None:
            cy = X.corrwith(pd.Series(np.asarray(y), index=X.index)).abs()
            drop |= set(cy[cy < self.corr_y].index)
        self.drop_ = sorted(drop)
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_)
