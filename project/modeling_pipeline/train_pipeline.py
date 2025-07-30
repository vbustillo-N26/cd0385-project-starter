import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

def regression_autogluon(train_df, feature_cols, target_col='count', time_limit=600, path='../autogluon_models/bike_sharing'):
    
    train_df = train_df[feature_cols + [target_col]]

    predictor = TabularPredictor(
        label=target_col,
        eval_metric="root_mean_squared_error",
        path=path
    ).fit(
        train_data=train_df, 
        time_limit=time_limit, 
        presets='best_quality',
        # Simple hyperparameter configuration - just specify different models to include
        hyperparameters={
            'GBM': {},  # Use default LightGBM with AutoGluon's automatic tuning
            'XGB': {},  # Use default XGBoost with AutoGluon's automatic tuning
            'CAT': [
                {
                    'iterations': 6000,
                    "learning_rate": 0.01,
                    "depth": 8,
                    "min_data_in_leaf": 50,
                    "max_ctr_complexity": 3,
                    "l2_leaf_reg": 10,
                    "rsm": 0.45,
                    "subsample": 0.65,
                },
                {
                    'iterations': 6000,
                    "learning_rate": 0.01,
                    "depth": 8,
                    "min_data_in_leaf": 50,
                    "max_ctr_complexity": 2,
                    "l2_leaf_reg": 10,
                    "rsm": 0.40,
                    "subsample": 0.70,
                }
            ],  # Use CatBoost
            'NN_TORCH': [    # Neural Network with different configurations
                {'num_layers': 2, 'dropout_prob': 0.1, 'hidden_size': 64},
                {'num_layers': 3, 'dropout_prob': 0.2, 'hidden_size': 128},
                {'num_layers': 4, 'dropout_prob': 0.3, 'hidden_size': 256}

            ],
            'RF': [    # Random Forest with different configurations
                {'n_estimators': 6000, 'max_depth': 7, 'max_features': 0.65},
                {'n_estimators': 5000, 'max_depth': 7, 'max_features': 0.70},
                {'n_estimators': 4000, 'max_depth': 8, 'max_features': 0.80},
                {'n_estimators': 3000, 'max_depth': 8, 'max_features': 0.9},
            ],
            'XT': [    # Extra Trees with different configurations
                {'n_estimators': 6000, 'max_depth': 7},
                {'n_estimators': 6000, 'max_depth': 9},
            ]
        },
        verbosity=0
    )
    
    return predictor