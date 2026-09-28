import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import json

def train_and_evaluate(df_train_features, df_val_features, train_labels, val_labels):
    print("Training LightGBM...")
    # prepare data
    drop_cols = ['source1_entity_id', 'candidate_entity_id']
    X_train = df_train_features.drop(columns=drop_cols)
    y_train = train_labels
    
    X_val = df_val_features.drop(columns=drop_cols)
    y_val = val_labels
    
    print(f"Train shape: {X_train.shape}, Positives: {y_train.sum()}")
    print(f"Val shape: {X_val.shape}, Positives: {y_val.sum()}")
    
    model = lgb.LGBMClassifier(
        objective='binary',
        is_unbalance=True,
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=500,
        max_depth=-1,
        random_state=42,
        importance_type='gain'
    )
    
    # Use early stopping
    callbacks = [lgb.early_stopping(stopping_rounds=50)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='average_precision',
        callbacks=callbacks
    )
    
    # Validation predictions
    val_preds = model.predict_proba(X_val)[:, 1]
    
    val_auc_pr = average_precision_score(y_val, val_preds)
    print(f"Validation AUC-PR: {val_auc_pr:.4f}")
    
    # Save validation predictions
    df_val_res = df_val_features[['source1_entity_id', 'candidate_entity_id']].copy()
    df_val_res['score'] = val_preds
    df_val_res['label'] = y_val
    df_val_res.to_csv('../output/val_predictions.csv', index=False)
    
    # Save model
    model.booster_.save_model('../output/lgbm_model.txt')
    
    return model

def construct_labels(df_cands, ground_truth_dict):
    labels = []
    for idx, row in df_cands.iterrows():
        s1 = row['source1_entity_id']
        t = row['candidate_entity_id']
        true_matches = ground_truth_dict.get(s1, set())
        labels.append(1 if t in true_matches else 0)
    return pd.Series(labels)
