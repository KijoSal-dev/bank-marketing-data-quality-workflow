# Bank Marketing Response Prediction

This project uses machine learning to estimate which customers are more likely to subscribe to a term deposit after a bank marketing campaign. The goal is to support campaign planning by ranking customers for outreach when contact capacity is limited.

The analysis and modeling workflow is in [module1-week2 projects.ipynb](./module1-week2%20projects.ipynb). The notebook profiles the data, compares classification models, evaluates a capacity-based outreach strategy, and exports the trained pipeline and supporting results.

## Dataset

The project uses the **Bank Marketing (additional) dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). It contains 41,188 records and 20 predictor variables describing customer, campaign, and economic information. The target column is `y`; `yes` indicates that the customer subscribed to a term deposit.

The notebook expects the CSV at:

```text
data/uci-bank-marketing/bank-additional-full.csv
```

If the CSV is not present, the notebook downloads the official UCI archive and loads the dataset.

## Project structure

```text
.
├── module1-week2 projects.ipynb
├── data/
│   └── uci-bank-marketing/
│       └── bank-additional-full.csv
└── portfolio_outputs/
    └── week01_bank_marketing_ml/
        ├── bank_response_pipeline.joblib
        ├── config.json
        ├── data_quality_report.csv
        ├── eda_overview.png
        ├── feature_importance.csv
        ├── feature_importance.png
        ├── metrics.json
        ├── model_comparison.csv
        ├── model_evaluation.png
        ├── ranked_outreach_list.csv
        ├── schema.json
        └── test_predictions.csv
```

## Setup and run

1. Use Python 3.9 or later and install the notebook dependencies:

   ```bash
   python -m pip install jupyter pandas numpy scikit-learn matplotlib seaborn joblib
   ```

2. From the project root, launch Jupyter:

   ```bash
   jupyter notebook
   ```

3. Open `module1-week2 projects.ipynb` and run the cells from top to bottom. The notebook creates the output directory and saves generated artifacts under `portfolio_outputs/week01_bank_marketing_ml/`.

The notebook uses a fixed random seed (`42`) to make the train/validation/test splits reproducible. It reserves 20% of the data for testing and uses a validation split to select the model based on average precision.

## Modeling approach

- The target is `y`, with `yes` treated as the positive class.
- Numeric and categorical predictors are preprocessed in a scikit-learn pipeline. Categorical variables are one-hot encoded; numeric features are scaled for logistic regression.
- Logistic regression and a random forest are compared. The random forest is selected based on validation average precision.
- `duration` is excluded from model predictors because call duration is only known after the call and would not be available when deciding whom to contact.
- The final model is evaluated on the held-out test data. In addition to a 0.5 probability cutoff, the notebook evaluates a capacity-based strategy that selects approximately the top 20% of scored customers.

## Results

On the held-out test set, the selected random forest achieved:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.814 |
| Average precision | 0.491 |
| Precision at 0.5 threshold | 0.434 |
| Recall at 0.5 threshold | 0.633 |
| F1 at 0.5 threshold | 0.515 |

At approximately 20% contact capacity, the ranking strategy had **37.4% precision**, **66.5% recall**, and **3.32x lift** over the test-set positive rate of **11.3%**. These are evaluation results on a held-out sample, not a guarantee of future campaign performance. The model should be piloted and monitored before being used for operational decisions.

## Generated artifacts

- `metrics.json`: selected model and test-set metrics, including the capacity-based results.
- `model_comparison.csv`: validation metrics for the candidate models.
- `bank_response_pipeline.joblib`: fitted preprocessing and model pipeline.
- `ranked_outreach_list.csv`: customers ranked by predicted response probability.
- `test_predictions.csv`: held-out test-set predictions and scores.
- `feature_importance.csv` and `feature_importance.png`: permutation feature importance.
- `data_quality_report.csv`: data quality summary.
- `eda_overview.png` and `model_evaluation.png`: exploratory and evaluation visualizations.
- `config.json` and `schema.json`: modeling configuration and expected feature schema.

## Important considerations

- Predicted probabilities indicate estimated likelihood, not certainty or causation.
- Use the scores to prioritize outreach, not as the sole basis for customer decisions. Validate business impact through a controlled campaign test and consider contact cost, customer experience, and opt-outs.
- Check performance across time and relevant customer groups, and monitor for changes in data quality or model performance.
- The saved `.joblib` model is a Python pickle-based artifact. Only load it from a trusted source, and use compatible versions of its dependencies.
