# Credit default prediction (Give Me Some Credit)

Continuous assessment project for **Machine Learning [P375]**, CUNEF Universidad. Initial submission: business problem, data source, scientific formulation and exploratory data analysis (EDA).

**Repository:** https://github.com/CarlosCUNEF/machine_learning_assignment

## Authors

| Full name | CUNEF email |
|---|---|
| Alejandro Alcázar | alejandro.alcazar@cunef.edu |
| Gregorio Atares | gregoriopedro.atares@cunef.edu |
| Carlos Rodrigo | carlos.rodrigo@cunef.edu |

## 1. Business problem

### Why it matters

Lending is the core business of a bank and also its main risk. Every loan that is not repaid is a direct loss, and it forces the bank to set aside capital as provisions. Each lending decision is a bet: if the bank is too permissive, defaults go up; if it is too strict, it turns away customers who would have paid and loses revenue.

The problem also has a social side. A good scoring system can grant credit to people with a short credit history but a good repayment profile, who would be excluded under rigid criteria. Better default prediction improves both the profitability of the bank and access to credit.

### Shortcomings of current solutions

- **Fixed rules** (for example, "reject if debt is more than X times income") are easy to explain, but they treat each variable separately and do not capture interactions between them. The thresholds are set by hand and age badly.
- **Traditional scorecards** (logistic regression on binned variables) are the industry standard and very interpretable, but they assume roughly linear and additive relationships, and they struggle with non-linear relationships and with extreme or missing values.
- In both cases, predictive power is limited by the manual design of variables and rules.

Machine Learning models can learn non-linear relationships and interactions directly from the data. Their challenge is to keep an acceptable level of interpretability, which regulators require for credit decisions.

**Business decision supported by the model:** who should be granted credit and on what terms, by estimating the probability that a customer will have a serious default in the next two years.

## 2. Data source

- **Dataset:** *Give Me Some Credit*, Kaggle competition from 2011.
- **Link:** https://www.kaggle.com/competitions/GiveMeSomeCredit
- **File used:** `cs-training.csv` (about 150,000 rows). The file `cs-test.csv` has no labels, so it cannot be used for evaluation and we will make the train/test split ourselves.

### Advantages

- The data is real, with real errors, missing values and extreme values, which gives the EDA content.
- The size is manageable.
- All variables are numeric, so any algorithm in the syllabus can be applied without friction.
- The business problem is clear and fits the profile of CUNEF.
- The class imbalance (around 7% positives) gives a solid basis for the choice of metrics.

### Disadvantages

- There are only 10 variables, which limits feature engineering in the final submission.
- There are no categorical variables.
- The origin is opaque: the institution, country and dates are unknown.
- The data is from 2011, so customer behaviour may have changed.
- The dataset has been heavily worked on in Kaggle, so there is little novelty.
- The test file has no labels.

## 3. Problem formulation

We treat credit scoring as a supervised learning problem. Each row of the dataset is a borrower, and the model has to estimate whether that borrower will have a serious delinquency in the next two years. The task is binary classification.

### Target

`SeriousDlqin2yrs` takes the value 1 if the person experienced 90 days past due delinquency or worse within two years, and 0 otherwise. About 7% of the borrowers in the labelled file are positive, so the classes are strongly imbalanced.

### Feature vector

Each borrower is described by 10 variables, all numeric:

| Variable | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Balance on credit cards and personal lines of credit divided by the sum of credit limits |
| `age` | Age of the borrower in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times the borrower was 30-59 days past due, but no worse, in the last two years |
| `DebtRatio` | Monthly debt payments, alimony and living costs divided by monthly gross income |
| `MonthlyIncome` | Monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans (installment or car loans) and lines of credit (credit cards) |
| `NumberOfTimes90DaysLate` | Times the borrower was 90 days or more past due |
| `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans, including home equity lines of credit |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times the borrower was 60-89 days past due, but no worse, in the last two years |
| `NumberOfDependents` | Number of dependents in the family, excluding the borrower |

### Choices that follow from the data

Accuracy is a poor metric here. A model that predicts "no default" for every borrower would be right about 93% of the time and would detect no defaulter. We will compare models with ROC-AUC and PR-AUC, and we will look at recall at a fixed precision, because missing a defaulter costs the bank more than reviewing a good borrower by hand.

The Kaggle test file has no labels, so we cannot evaluate on it. We will split the labelled file into train and test sets, stratified by the target, and keep the test set untouched until the final evaluation.

The EDA shows that some variables have missing values and some have impossible values. We decide in the EDA how to treat them (see sections 3 to 5 of notebook `01_EDA`), but every statistic learned from the data (the medians for imputation and the percentiles for capping) will be computed on the training set only and applied to the test set afterwards. If we computed them on the full file, information from the test set would leak into the training.

## 4. Exploratory data analysis

The full analysis is in `notebooks/01_EDA.ipynb`. Each section ends with its conclusions and the action we take. The corrected data of the notebook is `df_fixed`.

### Main findings

- Duplicates: There are 609 duplicated rows (0.41%). With no customer ID they look like different people with very simple profiles, so we keep them.
- Missing values: `MonthlyIncome` is missing in 19.82% of the rows and `NumberOfDependents` in 2.62%, always inside the missing-income rows. The missing income is related to the target (5.6% default rate against 7.0%), so it is imputed with the median and marked with the indicator `income_missing`.
- Errors: One age of 0. Codes 96 and 98 in the three late-payment counters (269 rows, with a default rate of 54.7%), replaced by missing values and marked with `late_code_flag`. `DebtRatio` holds a debt amount instead of a ratio when the income is missing, 0 or between 1 and 99 (more than 32,000 rows), so it is set to missing there.
- Outliers: The IQR rule is not enough on its own: it flags every late payment and every family with three dependents. Utilization above 10 is treated as an error (241 rows). The extreme values of `DebtRatio` and `MonthlyIncome` are real. They will be capped at the 99th percentile after the split, and income will also get a logarithm. Counts and ages over 100 are kept.
- Relationship with the target: The late-payment counters, utilization and age are the strongest variables. One late payment of 90 days or more raises the default rate from 4.6% to 33.7%. Open credit lines and real estate loans have a U-shaped relationship with default.
- Collinearity: The three late-payment counters have a correlation of 0.98 to 0.99 with the 96/98 codes and of 0.22 to 0.31 without them, so we keep the three. The highest correlation between other variables is 0.61 (`DebtRatio` and real estate loans).

### What is applied now and what after the split

Rules with a fixed threshold (an age below 18, the codes 96 and 98, utilization above 10, an income below 100) are applied in the EDA. Transformations that learn a value from the data (median imputation, capping at the 99th percentile, the logarithm of income after capping) will be fitted on the training set only in the final submission and then applied to the test set.

## 5. Repository structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── data/            # empty in the repo: place cs-training.csv here
└── notebooks/
    └── 01_EDA.ipynb
```

To reproduce the analysis, download `cs-training.csv` from Kaggle and save it in `data/`. The contents of that folder are excluded in `.gitignore`, so the data is not uploaded to the repository.
