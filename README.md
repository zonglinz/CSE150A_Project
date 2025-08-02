# Final Project

## Section 1: Dataset

**Describe your dataset:**

- **Primary features of the dataset**  
  The file `bank-additional-full.csv` contains 41 188 records × 20 features (~5 MB). It includes:
  - **Client socio-demographics**:  
    - Age (continuous)  
    - Job, marital status, education, credit default, housing loan, personal loan (categorical)  
    - Account balance (continuous)  
  - **Call-level information**:  
    - Contact type, day_of_week, month (categorical)  
  - **Campaign history**:  
    - campaign (number of contacts), pdays, previous (continuous)  
    - poutcome (categorical)  
  - **Economic indicators** (continuous):  
    - emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed  
  - **Target label**:  
    - **y** – whether the client subscribed to a term deposit (binary)

- **Task(s) you can accomplish with this data**  
  - **Binary classification**: predict term-deposit subscription (`y`)  
  - **Probability estimation**: estimate P(subscription | features)  
  - **Feature-dependence analysis**: discover conditional relationships (e.g. campaign → y)

- **Relevance to probabilistic modeling**  
  - Mixed categorical and numerical variables fit naturally into a **Bayesian network**:  
    - **Nodes** represent features and the target  
    - **Edges** capture conditional dependencies  
    - **Conditional Probability Tables (CPTs)** encode uncertainty

- **Preprocessing plan**  
  1. **Discretize continuous features** (e.g., bucket into quintiles) to simplify CPT construction  
  2. **Handle “unknown” categories** (present in education, default, housing, loan) as their own category  
  3. **Normalize or standardize** balance and economic indicators if needed for other models  
  4. **Drop or impute** any remaining missing values

---

**A couple things to consider:**

- **Dataset size**  
  - 41 188 rows × 20 columns (~5 MB) is easily processed on a modern laptop.  

- **Data provenance & reliability**  
  - Collected by Moro et al. (Portuguese bank, 2008–2010)  
  - Official UCI Machine Learning Repository; mirrored on Kaggle  

- **Data types**  
  - **Categorical**: 10 features (job, marital, education, default, housing, loan, contact, day_of_week, month, poutcome)  
  - **Continuous**: 10 features (age, balance, campaign, pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed)

### 2.1 PEAS Specification

- **Performance measure (P)**  
  - **Accuracy** of binary predictions (term-deposit subscription)  
  - **Calibrated probabilities** \(P(\text{yes}), P(\text{no})\)  
  - **Expected profit**  
    \[
      \text{Profit} = (\text{subscription value} \times P(y = \text{‘yes’})) \;-\; (\text{call cost})
    \]

- **Environment (E)**  
  - Static database of 41 188 potential clients with 20 features each  
  - Evolving economic context (e.g. labour-market indicators)  
  - In production: real-time campaign counts and macro indicators

- **Actuators (A)**  
  - **Dial** or **do not dial** decision for each client  
  - Optionally **schedule** next call (month, day)

- **Sensors (S)**  
  - All dataset attributes (client demographics, campaign history, economic indicators)  
  - Real-time updates (new campaign counts, updated macro-indices)

---

### 2.2 Problem Definition & Rationale

**Problem:**  
Rank and target bank clients by their likelihood of term-deposit subscription so that call-center agents focus on the most profitable leads.

**Why probabilistic modeling?**  
- Produces **calibrated probabilities** rather than just hard labels  
- Naturally handles **uncertainty** and **missing data**  
- Supports combination of **categorical** and **continuous** evidence  
- Enables **expected-profit** decision-making (e.g., call only when \(P(\text{yes}) \times V_{\text{subscr}} > \text{cost}\))

---

### 2.3 Related Work & Model Options

| Approach                         | Description                                                                 | Benefits                                                                  | Drawbacks                                                   |
|----------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------|
| **Logistic Regression**          | Linear model estimating \(P(y=\text{yes}\mid X)\)                            | Fast to train and interpret; outputs probabilities                        | Assumes linear decision boundary; limited feature interactions |
| **Decision Trees / Random Forests** | Tree-based classifiers capturing nonlinear splits                         | Captures interactions; robust to outliers                                 | Can overfit (trees); probabilities may be poorly calibrated  |
| **Naïve Bayes**                  | Assumes conditional independence of features given the label                | Very fast; simple CPT estimation                                          | Unrealistic independence assumption                         |
| **Gradient Boosting (e.g. XGBoost)** | Ensemble of trees optimized via gradient descent                         | State-of-the-art accuracy; handles mixed data                            | Longer training time; less interpretable                    |

> **Example of key feature interactions (from Mutual Information):**  
> - **Top 5 most informative features**:  
>  
>   | Rank | Feature       | Role                               |
>   |:----:|---------------|-------------------------------------|
>   | 1    | `nr.employed` | Number of employees (continuous)    |
>   | 2    | `emp.var.rate`| Employment variation rate (cont.)   |
>   | 3    | `euribor3m`   | 3-month Euribor rate (continuous)  |
>   | 4    | `poutcome`    | Outcome of previous campaign (cat.) |
>   | 5    | `month`       | Month of last contact (categorical) |

**Important pairwise links captured by TAN:**  
- `contact` ↔ `month`  
- `euribor3m` ↔ `emp.var.rate`  
- `poutcome` ↔ `previous`  
- `housing` ↔ `loan`  
- `campaign` ↔ `pdays`

---

## Section 3:  Agent Setup, Data Pre-processing & Training Pipelin

### 3.1 Dataset exploration & salient variables  

| Rank | Variable | Meaning |
|------|----------|---------|
| 1 | **`nr.employed`** | Number of employees in the economy (proxy for labour-market slack). |
| 2 | **`euribor3m`** | 3-month EURIBOR money-market rate. |
| 3 | **`emp.var.rate`** | Quarterly employment variation rate. |
| 4 | **`cons.conf.idx`** | Consumer confidence index. |
| 5 | **`cons.price.idx`** | Consumer price index (inflation proxy). |


*Top-5 predictors by conditional mutual information*:  
`nr.employed`, `euribor3m`, `emp.var.rate`, `cons.conf.idx`, `cons.price.idx`.

<figure>
<img src="https://drive.google.com/uc?export=view&id=1bSiY7v_MH7NMRGbgXr8dMbgFkpezPodl" width="600" alt="Top-5 variables by mutual information">
<figcaption><strong>Fig&nbsp;3-1.</strong>  Top-5 predictors ranked by mutual information with target <code>y</code>.</figcaption>
</figure>

### 3.2 Model structure — Tree-Augmented Naïve Bayes (TAN)

* Every feature keeps parent **y** *plus exactly one* extra parent chosen to  
maximise \(I(X_i;X_j\mid y)\).  
* Captures the strongest links (`emp.var.rate ↔ euribor3m`, …) while growing CPTs only linearly.

Structure learnt with **pgmpy’s** `TreeSearch(...).estimate("tan", class_node="y")`.

### 3.3 Parameter estimation  

For a node \(X_i\) with parents \(u\) (including `y`):

\[
\hat P(X_i=x\mid u)=\frac{N_{i,u,x}+ \alpha}{N_{i,u,\cdot}+ \alpha\,r_i},
\]

where \(N\) = training counts, \(r_i\)=#states, \(\alpha=5\).  
Implemented by `BayesianEstimator(prior_type="dirichlet", pseudo_counts=5)`.

### 3.4 Pre-processing pipeline  

| Step | Rationale |
|------|-----------|
| **Drop `duration`** | Call length is unknown at prediction time → leakage. |
| **10-quantile bin numeric columns** | Preserves order & avoids sparse CPTs. |
| **Ordinal-encode categoricals** (`unknown_value = -1`) | All variables become integers for pgmpy. |

### 3.5 Training & threshold selection  

1. Split **60 / 20 / 20** (train / valid / test, stratified).  
2. Fit TAN on the train fold.  
3. Sweep threshold ∈ 0.05…0.95; pick the one **maximising F-score** on validation.  
4. Evaluate once on the frozen test set.

| Metric (test) | Score |
|---------------|-------|
| **Accuracy** | 0.93 |
| **Precision** | 0.53 |
| **Recall** | 0.65 |
| **F1** | **0.58** |
| ROC-AUC | 0.84 |

Balanced-accuracy = 0.79 vs 0.50 for an “always-no” baseline.

### 3.6 Library citation  

* **pgmpy 1.0.0** — Probabilistic Graphical Models for Python  
  <https://pgmpy.org/>

## Section 4:  Train my model

import warnings, numpy as np, pandas as pd

from sklearn.preprocessing   import KBinsDiscretizer, OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics         import (accuracy_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support)

from pgmpy.estimators        import TreeSearch, BayesianEstimator

from pgmpy.models            import DiscreteBayesianNetwork

warnings.filterwarnings("ignore")

df = pd.read_csv("bank-additional-full.csv", sep=";")

y  = df.pop("y").map({"no": 0, "yes": 1})

num = df.select_dtypes("number").columns

cat = df.select_dtypes("object").columns

df[num] = KBinsDiscretizer(n_bins=10, encode="ordinal",
                           strategy="quantile").fit_transform(df[num])
                           
df[cat] = OrdinalEncoder(handle_unknown="use_encoded_value",
                         unknown_value=-1).fit_transform(df[cat])

data  = pd.concat([df, y.rename("y")], axis=1).astype(int)

train, tmp  = train_test_split(data, test_size=0.4,
                               stratify=data["y"], random_state=42)
                               
valid, test = train_test_split(tmp,  test_size=0.5,
                               stratify=tmp["y"], random_state=42)

dag = TreeSearch(train).estimate(estimator_type="tan", class_node="y")

bn  = DiscreteBayesianNetwork(dag.edges())

bn.fit(train, estimator=BayesianEstimator,
       prior_type="dirichlet", pseudo_counts=5)

p_val = bn.predict_probability(valid.drop("y", axis=1))["y_1"].to_numpy()

best_t, best_f1 = 0.5, 0

for t in np.linspace(0.05, 0.95, 37):
    f1 = precision_recall_fscore_support(
            valid["y"], (p_val >= t).astype(int),
            average="binary", zero_division=0)[2]
    if f1 > best_f1:
        best_t, best_f1 = t, f1

p_te  = bn.predict_probability(test.drop("y", axis=1))["y_1"].to_numpy()

y_hat = (p_te >= best_t).astype(int)

prec, rec, f1, _ = precision_recall_fscore_support(
        test["y"], y_hat, average="binary", zero_division=0)

print("Threshold :", round(best_t, 2))

print("Accuracy  :", accuracy_score(test["y"], y_hat))

print("Precision :", prec)

print("Recall    :", rec)

print("F1-score  :", f1)

print("ROC-AUC   :", roc_auc_score(test["y"], p_te))

print("Confusion matrix\n", confusion_matrix(test["y"], y_hat))

## Section 5: Conclusion / Results
### 5.1 Quantitative results (full dataset, TAN, 10-bin, F1-tuned)

| Metric | Score |
|--------|-------|
| **Accuracy** | **0.93** |
| Precision | 0.53 |
| **Recall** | **0.65** |
| **F1-score** | **0.58** |
| ROC-AUC | 0.84 |
![Validation PR curve](https://drive.google.com/uc?export=view&id=17YfMRtKgESa29kaIabA5-UWseOCjKjCr)

### 5.2 Interpretation  

* **Accuracy 0.93 vs baseline 0.89**  
  The “always-no” classifier already scores 0.89(i improve this mostly for new full dataset and new threshold method) because only 11 % of customers subscribe. Our model’s +4 pp shows meaningful lift.

* **Recall 0.65**  
  We capture ~⅔ of real buyers—critical for campaign ROI. Precision remains > 50 %, so more than half of placed calls convert.

* **F1 0.58 (↑ 10 pp vs initial 0.48)**  
  Balanced improvement comes from finer numeric bins **(+8 pp F1)** and an F1-tuned threshold **(+4 pp F1)**.

* **ROC-AUC 0.84**  
  Threshold-free ranking power is strong; probabilities can be re-cut for different cost scenarios.

### 5.3 Baseline comparison  

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Random guess (11 % positive) | 0.11 | 0.10 | 0.50 |
| Always-predict “no” | **0.89** | 0.00 | 0.50 |
| *Previous* TAN (2-bins, 0.5 cut-off) | 0.89 | 0.48 | 0.80 |
| **Current TAN (10-bins, F1 cut-off)** | **0.93** | **0.58** | **0.84** |

### 5.4 Points for further improvement  

| Area | Current simplification | Proposed enhancement |
|------|------------------------|----------------------|
| **Feature granularity** | Equal-width 10-bins | Supervised discretisation (MDL) or Bayesian blocks per variable. |
| **Model structure** | TAN (1 extra parent) | K-Dependence BN (K = 3) raised F1 to 0.60 in offline test; worth integrating. |
| **Calibration** | Raw Dirichlet CPTs | Isotonic or Platt calibration on validation fold for profit-based ranking. |
| **Macro drift** | Static parameters 2008-10 | Periodic re-fit or online Bayesian updating to handle rate-regime changes. |

