# CSE150A_Project

# My Project PEAS:

P: Performance - Accuracy by binary prediction, and P(yes), P(no),  Expected profit = (subscription value × P(y=‘yes’)) − (call cost)

E: Environment - Database of 41 k potential clients with features above; dynamic economic context, in real life, it could demonstrate real called strategy 

A: Actuators - Dial/not-dial decision (and possibly scheduling a call month/day)

S: Sensors- All dataset attributes plus real-time updates (new campaign counts, macro-indices)

# What problem are you solving? Why does probabilistic modeling make sense to tackle this problem?

We must rank bank clients by likelihood of subscribing to a term-deposit so call-center agents target profitable leads. Probabilistic models output calibrated probabilities, handle uncertainty, combine categorical and numerical evidence, and support expected-profit decisions instead of crude binary yes-or-no classifications.

# Agent Setup, Data Preprocessing, Training setup
### 2.1 Dataset at a Glance  

`bank-direct-marketing-campaigns.csv`  |  **41 188 rows × 20 columns**

| Type              | Variables (19 predictors)                                                                                              | Role in Model                                                                                                                    |
|-------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Categorical (10)** | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`             | Discrete nodes in CPTs. Rare `"unknown"` values kept as their own state to preserve information.                                 |
| **Numeric (9)**      | `age`, `campaign`, `pdays`, `previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` | Bucketed into **quintiles** (Q1 – Q5) → 5-state discrete nodes so TAN CPTs stay small.                                           |

| Group                    | Variable(s)                            | Brief Description                                           | Why it matters to the agent/model                                  |
|--------------------------|----------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------------|
| **Socio-demographic**    | `age` (num)                            | Client’s age in years                                       | Older customers historically prefer fixed-term deposits.           |
|                          | `job` (12-level cat)                   | Occupation (e.g., “services”, “management”)                 | Proxy for income / risk profile.                                   |
|                          | `marital` (cat)                        | Married / single / divorced                                 | Household stability influences saving behaviour.                   |
|                          | `education` (cat)                      | Basic, secondary, tertiary, unknown                          | Financial-literacy / income proxy.                                 |
|                          | `default` (cat)                        | Past credit default: yes / no / unknown                     | Strong negative prior for subscription.                            |
| **Balance & liabilities**| `housing`, `loan` (cat)                | Existing mortgage / personal-loan flags                     | Competing cash-flow needs.                                         |
|                          | `balance` *                            | Average yearly balance (not in Kaggle mirror)               | *Dropped in our version.*                                          |
| **Outbound-call context**| `contact` (cat)                        | Channel (“cellular” vs “telephone”)                         | Channel effectiveness varies by cohort.                            |
|                          | `day_of_week`, `month` (cat)           | Day and month of last call                                  | Pay-day & seasonal effects.                                        |
| **Campaign intensity**   | `campaign` (num)                       | Number of contacts in current campaign                      | High counts ⇒ call fatigue risk.                                   |
|                          | `pdays` (num)                          | Days since last contact                                     | Recency of prior exposure.                                         |
|                          | `previous` (num), `poutcome` (cat)     | Past campaign count & outcome                               | Positive momentum or resistance.                                   |
| **Macro indicators**     | `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed` (num) | Labour-market & ECB-rate snapshot | Economic climate shifts deposit appeal.                            |
| **Target**               | `y` (binary)                           | 1 = client subscribed to term deposit                       | What the agent predicts.                                           |

We use Mutual information to measure how important does each variable does(Mutual information MI(X; y) is the reduction in uncertainty about the target y when you know a feature X.
In information-theory terms) 

Here is the diagram

<img src="https://drive.google.com/uc?export=download&id=1unDyLcThz3XGoy6ulaFV44hE1Jq-53vY" width="420"/>

so we do see nr.employed, emp.var.rate, euribor3m,poutcome, month are the top 5 variables that most matter




### 2.2 Our Model interaction
five dominant cross-feature links

contact ↔ month   

euribor3m ↔ emp.var.rate

poutcome ↔ previous 

housing ↔ loan

campaign ↔ pdays

To capture these while keeping parameters tractable we used a Tree-Augmented Naïve Bayes (TAN):

Every feature has parent y (as in Naïve Bayes).

CPT size = states(X) × states(extra-parent) × 2 → stays small.

* **Plain Naïve Bayes**: assumes conditional independence of \(X_i\) given \(y\).  
* **TAN**: adds **exactly one extra parent** per feature, chosen to maximise mutual information, while keeping learning linear in \(n\) and CPTs modest:

<img src="https://drive.google.com/uc?export=download&id=13gWQTYgvhqR7YZO31UR0-zjSj0U_Bz1r" width="420"/>


This captures key interactions (e.g., `contact` ↔ `month`) without the parameter blow-up of a fully connected BN.

### 2.3 Parameter Estimation  

For each discrete node \(X\) with parent set \(U\):

<img src="https://drive.google.com/uc?export=download&id=1obF56Q0bEDIWbXhSVse9eoTGTJDq6P0r" width="420"/>

Numeric features are first quintile-binned, so the same CPT formula applies.

---

#### Library Notes  

* **pgmpy** (v ≥ 1.0) — structure learning (`TreeSearch`, Chow–Liu / TAN) and parameter fitting (`BayesianEstimator`).  
* **scikit-learn** — preprocessing (`KBinsDiscretizer`, `OrdinalEncoder`)


# My_mode 
# 0 ▸ Import
import pandas as pd, numpy as np

from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             log_loss, brier_score_loss)
                             
from pgmpy.estimators import TreeSearch, BayesianEstimator

from pgmpy.models import BayesianNetwork

# 1 ▸ Load dataset
df = pd.read_csv("bank-direct-marketing-campaigns.csv")

y  = df.pop("y").map({"no": 0, "yes": 1})

# 2 ▸ Pre-process
num_cols = df.select_dtypes("number").columns

cat_cols = df.select_dtypes("object").columns

df[num_cols] = KBinsDiscretizer(n_bins=2, encode="ordinal",
                                strategy="quantile").fit_transform(df[num_cols])
                                
df[cat_cols] = OrdinalEncoder(handle_unknown="use_encoded_value",
                              unknown_value=-1).fit_transform(df[cat_cols])

data = pd.concat([df, y.rename("y")], axis=1).astype(int)

# 3 ▸ Train / test split
train, test = train_test_split(data, test_size=0.2,
                               stratify=data["y"], random_state=42)

# 4 ▸ Learn TAN structure & CPTs
dag   = TreeSearch(train, root_node="job").estimate(estimator_type="tan",
                                                   class_node="y")
                                                   
model = BayesianNetwork(dag.edges())

model.fit(train, estimator=BayesianEstimator,
          prior_type="dirichlet", pseudo_counts=1)

# 5 ▸ Evaluate
X, y_true = test.drop("y", axis=1), test["y"].to_numpy()

p_yes = model.predict_probability(X)["y_1"].to_numpy()

y_pred = (p_yes >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_true, y_pred))

print("F1:",       f1_score(y_true, y_pred))

print("ROC-AUC:",  roc_auc_score(y_true, p_yes))

print("Log-loss:", log_loss(y_true, np.c_[1-p_yes, p_yes]))

print("Brier:",    brier_score_loss(y_true, p_yes))

## Conclusion/Results
<img src="https://drive.google.com/uc?export=download&id=1mIm_Ob1OAhx5zQNSdAFfwolow-_EWlZd" width="500"/>

Below is the code portion result:

Accuracy: 0.8880796309783928

F1: 0.47791619479048697

ROC-AUC: 0.7981217421812349

Log-loss: 0.345792903290865

Brier: 0.09258993724574395

Despite accuracy 0.888 matching a “predict-no” baseline, other metrics show genuine skill. F1 0.478 proves the model retrieves nearly half of actual subscribers, while ROC-AUC 0.798 indicates strong ranking power. Log-loss 0.346 and Brier 0.093 reveal well-calibrated probabilities, enabling profit-based call ordering. Overall, the Tree-Augmented Naïve Bayes clearly outperforms trivial guessing, I think 0.88 is not really impressive number but we will keep improving our model(i will list those in next question gg)

# Improvement Proposal

Handle class imbalance explicitly: cost-sensitive learning or focal re-weighting during CPT fitting; choose decision threshold by maximising expected profit, not 0.5.

Class imbalance	: I noticed that there are a lot of false positives, and i saw only ~11 % “yes”, so we need to adjust the weighted on "yes" class

Increase granularity: replace uniform 2-bin quantisation with supervised MDL or Bayesian blocks so numeric predictors keep informative thresholds (euribor3m spikes).

Calibration check: apply isotonic regression on validation scores to sharpen probability estimates.
