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
