# Predicting Hospital Length of Stay: Structured Data vs. Clinical Notes

A data science project that predicts patient length of stay (LOS) **at admission** for a hospital, using two approaches and comparing them: traditional machine learning on structured data, and transformer language models on clinical notes. It ends with an ethics and governance analysis.

Built as the summative lab for a Data Science and AI bootcamp, using Google Colab.

## The problem

A hospital wants to improve bed management and staffing. This project asks how well length of stay can be predicted from information available when a patient is admitted, which factors drive long stays, and whether the model is fair and safe to use.

## Data

- 10,000 **synthetic** patient records, 31 columns, admissions from May 2024 to May 2025.
- Structured fields (demographics, diagnosis, comorbidities, vitals, admission type, surgery) plus a free-text clinical note per patient.
- The dataset was provided by the bootcamp and is **not included** in this repository.

## Approach

**Part 1: Structured data.** Exploratory analysis, preprocessing, feature engineering, and four model families (Linear/Ridge, Random Forest, Gradient Boosting) with 5-fold cross-validation and grid search. All preprocessing sits inside a scikit-learn Pipeline so scaling and encoding are fit only on training folds.

**Part 2: Clinical notes.** Text exploration and cleaning, then DistilBERT used two ways: frozen embeddings with a Ridge regressor, and end-to-end fine-tuning with a regression head. The text and structured predictions are then averaged and compared by patient group.

**Part 3: Ethics.** Privacy and consent, representation and bias, per-group fairness analysis with bootstrap confidence intervals, an exact per-patient explanation of the final model, and a monitoring and governance plan.

## Results (held-out test set, 2,000 patients)

| Model | RMSE (days) | MAE (days) | R² |
|---|---|---|---|
| Mean baseline | 4.03 | 2.84 | 0.00 |
| Linear Regression | 3.45 | 2.26 | 0.27 |
| Ridge (tuned) | 3.45 | 2.26 | 0.27 |
| Random Forest (tuned) | 3.44 | 2.24 | 0.27 |
| **Gradient Boosting (tuned)** | **3.41** | **2.21** | **0.28** |
| DistilBERT embeddings + Ridge | 3.45 | 2.25 | 0.27 |
| DistilBERT fine-tuned | 3.45 | 2.23 | 0.27 |
| Average (Gradient Boosting + fine-tuned DistilBERT) | 3.41 | 2.20 | 0.29 |

The final structured model was selected using cross-validation scores, not test scores.

## Key findings

- **Model choice barely matters.** Every model plateaus around R² 0.27 to 0.29. The best model is purely additive (depth-1 trees), so the limit is the information in the features, not the algorithm.
- **What drives length of stay:** a high-risk diagnosis group (sepsis, stroke, heart failure, COPD), age, surgery, number of comorbidities, and emergency admission. Vitals, demographics, insurance, and department add almost nothing.
- **Typical accuracy:** about 61% of predictions are within 2 days and 81% within 3 days. That is useful for planning a ward, but not precise enough for decisions about an individual patient.
- **Blind spot:** stays of 15+ days (about 5% of patients) were under-predicted by about 11 days on average. Nothing in the admission-time data distinguishes them.
- **Data leakage found and fixed.** The clinical notes contained ICU status, which is not known at admission. A first cleanup that simply deleted the ICU sentence left a detectable gap and produced a suspiciously good text model (R² 0.340). Replacing the sentence instead brought it back to 0.270. The lesson: removing a leaky field is not enough, because its absence can also be a signal.
- **Fairness:** error is similar across gender, race/ethnicity, insurance, income, and age groups. Apparent disparities in small groups were driven by a few long-stay patients and did not hold up in the full dataset. Groups with fewer than about 110 patients are too small to draw firm conclusions.

## Limitations

- The data is synthetic, from one hospital and one year, so results say little about real-world performance.
- Surgery is assumed to be known at admission.
- Feature importance describes what the model uses, not causal effects.
- Fairness was assessed on the test set only, and small groups have wide uncertainty.

## How to run

1. Open the notebook in Google Colab.
2. Set **Runtime, Change runtime type, T4 GPU** (needed for the fine-tuning section).
3. Upload `hospital_patient_data.csv` to the Colab session (it is not included here).
4. Run all cells. Fine-tuning takes about 5 minutes on a T4.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, PyTorch, and Hugging Face Transformers (DistilBERT).
