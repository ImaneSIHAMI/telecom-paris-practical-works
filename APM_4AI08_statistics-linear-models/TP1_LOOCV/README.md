# TP1 - Leave-One-Out Cross-Validation

This practical work studies ordinary least squares and compares two
implementations of leave-one-out cross-validation:

- a naive method that fits the model once for every deleted observation;
- an efficient method based on the leverage values of the hat matrix.

The central identity is

$$
\widehat{\varepsilon}^{(i)}_i
=
\frac{\widehat{\varepsilon}_i}{1-h_{ii}}.
$$

## Files

- [`tp1_loocv_notebook.ipynb`](tp1_loocv_notebook.ipynb): executed notebook
  containing the implementation, numerical checks, and timing experiment.
- [`report/tp1_loocv_report.pdf`](report/tp1_loocv_report.pdf): final report.
- [`report/tp1_loocv_report.tex`](report/tp1_loocv_report.tex): LaTeX source.
- [`requirements.txt`](requirements.txt): Python dependencies.

## Running the notebook

```bash
python -m pip install -r requirements.txt
jupyter notebook tp1_loocv_notebook.ipynb
```

## Note

This repository contains personal academic work and is not an official course
solution.

