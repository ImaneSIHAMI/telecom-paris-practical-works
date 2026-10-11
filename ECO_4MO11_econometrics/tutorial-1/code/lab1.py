"""Tutorial 1: OLS mechanics, simulation, and wage interpretation.

Student version. Keep the supplied folder structure and complete the items
marked TODO. The data and output paths are detected automatically.
The supplied sections continue running when a TODO is unfinished, so one
unfinished calculation does not prevent you from reaching the next section.

Requirements: numpy, pandas, matplotlib, statsmodels.
"""

# %% 0. Paths and imports - keep the supplied folder structure unchanged
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH_DATA = HERE.parent / "data" / "wage1.csv"
PATH_OUTPUT = HERE / "output"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def check_paths() -> None:
    """Check the supplied folder structure and create the output folder."""
    if not PATH_DATA.is_file():
        raise FileNotFoundError(
            f"Data file not found: {PATH_DATA}\n"
            "Extract the complete Tutorial 1 ZIP and keep the code and data "
            "folders together."
        )
    PATH_OUTPUT.mkdir(parents=True, exist_ok=True)


check_paths()
rng = np.random.default_rng(42)


def report_todo(label: str) -> None:
    print(f"TODO not completed: {label}")


# %% Exercise 3 simulation
exercise3_rng = np.random.default_rng(42)
samples = exercise3_rng.normal(loc=2, scale=1, size=(20, 10))
# TODO F1: extract the first observation and mean from each row.
first_observations = samples[:, 0]
sample_means = samples.mean(axis=1)
if first_observations is None or sample_means is None:
    report_todo("F1 Exercise 3 simulated estimators")
else:
    comparison = pd.DataFrame(
        {"first_observation": first_observations, "sample_mean": sample_means}
    )
    print(comparison.describe())
    comparison.to_csv(PATH_OUTPUT / "exercise3_comparison.csv", index=False)
    print("Ranges:")
    print(" first_observation:", first_observations.min(), first_observations.max())
    print(" sample_mean:", sample_means.min(), sample_means.max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    axes[0].hist(first_observations, bins=8, edgecolor="white")
    axes[0].axvline(2, linewidth=1.5, linestyle="--")
    axes[0].set(title="Estimator 1: first observation", xlabel="Estimate", ylabel="Count")
    axes[1].hist(sample_means, bins=8, edgecolor="white")
    axes[1].axvline(2, linewidth=1.5, linestyle="--")
    axes[1].set(title="Estimator 2: sample mean", xlabel="Estimate")
    fig.tight_layout()
    fig.savefig(PATH_OUTPUT / "exercise3_sampling_distributions.png", dpi=160)
    plt.show()
    plt.close(fig)


# %% Lab 1 - seeing exogeneity and bias by simulation
def ols_slope(x: np.ndarray, y: np.ndarray):
    """Return the OLS slope for a regression of y on a constant and x."""
    # TODO L1.1: replace None with the centred-sums formula.
    # Hint: xd = x - x.mean(); yd = y - y.mean()
    xd = x - x.mean()
    yd = y - y.mean()
    return np.sum(xd * yd) / np.sum(xd ** 2)


n, n_rep = 200, 5000
beta1_true, rho, sigma = 0.5, 0.4, 2.0
x = rng.normal(size=n)  # keep the same x values in every replication
test_y = 1 + beta1_true * x + rng.normal(scale=sigma, size=n)

if ols_slope(x, test_y) is None:
    report_todo("L1.1 OLS slope function")
else:
    b1_exogenous = np.empty(n_rep)
    b1_correlated = np.empty(n_rep)
    for r in range(n_rep):
        epsilon = rng.normal(scale=sigma, size=n)
        y_exogenous = 1 + beta1_true * x + epsilon
        y_correlated = 1 + beta1_true * x + rho * x + epsilon
        b1_exogenous[r] = ols_slope(x, y_exogenous)
        b1_correlated[r] = ols_slope(x, y_correlated)

    sst_x = ((x - x.mean()) ** 2).sum()
    print("Mean slope when u is independent of x:", b1_exogenous.mean())
    print("Target under exogeneity:", beta1_true)
    print("Mean slope when u is correlated with x:", b1_correlated.mean())
    print("Target when E(u|x) = rho*x:", beta1_true + rho)
    print("Empirical variance under exogeneity:", b1_exogenous.var())
    print("Theoretical variance conditional on x:", sigma**2 / sst_x)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, estimates, target, title in [
        (axes[0], b1_exogenous, beta1_true, "A: u independent of x"),
        (axes[1], b1_correlated, beta1_true + rho, "B: u correlated with x"),
    ]:
        ax.hist(estimates, bins=50, density=True, color="#287c8e", edgecolor="white")
        ax.axvline(beta1_true, color="#b04e16", linewidth=2, label="Structural slope")
        ax.axvline(target, color="#333333", linewidth=2, linestyle="--",
                   label="Distribution centre")
        ax.set(xlabel="Estimated slope", title=title)
    axes[0].set_ylabel("Density")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PATH_OUTPUT / "lab1_sampling_distribution.png", dpi=160)
    plt.show()
    plt.close(fig)

# WRITE L1.2: Explain why one estimate can be far from 0.5 under A and why the
# distribution under B is centred near 0.9. Identify the exogeneity condition
# that holds under A and fails under B. Would more replications remove the bias?


# %% Lab 2, Step 1 - inspect before estimating
raw = pd.read_csv(PATH_DATA)
variables = ["wage", "educ"]
print(raw.head())
print("Rows and columns:", raw.shape)
print(raw[variables].describe())
print("Missing values:\n", raw[variables].isna().sum())

keep = raw[["wage", "educ"]].notna().all(axis=1) & (raw["wage"] > 0)
df = raw.loc[keep].copy()
print("Used / excluded:", len(df), len(raw) - len(df))
df["log_wage"] = np.log(df["wage"])
df["wage_cents"] = 100 * df["wage"]

# TODO L2.1: calculate the median wage and the minimum and maximum education.
wage_median = df["wage"].median()
education_range = (df["educ"].min(), df["educ"].max())
if wage_median is None or education_range is None:
    report_todo("L2.1 median wage and education range")
else:
    print("Median wage:", wage_median)
    print("Education range:", education_range)

# WRITE L2.2: Define the sample, observation unit, wage units, education units,
# and the number of observations used in the regressions.


# %% Lab 2, Step 2 - visualise and form a prediction
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].hist(df.wage, bins=20, color="#287c8e", edgecolor="white")
axes[0].set(xlabel="Hourly wage (1976 US dollars)", ylabel="Number of workers")
axes[1].hist(df.log_wage, bins=20, color="#287c8e", edgecolor="white")
axes[1].set(xlabel="Log hourly wage", ylabel="Number of workers")
axes[2].scatter(df.educ, df.wage, alpha=0.35, s=14, color="#287c8e")
axes[2].set(xlabel="Education (years)", ylabel="Hourly wage (1976 US dollars)")
fig.tight_layout()
fig.savefig(PATH_OUTPUT / "lab2_distributions_and_scatter.png", dpi=160)
plt.show()
plt.close(fig)

# TODO L2.3: calculate the sample correlation between wage and education.
wage_education_correlation = df["wage"].corr(df["educ"])
if wage_education_correlation is None:
    report_todo("L2.3 wage-education correlation")
else:
    print("Wage-education correlation:", wage_education_correlation)

# WRITE L2.4: Predict the sign of the slope and explain why the scatterplot
# does not suggest accurate prediction of every worker's wage.


# %% Lab 2, Step 3 - fit and read a simple regression
levels = smf.ols("wage ~ educ", data=df).fit()
print("Levels coefficients:\n", levels.params)
print("n and R squared:", levels.nobs, levels.rsquared)

# TODO L2.5: reproduce the slope and intercept with centred sums.
xd = df["educ"] - df["educ"].mean()
yd = df["wage"] - df["wage"].mean()
centred_slope = (xd * yd).sum() / (xd ** 2).sum()
centred_intercept = df["wage"].mean() - centred_slope * df["educ"].mean()
if centred_slope is None or centred_intercept is None:
    report_todo("L2.5 centred-sums slope and intercept")
else:
    print("Centred-sums coefficients:", centred_intercept, centred_slope)
    print("Software coefficients:\n", levels.params)

grid = pd.DataFrame({"educ": np.linspace(df.educ.min(), df.educ.max(), 100)})
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(df.educ, df.wage, alpha=0.35, s=15, color="#287c8e")
ax.plot(grid.educ, levels.predict(grid), color="#b04e16", label="OLS fitted line")
ax.set(xlabel="Education (years)", ylabel="Hourly wage (1976 US dollars)")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PATH_OUTPUT / "lab2_fitted_line.png", dpi=160)
plt.show()
plt.close(fig)

# WRITE L2.6: Report the fitted equation, interpret one and four additional
# years of education, state what is held fixed, and discuss the intercept.


# %% Lab 2, Step 4 - fitted values and residuals
workers = pd.DataFrame({"educ": [12, 16]})
workers["fitted_wage"] = levels.predict(workers)
print(workers)

df["fitted_wage"] = levels.fittedvalues
df["residual"] = levels.resid
print(df[["educ", "wage", "fitted_wage", "residual"]].head())

# TODO L2.7: calculate the two mechanical OLS checks.
residual_sum = df["residual"].sum()
fitted_at_mean_education = levels.params["Intercept"] + levels.params["educ"] * df["educ"].mean()
if residual_sum is None or fitted_at_mean_education is None:
    report_todo("L2.7 residual sum and fitted value at mean education")
else:
    print("Residual sum:", residual_sum)
    print("Fitted at mean education:", fitted_at_mean_education)
    print("Mean wage:", df.wage.mean())

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(levels.fittedvalues, levels.resid, alpha=0.35, s=15, color="#287c8e")
ax.axhline(0, color="#b04e16")
ax.set(xlabel="Fitted hourly wage", ylabel="Residual (dollars/hour)")
fig.tight_layout()
fig.savefig(PATH_OUTPUT / "lab2_residuals.png", dpi=160)
plt.show()
plt.close(fig)

# WRITE L2.8: Calculate and interpret the first residual. Explain why the two
# OLS identities do not validate exogeneity. Report one fitted value while
# distinguishing it from an individual outcome.


# %% Lab 2, Step 5 - compare specifications and units
logs = smf.ols("log_wage ~ educ", data=df).fit()
cents = smf.ols("wage_cents ~ educ", data=df).fit()

specifications = pd.DataFrame(
    [
        {"model": label, "intercept": fit.params["Intercept"],
         "slope": fit.params["educ"], "slope_units": units,
         "n": int(fit.nobs), "R2": fit.rsquared}
        for label, fit, units in [
            ("A: wage", levels, "dollars/hour/year of education"),
            ("B: log wage", logs, "log points/year of education"),
            ("C: wage cents", cents, "cents/hour/year of education"),
        ]
    ]
)
print(specifications.to_string(index=False))
specifications.to_csv(PATH_OUTPUT / "lab2_specifications.csv", index=False)

workers["fitted_log_wage"] = logs.predict(workers)
workers["exp_fitted_log_wage"] = np.exp(workers["fitted_log_wage"])
print(workers)
workers.to_csv(PATH_OUTPUT / "lab2_predictions.csv", index=False)

# TODO L2.9: calculate the approximate and exact exponentiated-log percentage
# comparisons for one additional year of education.
approximate_percentage = 100 * logs.params["educ"]
exact_percentage = 100 * (np.exp(logs.params["educ"]) - 1)
if approximate_percentage is None or exact_percentage is None:
    report_todo("L2.9 approximate and exact percentage comparisons")
else:
    print("Approximate percentage comparison:", approximate_percentage)
    print("Exact exponentiated-log comparison:", exact_percentage)

# TODO L2.10: verify how the cents coefficients, residuals, and R squared
# relate to the dollar model. Store True or False in each item.
cents_coefficients_check = np.allclose(cents.params.values, 100 * levels.params.values)
cents_residuals_check = np.allclose(cents.resid.values, 100 * levels.resid.values)
cents_r2_check = np.isclose(cents.rsquared, levels.rsquared)
print("Cents checks:", cents_coefficients_check, cents_residuals_check, cents_r2_check)

# WRITE L2.11: Compare the dollar and percentage associations. Explain why the
# two R squared values are not directly comparable, why exponentiated fitted
# log wages are not automatically conditional mean wages, and why neither
# specification by itself identifies a causal return to education.
