"""Tutorial 2 student file: controls, partialling out, and collinearity.

Some commands are provided as examples; complete the analogous TODO items and the WRITE prompts.
The Lab 3 Monte Carlo code is supplied so that its interpretation is the focus.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
PATH_DATA = HERE.parent / "data" / "WAGE1.raw"
PATH_OUTPUT = HERE / "output"
PATH_OUTPUT.mkdir(parents=True, exist_ok=True)

RAW_COLUMNS = [
    "wage",
    "educ",
    "exper",
    "tenure",
    "nonwhite",
    "female",
    "married",
    "numdep",
    "smsa",
    "northcen",
    "south",
    "west",
    "construc",
    "ndurman",
    "trcommpu",
    "trade",
    "services",
    "profserv",
    "profocc",
    "clerocc",
    "servocc",
    "lwage",
    "expersq",
    "tenursq",
]


def heading(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def report_todo(label):
    print(f"TODO not completed: {label}")


if not PATH_DATA.is_file():
    raise FileNotFoundError(
        f"Data file not found: {PATH_DATA}\n"
        "Keep the code and data folders together inside the Tutorial 2 folder."
    )


# %% Lab 1, Step 1: define the comparison before estimation
heading("Lab 1, Step 1 - Define the comparison before estimation")
# The original .raw file is whitespace-separated and has no header row.
raw = pd.read_csv(PATH_DATA, sep=r"\s+", header=None, names=RAW_COLUMNS)
variables = ["wage", "educ", "exper", "tenure"]
df = raw[variables].dropna().copy()

if not (df["wage"] > 0).all():
    raise ValueError("All wages must be positive before taking logarithms.")

df["log_wage"] = np.log(df["wage"])
regressor_correlations = df[["educ", "exper", "tenure"]].corr()

print("Observations used:", len(df))
print("Regressor correlations:\n", regressor_correlations)
regressor_correlations.to_csv(PATH_OUTPUT / "lab1_regressor_correlations.csv")

# WRITE L1.1: Describe the comparison made by Model B's education coefficient.
# State what is held fixed and why the comparison is not automatically causal.
# ANSWER L1.1: Model B compares workers with the same years of experience
# and tenure but different education levels. The coefficient is a conditional
# association, not automatically causal: ability, family background, or
# unobserved opportunities may affect both education and wages.

# WRITE L1.2: Use the correlation signs to predict how omitting experience and
# tenure may move the simple education coefficient. Explain why this is only a
# prediction when several variables are omitted.
# ANSWER L1.2:
# Education is negatively correlated with experience (-0.2995) and tenure
# (-0.0562). If both have positive partial wage effects, omitting them tends
# to bias the education slope downward. This is suggestive, not a proof: with
# multiple omitted controls, the exact bias depends on their auxiliary
# regression slopes.





# %% Lab 1, Step 2: estimate and explain the coefficient change
heading("Lab 1, Step 2 - Estimate and explain the coefficient change")

# Model A is provided as an example.
simple = smf.ols("log_wage ~ educ", data=df).fit()

# COMPLETED L1.3: Estimate Model B by adding experience and tenure.
# Follow the Model A command above.
# ANSWER L1.3: Add both controls, including the intercept by default.

multiple = smf.ols("log_wage ~ educ + exper + tenure", data=df).fit()

if multiple is None:
    report_todo("L1.3 Model B regression")
else:
    comparison = pd.DataFrame(
        {
            "model": ["A: education only", "B: add experience and tenure"],
            "education_slope": [simple.params["educ"], multiple.params["educ"]],
            "experience_slope": [np.nan, multiple.params["exper"]],
            "tenure_slope": [np.nan, multiple.params["tenure"]],
            "n": [int(simple.nobs), int(multiple.nobs)],
            "R2": [simple.rsquared, multiple.rsquared],
        }
    )
    print(comparison.to_string(index=False))
    print("Education slope change (B minus A):", multiple.params["educ"] - simple.params["educ"])
    print("Approximate education percent effect in B:", 100 * multiple.params["educ"])
    print("Exact education percent effect in B:", 100 * np.expm1(multiple.params["educ"]))
    comparison.to_csv(PATH_OUTPUT / "lab1_model_comparison.csv", index=False)

# WRITE L1.4: How much does the education coefficient change? Interpret Model
# B's coefficient, using an approximate percentage and stating what is held fixed.
# ANSWER L1.4:
# The education slope rises from 0.082744 (A) to 0.092029 (B), a change of
# +0.009285 log points, or +0.9285 percentage points in the approximate
# interpretation. In B, one additional year of education is associated with
# about 9.20% higher hourly wages, holding experience and tenure fixed.





# WRITE L1.5: Does the direction of the change match your Step 1 prediction?
# Explain what happens when Model A leaves experience and tenure in its error.
# ANSWER L1.5:
# Yes: the education slope increases after adding controls, consistent with
# the predicted downward omitted-variable bias in A. Model A's error includes
# experience and tenure, whose associations with education and wages affect
# its education coefficient.




# WRITE L1.6: Why was Model B's R2 guaranteed not to be smaller? Does the
# increase make zero conditional mean more credible?
# ANSWER L1.6:
# Adding regressors expands the OLS feasible set, so RSS cannot increase and
# R2 cannot fall on the same sample. R2 rises from 0.185806 to 0.316013, but
# this does not establish the zero conditional mean assumption.




# %% Lab 1, Step 3: audit exogeneity as part of the analysis
heading("Lab 1, Step 3 - Audit exogeneity")

# WRITE L1.7: Explain E(u | educ, exper, tenure) = 0 in practical terms. Name
# three plausible determinants left in u, explain their links with education,
# and identify the one you find most concerning.
# ANSWER L1.7:
# Zero conditional mean requires unobserved wage determinants to average zero
# at every combination of educ, exper and tenure. Possible confounders:
# ability (influences school performance and productivity), parental resources
# (influences education and labor-market opportunities), and local
# opportunities (influences schooling and wages). Unmeasured ability is
# especially concerning.




# WRITE L1.8: Compare parents' education, a pre-school test score, and current
# occupation as possible controls. Which may proxy for confounding, and which
# may lie on a pathway from education to wages?
# ANSWER L1.8:
# Parents' education and a pre-school test score can proxy for pre-existing
# family background and ability. Current occupation may be affected by
# education and thus mediate part of its wage effect; controlling for it
# changes the causal question.




# WRITE L1.9: Explain why adding every available variable is a poor rule for
# choosing controls. Give two reasons.
# ANSWER L1.9:
# First, controlling for post-schooling mediators such as current occupation
# or possibly tenure removes parts of the total effect. Second, conditioning
# on variables affected by selection or common effects (colliders) can
# introduce bias. Controls must be chosen using their timing and causal roles.




# WRITE L1.10: Write the three-sentence conclusion requested in the handout.
# ANSWER L1.10:
# Holding experience and tenure fixed, one extra year of education is
# associated with about 9.20% higher hourly wages. The slope increases
# relative to the education-only model because the omitted controls are
# associated with education and wages. Unmeasured ability and family
# background remain important threats to a causal interpretation.






# %% Lab 2: what partialling out does and does not show
heading("Lab 2 - What partialling out does and does not show")

# Follow the procedure from Lecture 2, slide 14.
educ_aux = smf.ols("educ ~ exper + tenure", data=df).fit()
education_residual = educ_aux.resid

# WRITE L2.1: What does a positive education residual mean?
# ANSWER L2.1:
# A positive residual from educ ~ exper + tenure means a worker has more
# education than predicted by their experience and tenure, relative to the
# sample's fitted conditional linear relationship.



# COMPLETED L2.2: Use the slide-14 formula to calculate the education coefficient.
# The @ operator computes the required sums of cross-products.
# ANSWER L2.2: Sum_i(rx_i * log(wage_i)) / Sum_i(rx_i^2).

lecture_slope = float(education_residual @ df["log_wage"] / (education_residual @ education_residual))

if lecture_slope is None:
    report_todo("L2.2 slide-14 partialling-out calculation")
else:
    print("Slide-14 coefficient:", lecture_slope)
    if multiple is not None:
        print("Model B education coefficient:", multiple.params["educ"])

# At the end, verify the equivalent residual-on-residual procedure.
log_wage_aux = smf.ols("log_wage ~ exper + tenure", data=df).fit()
fwl_data = pd.DataFrame(
    {
        "education_residual": education_residual,
        "log_wage_residual": log_wage_aux.resid,
    }
)
symmetric_fwl = smf.ols(
    "log_wage_residual ~ education_residual", data=fwl_data
).fit()

print("Residual-on-residual intercept:", symmetric_fwl.params["Intercept"])
print("Residualized education has mean:", education_residual.mean())
print("FWL matches multiple regression:", np.isclose(lecture_slope, multiple.params["educ"]))
print(
    "Residual-on-residual coefficient:",
    symmetric_fwl.params["education_residual"],
)
if lecture_slope is not None:
    print(
        "Same as the slide-14 coefficient:",
        np.isclose(symmetric_fwl.params["education_residual"], lecture_slope),
    )

# WRITE L2.3: Explain why the residual-on-residual intercept is zero and why
# its slope equals the slide-14 result. Does this equality establish exogeneity?
# ANSWER L2.3:
# Both auxiliary regressions contain an intercept, so their residuals have
# sample mean zero; residual-on-residual regression therefore has a zero
# intercept up to rounding. FWL makes its slope equal to the full-model educ
# slope (0.092029). This is an algebraic identity, not evidence of exogeneity.






# %% Lab 3: when a precise regression answers the wrong question
heading("Lab 3 - When a precise regression answers the wrong question")
rng = np.random.default_rng(2026)


def simulate_design(rho, repetitions=3_000, n=150):
    """Return full-model b1, full-model b2, and the short-model x1 slope."""
    estimates = np.empty((repetitions, 3))

    for replication in range(repetitions):
        x1 = rng.normal(size=n)
        z = rng.normal(size=n)
        # Construct x2 with correlation rho with x1.
        x2 = rho * x1 + np.sqrt(1 - rho**2) * z
        u = rng.normal(size=n)
        y = 1 + x1 + x2 + u

        full_design = np.column_stack((np.ones(n), x1, x2))
        # Solve OLS directly for speed inside the loop.
        full_slopes = np.linalg.lstsq(full_design, y, rcond=None)[0][1:]

        x1_centered = x1 - x1.mean()
        short_slope = (x1_centered @ (y - y.mean())) / (x1_centered @ x1_centered)
        estimates[replication] = [full_slopes[0], full_slopes[1], short_slope]

    return estimates


def summarize_simulation(rho, estimates):
    slope_sum = estimates[:, 0] + estimates[:, 1]
    return {
        "rho": rho,
        "mean_full_b1": estimates[:, 0].mean(),
        "sd_full_b1": estimates[:, 0].std(ddof=1),
        "mean_full_b2": estimates[:, 1].mean(),
        "sd_full_b2": estimates[:, 1].std(ddof=1),
        "sd_full_sum": slope_sum.std(ddof=1),
        "corr_full_b1_b2": np.corrcoef(estimates[:, 0], estimates[:, 1])[0, 1],
        "mean_short_b1": estimates[:, 2].mean(),
        "sd_short_b1": estimates[:, 2].std(ddof=1),
    }


weak = simulate_design(0.2)
strong = simulate_design(0.99)
simulation_summary = pd.DataFrame(
    [summarize_simulation(0.2, weak), summarize_simulation(0.99, strong)]
)

print(simulation_summary.to_string(index=False))
simulation_summary.to_csv(PATH_OUTPUT / "lab3_simulation_summary.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

axes[0].hist(weak[:, 0], bins=35, density=True, alpha=0.65, label=r"$\rho=0.2$")
axes[0].hist(strong[:, 0], bins=35, density=True, alpha=0.55, label=r"$\rho=0.99$")
axes[0].axvline(1, color="#b04e16", linewidth=2, label="Partial effect")
axes[0].set(xlabel=r"Full-model $\hat\beta_1$", ylabel="Density")
axes[0].legend(frameon=False)

axes[1].scatter(strong[:, 0], strong[:, 1], alpha=0.18, s=9)
line = np.linspace(strong[:, 0].min(), strong[:, 0].max(), 100)
axes[1].plot(line, 2 - line, color="#b04e16", linewidth=2)
axes[1].set(xlabel=r"Full-model $\hat\beta_1$", ylabel=r"Full-model $\hat\beta_2$")

axes[2].hist(strong[:, 0], bins=35, density=True, alpha=0.55, label="Full model")
axes[2].hist(strong[:, 2], bins=35, density=True, alpha=0.55, label="Short model")
axes[2].axvline(1, color="#b04e16", linewidth=2, label="Partial effect 1")
axes[2].axvline(1.99, color="#287c8e", linewidth=2, label="Short target 1.99")
axes[2].set(xlabel=r"Estimated coefficient on $x_1$", ylabel="Density")
axes[2].legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(PATH_OUTPUT / "lab3_multicollinearity.png", dpi=180)
plt.close(fig)

# WRITE L3.1: Compare the centres and sampling standard deviations of the full
# and short estimates. Explain why high collinearity does not bias the full
# estimate, why the full slopes move in opposite directions, and why the short
# estimate is precise around the wrong target.
# ANSWER L3.1:
# The full-model b1 remains centred near 1 for rho=0.2 and 0.99, but its SD
# grows from 0.0841 to 0.5946; high collinearity raises variance, not bias.
# Full b1 and b2 are strongly negatively correlated at rho=0.99 (-0.9901), so
# their sum is relatively stable (SD 0.0836). The short slope is centred near
# 1+rho (1.2 or 1.99); at rho=0.99 it is precise (SD 0.0841) but biased for
# the partial effect 1. Dropping x2 does not solve the causal problem.





print("\nSimulation outputs saved in", PATH_OUTPUT)
print("\nAll TODO calculations completed; written interpretations are documented in comments and the LaTeX report.")
