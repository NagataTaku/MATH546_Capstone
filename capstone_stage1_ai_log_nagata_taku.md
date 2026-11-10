# Capstone_Stage1 — AI Use Log
**Name:** Taku Nagata **NetID:** tn157 **Date:** 2026-11-09
## 1. Platforms and Models
- ChatGPT (GPT-5, web)
## 2. Agents and Tooling
ChatGPT was text-only; I copied code in and out myself.
## 3. Communication History
# Conversation Log

# AI Use Log

## Exchange 1

### User

I'm using XGBoost to predict the squared log-return of Bitcoin for 7 days from each date. Currently, the explanatory variables are the squared log-return for the past 7 days, the 7-day moving average, the 30-day moving average, and the price range from the previous day. I want to add a flag to the explanatory variables that changes depending on the time interval to the prediction. How should I edit the data to achieve this?

Please provide the answer in English, as concisely as possible.

### Assistant

Add a **forecast horizon (`h`)** variable to your dataset, where `h` ranges from 1 to 7.

For each date, create 7 rows with the same explanatory variables but different forecast horizons and target values.

```python
rows = []

for h in range(1, 8):
    df_h = df.copy()
    df_h["horizon"] = h
    df_h["target"] = df["squared_log_return"].shift(-h)
    rows.append(df_h)

df_long = pd.concat(rows).dropna()
```

The resulting dataset will look like this:

| Date | Lag 1 | Lag 2 | MA 7 | MA 30 | Range | Horizon | Target |
|---|---|---|---|---|---|---|---|
| Day 100 | ... | ... | ... | ... | ... | 1 | Day 101 |
| Day 100 | ... | ... | ... | ... | ... | 2 | Day 102 |
| Day 100 | ... | ... | ... | ... | ... | 3 | Day 103 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| Day 100 | ... | ... | ... | ... | ... | 7 | Day 107 |

You can then train **one XGBoost model** using `horizon` as an additional explanatory variable.

This approach, called *horizon-conditioned forecasting*, allows XGBoost to learn how the relationship between explanatory variables and squared log-returns changes across forecast horizons.

**Important:** Split the data chronologically before training, and ensure that all target dates in the training set precede the validation period to avoid data leakage.

---

## Exchange 2

### User

Using the format of the dataframe you provided, how should I display the data to clearly demonstrate that no data leakage has occurred?

### Assistant

To demonstrate that **no data leakage occurs**, display the forecast origin date, the latest date used for explanatory variables, and the target date.

### 1. Add date columns

```python
df_long["feature_end_date"] = df_long["date"]
df_long["target_date"] = (
    df_long["date"] + pd.to_timedelta(df_long["horizon"], unit="D")
)
```

### 2. Display the dataset

| Forecast origin | Feature end date | Horizon | Target date | Target |
|---|---|---|---|---|
| 2024-01-01 | 2024-01-01 | 1 | 2024-01-02 | \(r_{t+1}^2\) |
| 2024-01-01 | 2024-01-01 | 2 | 2024-01-03 | \(r_{t+2}^2\) |
| 2024-01-01 | 2024-01-01 | 7 | 2024-01-08 | \(r_{t+7}^2\) |
| 2024-01-02 | 2024-01-02 | 1 | 2024-01-03 | \(r_{t+1}^2\) |

Here, all explanatory variables must be available by the forecast origin date.

### 3. Verify no leakage

```python
assert (df_long["feature_end_date"] < df_long["target_date"]).all()
```

For chronological train-validation splitting, also verify:

```python
assert train["target_date"].max() < valid["date"].min()
```

### 4. Suggested explanation for your report

> To prevent look-ahead bias, all explanatory variables are constructed using information available at or before the forecast origin date. The forecast horizon ranges from 1 to 7 days, and the target variable is the squared log-return at the corresponding future date. The dataset explicitly records the latest information date and target date to verify their temporal ordering.

**Important:** These checks demonstrate correct date ordering, but you should also verify that moving averages and other features do not inadvertently incorporate future observations.

---

## Exchange 3

### User

For a set of graphs arranged in a 2x2 grid, I would like to add a title above the graphs in the first row and the second row, respectively.

### Assistant

Use `fig.text()` to add a centered title above each row of subplots.

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Your plots
# ...

# Row titles
fig.text(0.5, 0.97, "First Row Title",
         ha="center", fontsize=14, fontweight="bold")

fig.text(0.5, 0.50, "Second Row Title",
         ha="center", fontsize=14, fontweight="bold")

# Adjust spacing
fig.subplots_adjust(top=0.90, hspace=0.55)

plt.show()
```

- `0.5` (first argument): Horizontal center of the figure.
- `0.97` and `0.50`: Vertical positions of the row titles.
- `hspace`: Vertical spacing between subplot rows.

Adjust the vertical positions and `hspace` as needed to prevent overlapping.

---

## Exchange 4

### User

Please output the log of our exchange so far (including all my questions and your answers) in Markdown format. please output the format I can copy.

## 4. AI Audit
**Exchange 1**
- The AI ​​suggested adding an explanatory variable for the forecast horizon taking values ​​from 1 to 7 to represent the length of the forecast.
- According to Lecture 16, tree-based ML models partition observations based on the values of explanatory variables. Furthermore, unlike linear regression models, they do not assign a single coefficient to each explanatory variable. Therefore, these models can distinguish between 1-day-ahead and 7-day-ahead predictions using the forecast horizon as a numerical feature, without requiring dummy variables.
- Accepted

**Exchange 2**
- The AI suggested adding feature_end_date and target_date columns to verify that the explanatory variables were based on information available before the target date and that training targets did not extend into the validation period.
- I decided not to add the feature_end_date column because the latest date used for constructing the explanatory variables was already represented by the DataFrame index. Furthermore, target_date was defined as the index plus target_horizon, ensuring that the target date was later than the forecast origin. I also checked the feature construction process to ensure that no future observations were used. Therefore, I considered the additional feature_end_date check redundant. However, I accepted the target_date column because it explicitly identifies the date being predicted. I also accepted the check comparing training target dates with the start of the validation period to confirm that all training targets remained within the designated training period.
- Partially accepted

**Exchange 3**
- The AI suggested using `fig.text` and `fig.subplots_adjust` function to add a title above the graphs.

- Regarding `fig.text`, I consulted the official Matplotlib documentation and learned that it is a function used to add text directly to a figure. I also understood that the first two parameters specify the coordinates for the text's placement within the figure, so I made some minor adjustments to improve the visual layout.
As for the `fig.subplots_adjust` function, I checked the official Matplotlib documentation and found that it is used to adjust the layout such as the margins of subplots within a figure. Consequently, I changed its parameters to eliminate unnecessary white space.

- Accepted
