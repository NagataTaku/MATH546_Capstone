# MATH546 Capstone Project

This project forecasts Bitcoin squared log returns over a 7-day horizon using Student-t GARCH, XGBoost, and a Naive Baseline.

## Project Structure

```text
Capstone/
├── capstone_support/        # Provided data and utilities
└── MATH546_Capstone/
    ├── capstone_stage1.ipynb  # Main Jupyter Notebook
    ├── capstone_stage1.html   # Output html
    ├── requirements.txt       # Python dependencies
    └── README.md
```

## How to Run

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Open `capstone_stage1.ipynb` in Jupyter Notebook.
3. Execute `Run All`


## Data Source

Bitcoin data are provided through the course's `capstone_support` package.

Original data sources and preparation details are available in
`https://finance.yahoo.com/quote/BTC-USD/history/`.