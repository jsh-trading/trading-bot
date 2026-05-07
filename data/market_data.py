# market_data.py
# This script downloads daily stock price data from the internet
# and saves it into a local database file on your computer.

# --- Imports: bring in tools we need ---
import yfinance as yf       # yfinance talks to Yahoo Finance to get stock prices
import sqlite3              # sqlite3 lets us create and use a local database file
import os                   # os helps us work with file paths and folders
from watchlist import WATCHLIST  # Import the central list of tickers we want to track

# How far back to download data.
# "1y" = 1 year, "6mo" = 6 months, "2y" = 2 years, etc.
PERIOD = "1y"

# Path to the database file. It will be created automatically if it doesn't exist.
# __file__ is the path of this script; we place the .db file in the same folder.
DB_PATH = os.path.join(os.path.dirname(__file__), "trading_bot.db")


def create_table(conn):
    """
    Creates the database table that will hold price data,
    but only if it doesn't already exist.

    A table is like a spreadsheet with rows (one per day) and columns
    (ticker, date, open, high, low, close, volume).
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_prices (
            ticker  TEXT,       -- Stock symbol, e.g. "AAPL"
            date    TEXT,       -- Date in YYYY-MM-DD format
            open    REAL,       -- Price at market open
            high    REAL,       -- Highest price of the day
            low     REAL,       -- Lowest price of the day
            close   REAL,       -- Price at market close
            volume  INTEGER,    -- Number of shares traded that day
            PRIMARY KEY (ticker, date)  -- Each ticker+date combo must be unique
        )
    """)
    conn.commit()


def fetch_and_save(ticker, conn):
    """
    Downloads price data for one ticker symbol and saves it to the database.
    Skips rows that are already saved so you don't get duplicates.
    """
    print(f"  Downloading data for {ticker}...")

    # Download the historical data using yfinance.
    # Using .Ticker().history() returns a simple single-level table,
    # which avoids a multi-column issue in newer versions of yfinance.
    df = yf.Ticker(ticker).history(period=PERIOD)

    if df.empty:
        print(f"  WARNING: No data returned for {ticker}. Skipping.")
        return

    # df is a table of rows (each row = one trading day)
    # We loop through each row and save it to the database
    rows_added = 0
    for date, row in df.iterrows():
        date_str = str(date.date())  # Convert the date to a simple "YYYY-MM-DD" string

        # Try to insert this row. If ticker+date already exists, do nothing (OR IGNORE).
        result = conn.execute("""
            INSERT OR IGNORE INTO daily_prices (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker,
            date_str,
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"]),
        ))
        rows_added += result.rowcount

    conn.commit()  # Save all changes to disk
    print(f"  Saved {rows_added} new rows for {ticker}.")


def main():
    """
    Main function: sets everything up and runs the download for each ticker.
    """
    print(f"Database will be saved to: {DB_PATH}\n")

    # Open (or create) the database file
    conn = sqlite3.connect(DB_PATH)

    # Make sure the table exists
    create_table(conn)

    # Loop through each ticker and download its data
    for ticker in WATCHLIST:
        fetch_and_save(ticker, conn)

    # Close the database connection when done
    conn.close()
    print("\nAll done! Data has been saved to trading_bot.db.")


# This means: only run main() if we run THIS file directly.
# If another script imports this file, main() won't run automatically.
if __name__ == "__main__":
    main()
