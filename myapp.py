import yfinance as yf
import streamlit as st
import plotly.express as px

st.write("""
# Simple Stock Price App

Shown are the stock closing price and volume of Google!

""")

tickerSymbol = 'GOOGL'
tickerData = yf.Ticker(tickerSymbol)
tickerDF = tickerData.history(
    period='1d',
    start='2022-11-01',
    end='2022-11-18'
)

st.write(tickerDF.head(5)) 



