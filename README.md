# Time Series Analysis: Stock Price Prediction using ARIMA, LSTM, and GRU

## 📊 Project Overview

This project implements and compares three different time series forecasting approaches to predict Google (GOOGL) stock prices:
- **ARIMA** - Traditional statistical model
- **LSTM** - Deep learning with Long Short-Term Memory networks
- **GRU** - Deep learning with Gated Recurrent Units

The analysis includes comprehensive exploratory data analysis (EDA), technical indicators, model training, evaluation, and comparison.

## 🎯 Objectives

- Analyze historical stock price patterns and trends
- Build multiple forecasting models using different approaches
- Compare model performance using various metrics
- Provide insights for stock price prediction

## 📁 Project Structure

```
.
├── improved_time_series_analysis.ipynb    # Main Jupyter notebook
├── README.md                              # Project documentation
└── requirements.txt                       # Python dependencies
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Financial Data**: yfinance
- **Statistical Models**: statsmodels
- **Deep Learning**: TensorFlow, Keras
- **Machine Learning**: scikit-learn

## 📦 Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn yfinance ta statsmodels tensorflow scikit-learn
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook improved_time_series_analysis.ipynb
```

2. Run cells sequentially from top to bottom

3. The notebook will:
   - Download Google stock data automatically
   - Perform exploratory data analysis
   - Calculate technical indicators
   - Train ARIMA, LSTM, and GRU models
   - Generate predictions and visualizations
   - Compare model performance

## 📈 Features

### 1. Exploratory Data Analysis
- Monthly average price analysis
- Time series visualization
- Distribution analysis with box plots
- Volume analysis over time

### 2. Technical Indicators
- Exponential Weighted Moving Averages (EWMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)

### 3. Statistical Modeling
- Stationarity testing with Augmented Dickey-Fuller test
- ACF and PACF analysis
- ARIMA model implementation

### 4. Deep Learning Models
- LSTM neural network with multiple layers
- GRU neural network architecture
- Early stopping and learning rate scheduling
- Dropout regularization

### 5. Model Evaluation
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² Score
- Residual analysis
- Visual comparison of predictions

## 📊 Visualizations

All visualizations include:
- Professional styling with custom color schemes
- Grid lines (major and minor) for better readability
- Enhanced legends with styled frames
- Skyblue borders and consistent formatting
- Clear titles and axis labels

## 🔍 Key Findings

- **ARIMA**: Effective for capturing linear trends and seasonality
- **LSTM**: Superior at learning complex, non-linear patterns
- **GRU**: Comparable performance to LSTM with faster training

Performance metrics vary based on the data and market conditions. The notebook provides detailed comparison charts.

## 📝 Model Descriptions

### ARIMA (AutoRegressive Integrated Moving Average)
A classical statistical method that combines autoregression, differencing, and moving averages. Uses historical patterns to forecast future values.

### LSTM (Long Short-Term Memory)
A deep learning architecture with memory cells that can learn long-term dependencies. Uses 60-day sequences to predict next-day prices.

### GRU (Gated Recurrent Unit)
A simplified LSTM variant with fewer parameters. Offers faster training while maintaining comparable accuracy.

## 📌 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
yfinance>=0.1.70
ta>=0.10.0
statsmodels>=0.13.0
tensorflow>=2.8.0
scikit-learn>=1.0.0
```

## 🔮 Future Improvements

- Incorporate external features (news sentiment, market indicators)
- Implement ensemble methods combining multiple models
- Add prediction intervals for uncertainty quantification
- Experiment with transformer-based architectures
- Include multivariate analysis with multiple stocks

## 📧 Contact

For questions or feedback about this project, please open an issue or contact the project maintainer.

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

- Data source: Yahoo Finance (via yfinance library)
- Inspired by financial time series analysis techniques
- Built with open-source libraries from the Python community

---

**Note**: This project is for educational purposes only. Stock price predictions should not be used as the sole basis for investment decisions. Always consult with financial professionals before making investment choices.
