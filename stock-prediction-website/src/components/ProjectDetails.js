import React from "react";
import { Container, Paper, Typography, Grid, Box } from "@mui/material";

const ProjectDetails = () => {
  return (
    <Container>
      <Paper elevation={3} sx={{ padding: 4, marginTop: 2, backgroundColor: "#1e1e1e", color: "#ffffff" }}>
        
        {/* Page Title */}
        <Typography variant="h4" gutterBottom align="center">
          Stock Price Prediction - Project Details
        </Typography>

        {/* Project Image Section */}
        <Box mt={4} mb={3} textAlign="center">
          <Typography variant="h5" gutterBottom>Proposed Model for Stock Price Prediction</Typography>
          <img 
            src="https://github.com/Shaik-36/stock-price-prediction-dissertation/blob/main/Proposed%20Model%20-%20Stock%20Price%20Prediction.jpg?raw=true" 
            alt="Proposed Model - Stock Price Prediction"
            style={{ width: "100%", maxWidth: "800px", borderRadius: "8px", boxShadow: "0px 4px 10px rgba(255,255,255,0.2)" }}
          />
          <Typography variant="body2" color="secondary" sx={{ marginTop: 1 }}>
            The architecture of our proposed model for stock price prediction.
          </Typography>
        </Box>

        {/* Methodology Section */}
        <Box mt={4} mb={3}>
          <Typography variant="h5" gutterBottom>Methodology</Typography>
          <Typography variant="body1">
            The methodology employed in this study integrates <strong>time-series stock data</strong> with <strong>sentiment analysis</strong> from social media to enhance predictive accuracy. This section outlines the key steps taken in data collection, preprocessing, feature extraction, and model training.
          </Typography>

          <Typography variant="h6" color="primary">🔹 Step 1: Data Collection</Typography>
          <Typography variant="body1">
            Two primary datasets were used:
          </Typography>
          <ul>
            <li><strong>Stock Market Data:</strong> Collected using the <strong>Yahoo Finance API</strong> for companies such as McDonald's, IBM, Procter & Gamble, and Nike.</li>
            <li><strong>Twitter Sentiment Data:</strong> Over <strong>100,000+ stock-related tweets</strong> were gathered, covering discussions and investor sentiment.</li>
          </ul>

          <Typography variant="h6" color="primary">🔹 Step 2: Sentiment Analysis</Typography>
          <Typography variant="body1">
            Sentiment scores were assigned to each tweet using:
          </Typography>
          <ul>
            <li><strong>VADER (Valence Aware Dictionary and sEntiment Reasoner):</strong> A rule-based model for analyzing textual sentiment.</li>
            <li><strong>TextBlob:</strong> A natural language processing library used to compute polarity and subjectivity.</li>
          </ul>
          <Typography variant="body1">
            Each tweet was categorized into <strong>positive, negative, or neutral</strong> sentiments based on its calculated sentiment score.
          </Typography>

          <Typography variant="h6" color="primary">🔹 Step 3: Stock Market Indicators</Typography>
          <Typography variant="body1">
            In addition to historical price data, <strong>technical indicators</strong> were computed to enhance model learning:
          </Typography>
          <ul>
            <li><strong>Moving Averages (MA_7, MA_20):</strong> Identifies stock trends.</li>
            <li><strong>Relative Strength Index (RSI):</strong> Measures stock momentum.</li>
            <li><strong>MACD (Moving Average Convergence Divergence):</strong> Captures market fluctuations.</li>
            <li><strong>Bollinger Bands:</strong> Identifies overbought or oversold stock conditions.</li>
          </ul>

          <Typography variant="h6" color="primary">🔹 Step 4: Data Preprocessing</Typography>
          <Typography variant="body1">
            The datasets were cleaned and transformed to ensure high-quality inputs for the LSTM model:
          </Typography>
          <ul>
            <li><strong>Normalization:</strong> Stock prices and sentiment scores were scaled between 0 and 1.</li>
            <li><strong>Time Series Windowing:</strong> Stock prices were grouped into time windows to enable LSTM to learn long-term dependencies.</li>
            <li><strong>Data Merging:</strong> Sentiment scores were combined with stock prices based on matching timestamps.</li>
          </ul>

          <Typography variant="h6" color="primary">🔹 Step 5: Model Training</Typography>
          <Typography variant="body1">
            The processed data was fed into an <strong>LSTM deep learning model</strong> trained to predict future stock prices. The model used:
          </Typography>
          <ul>
            <li><strong>50 LSTM units:</strong> Captures sequential dependencies in stock data.</li>
            <li><strong>Dropout Layer (0.2):</strong> Prevents overfitting.</li>
            <li><strong>Dense Layer:</strong> Produces the final predicted stock price.</li>
            <li><strong>Adam Optimizer:</strong> Used for efficient learning.</li>
            <li><strong>Mean Squared Error (MSE):</strong> Used as the loss function.</li>
          </ul>
        </Box>

        {/* Model Performance */}
        <Box mt={4} mb={3}>
          <Typography variant="h5" gutterBottom>Model Performance & Accuracy</Typography>
          <Typography variant="body1">
            The LSTM model showed significant improvements in stock price prediction compared to traditional models.
          </Typography>

          <Grid container spacing={3} mt={2}>
            <Grid item xs={12} sm={6}>
              <Paper elevation={3} sx={{ padding: 3, backgroundColor: "#252525" }}>
                <Typography variant="h6" color="secondary">📈 Model Accuracy</Typography>
                <ul>
                  <li><strong>McDonald's (MCD):</strong> R² improved from <strong>0.821 to 0.916</strong>.</li>
                  <li><strong>IBM:</strong> MSE significantly reduced.</li>
                  <li><strong>Procter & Gamble (PG):</strong> Achieved R² of <strong>0.914</strong>.</li>
                  <li><strong>Nike (NKE):</strong> RMSE reduced to <strong>1.68</strong>, ensuring higher accuracy.</li>
                </ul>
              </Paper>
            </Grid>
          </Grid>
        </Box>

        {/* Conclusion */}
        <Box mt={3} mb={3}>
          <Typography variant="h5" gutterBottom>Conclusion</Typography>
          <Typography variant="body1">
            The integration of <strong>sentiment analysis</strong> and <strong>LSTM deep learning</strong> significantly improved stock price prediction accuracy. The results show that market sentiment, as measured by social media discussions, has a tangible impact on stock trends.
          </Typography>

          <Typography variant="h6" color="primary">🔷 Future Improvements</Typography>
          <ul>
            <li>Enhancing the model with <strong>real-time sentiment tracking</strong>.</li>
            <li>Exploring <strong>Transformer-based NLP models</strong> (BERT, GPT) for deeper analysis.</li>
            <li>Integrating <strong>Reinforcement Learning</strong> to improve prediction-based trading strategies.</li>
          </ul>
        </Box>

      </Paper>
    </Container>
  );
};

export default ProjectDetails;
