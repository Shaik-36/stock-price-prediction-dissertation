import React from "react";
import { Container, Paper, Typography, Grid, Box } from "@mui/material";

const Home = () => {
  return (
    <Container>
      <Paper elevation={3} sx={{ padding: 4, marginTop: 2, backgroundColor: "#1e1e1e", color: "#ffffff" }}>
        
        {/* Dissertation Information */}
        <Typography variant="h4" gutterBottom align="center">
          Optimizing Stock Price Prediction using LSTM & Sentiment Analysis
        </Typography>

        <Typography variant="h6" align="center" color="secondary">
          A Dissertation presented to the Faculty of Huddersfield University
        </Typography>

        <Typography variant="body2" align="center">
          In partial fulfillment of the requirements for the degree of  
          <strong> Master of Science (M.Sc.) in Computing</strong>
        </Typography>

        <Typography variant="body1" align="center" sx={{ marginTop: 1 }}>
          <strong>Author:</strong> Imamuddin Shaik &nbsp; | &nbsp; <strong>Huddersfield, UK</strong> &nbsp; | &nbsp; <strong>January 2024</strong>
        </Typography>

        {/* Email Address */}
        <Typography variant="body1" align="center" sx={{ marginTop: 1 }}>
          <strong>Email:</strong> <a href="mailto:imamshan369@gmail.com" style={{ color: "#4db6ac", textDecoration: "none" }}>imamshan369@gmail.com</a>
        </Typography>

        {/* Project Image Section */}
        <Box mt={4} mb={3} textAlign="center">
          <Typography variant="h5" gutterBottom>Project Visualization</Typography>
          <img 
            src="https://github.com/Shaik-36/stock-price-prediction-dissertation/blob/main/Stock%20Price%20Prediction%20Project%20-%20Image.jpg?raw=true" 
            alt="Stock Price Prediction Project"
            style={{ width: "100%", maxWidth: "800px", borderRadius: "8px", boxShadow: "0px 4px 10px rgba(255,255,255,0.2)" }}
          />
          <Typography variant="body2" color="secondary" sx={{ marginTop: 1 }}>
            A visual representation of the Stock Price Prediction Project.
          </Typography>
        </Box>

        {/* Abstract */}
        <Box mt={4} mb={3}>
          <Typography variant="h5" gutterBottom>Abstract</Typography>
          <Typography variant="body1">
            Predicting stock prices is challenging due to dynamic market forces and investor sentiment. 
            This research integrates <strong>Long Short-Term Memory (LSTM) models</strong> with <strong>sentiment analysis</strong> 
            of <strong>100,000+ tweets per stock</strong> from <strong>Dow 30 stocks</strong> to improve prediction accuracy. 
            Sentiment scores are generated using <strong>VADER and TextBlob</strong>, then combined with financial data 
            from <strong>Yahoo Finance</strong>. Our model significantly enhances stock prediction, particularly for 
            <strong>McDonald's (MCD)</strong> stock, increasing <strong>R² from 0.821 to 0.916</strong>.
          </Typography>
        </Box>

        {/* Research Methodology */}
        <Box mt={3} mb={3}>
          <Typography variant="h5" gutterBottom>Methodology</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Typography variant="h6" color="primary">📊 Data Sources</Typography>
              <ul>
                <li>Stock market data from <strong>Yahoo Finance</strong>.</li>
                <li>Twitter sentiment data with over <strong>100,000 tweets per stock</strong>.</li>
                <li>Sentiment analysis applied using <strong>VADER & TextBlob</strong>.</li>
              </ul>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="h6" color="primary">🛠 Model & Techniques</Typography>
              <ul>
                <li>Data Cleaning, Tokenization, and Feature Extraction.</li>
                <li>Training <strong>LSTM deep learning models</strong> with sentiment scores.</li>
                <li>Hyperparameter tuning for <strong>best predictive accuracy</strong>.</li>
              </ul>
            </Grid>
          </Grid>
        </Box>

        {/* Research Objectives */}
        <Box mt={3} mb={3}>
          <Typography variant="h5" gutterBottom>Research Objectives</Typography>
          <ul>
            <li>Analyze <strong>Twitter sentiment's impact</strong> on stock prices.</li>
            <li>Enhance stock market prediction using <strong>LSTM & NLP techniques</strong>.</li>
            <li>Compare model performance with and without <strong>sentiment data</strong>.</li>
            <li>Optimize <strong>large-scale text processing</strong> for financial analytics.</li>
          </ul>
        </Box>

        {/* Key Findings */}
        <Box mt={3} mb={3}>
          <Typography variant="h5" gutterBottom>Key Findings</Typography>
          <Grid container spacing={3} mt={2}>
            <Grid item xs={12} sm={6}>
              <Paper elevation={3} sx={{ padding: 3, backgroundColor: "#252525" }}>
                <Typography variant="h6" color="secondary">📈 Model Accuracy</Typography>
                <ul>
                  <li><strong>McDonald's (MCD)</strong>: R² improved from <strong>0.821 to 0.916</strong>.</li>
                  <li><strong>IBM</strong>: MSE significantly reduced.</li>
                  <li><strong>Procter & Gamble (PG)</strong>: Achieved R² of <strong>0.914</strong>.</li>
                  <li><strong>Nike (NKE)</strong>: RMSE reduced to <strong>1.68</strong>, ensuring higher accuracy.</li>
                </ul>
              </Paper>
            </Grid>
          </Grid>
        </Box>

        {/* Conclusion */}
        <Box mt={3} mb={3}>
          <Typography variant="h5" gutterBottom>Conclusion</Typography>
          <Typography variant="body1">
            This dissertation demonstrates that <strong>machine learning models combined with sentiment analysis</strong> 
            significantly improve stock market predictions. By integrating <strong>financial and social media data</strong>, 
            this research provides valuable insights for traders and investors.
          </Typography>
        </Box>

      </Paper>
    </Container>
  );
};

export default Home;
