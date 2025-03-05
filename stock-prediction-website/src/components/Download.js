import React from 'react';
import { Container, Paper, Typography, Button } from '@mui/material';

const Download = () => {
  return (
    <Container>
      <Paper elevation={3} sx={{ padding: 3, marginTop: 2 }}>
        <Typography variant="h4" gutterBottom>Download Project Files</Typography>
        <Button variant="contained" color="primary" href="/dissertation.pdf" download>
          Download Dissertation
        </Button>
        <Button variant="contained" color="secondary" href="/datasets.zip" download sx={{ marginLeft: 2 }}>
          Download Datasets
        </Button>
      </Paper>
    </Container>
  );
};

export default Download;
