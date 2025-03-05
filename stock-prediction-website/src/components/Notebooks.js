import React from 'react';
import { Container, Paper, Typography } from '@mui/material';

const Notebooks = () => {
  return (
    <Container>
      <Paper elevation={3} sx={{ padding: 3, marginTop: 2 }}>
        <Typography variant="h4" gutterBottom>Jupyter Notebook Output</Typography>
        <iframe src="/notebook.html" width="100%" height="600px" title="Notebook Output" />
      </Paper>
    </Container>
  );
};

export default Notebooks;
