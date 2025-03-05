import React from 'react';
import { Link } from 'react-router-dom';
import { AppBar, Toolbar, Button, Typography } from '@mui/material';

const Navbar = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Stock Price Prediction
        </Typography>
        <Button color="inherit" component={Link} to="/">Home</Button>
        <Button color="inherit" component={Link} to="/project">Project Details</Button>
        <Button color="inherit" component={Link} to="/notebooks">Jupyter Outputs</Button>
        <Button color="inherit" component={Link} to="/download">Download</Button>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;
