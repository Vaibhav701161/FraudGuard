import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Grid, Paper, Typography, Button, List, ListItem, ListItemText, ListItemIcon, Switch, FormControlLabel, Chip } from '@material-ui/core';
import { Alert, AlertTitle } from '@material-ui/lab';
import { TrendingUp, TrendingDown, Warning } from '@material-ui/icons';

const initialTransactionData = [
  { name: 'Mon', transactions: 4000, fraudulent: 240 },
  { name: 'Tue', transactions: 3000, fraudulent: 139 },
  { name: 'Wed', transactions: 2000, fraudulent: 980 },
  { name: 'Thu', transactions: 2780, fraudulent: 390 },
  { name: 'Fri', transactions: 1890, fraudulent: 490 },
  { name: 'Sat', transactions: 2390, fraudulent: 350 },
  { name: 'Sun', transactions: 3490, fraudulent: 430 },
];

const initialRiskData = [
  { name: 'Low Risk', value: 400 },
  { name: 'Medium Risk', value: 300 },
  { name: 'High Risk', value: 300 },
];

const COLORS = ['#00C49F', '#FFBB28', '#FF8042'];

const initialAlerts = [
  { severity: 'high', message: 'Unusual large transaction of $50,000 detected' },
  { severity: 'medium', message: 'Multiple failed login attempts from IP 192.168.1.1' },
  { severity: 'low', message: 'New device used for account access' },
];

function Dashboard() {
  const [transactionData, setTransactionData] = useState(initialTransactionData);
  const [riskData, setRiskData] = useState(initialRiskData);
  const [recentAlerts, setRecentAlerts] = useState(initialAlerts);
  const [isMonitoring, setIsMonitoring] = useState(true);
  const [transactions, setTransactions] = useState([]);

  useEffect(() => {
    if (isMonitoring) {
      const interval = setInterval(() => {
        setTransactionData(prevData => {
          const newData = [...prevData];
          newData[6] = {
            ...newData[6],
            transactions: newData[6].transactions + Math.floor(Math.random() * 100),
            fraudulent: newData[6].fraudulent + Math.floor(Math.random() * 10)
          };
          return newData;
        });

        setRiskData(prevData => {
          return prevData.map(item => ({
            ...item,
            value: item.value + Math.floor(Math.random() * 50) - 25
          }));
        });

        if (Math.random() > 0.7) {
          setRecentAlerts(prevAlerts => [
            {
              severity: Math.random() > 0.5 ? 'high' : 'medium',
              message: `New suspicious activity detected at ${new Date().toLocaleTimeString()}`
            },
            ...prevAlerts.slice(0, 2)
          ]);
        }

        const newTransaction = {
          amount: Math.floor(Math.random() * 10000),
          merchant: `Merchant ${Math.floor(Math.random() * 100)}`,
          riskScore: Math.floor(Math.random() * 100),
          status: Math.random() > 0.8 ? 'Flagged' : 'Approved'
        };
        setTransactions(prev => [newTransaction, ...prev].slice(0, 10));
      }, 3000);

      return () => clearInterval(interval);
    }
  }, [isMonitoring]);

  const handleMonitoringToggle = () => {
    setIsMonitoring(!isMonitoring);
  };

  const simulateFraudulentActivity = () => {
    setTransactionData(prevData => {
      const newData = [...prevData];
      newData[6] = {
        ...newData[6],
        fraudulent: newData[6].fraudulent + 500
      };
      return newData;
    });

    setRecentAlerts(prevAlerts => [
      {
        severity: 'high',
        message: `Large fraudulent activity detected at ${new Date().toLocaleTimeString()}`
      },
      ...prevAlerts.slice(0, 2)
    ]);
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Alert severity="info">
          <AlertTitle>Welcome to FraudGuard Dashboard</AlertTitle>
          Real-time fraud detection and prevention system {isMonitoring ? 'active' : 'paused'}. 
          <FormControlLabel
            control={<Switch checked={isMonitoring} onChange={handleMonitoringToggle} />}
            label={isMonitoring ? 'Monitoring' : 'Paused'}
          />
        </Alert>
      </Grid>
      
      <Grid item xs={12} md={8}>
        <Paper elevation={3} className="dashboard-card">
          <Typography variant="h6" gutterBottom>Transaction Overview</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={transactionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="transactions" fill="#8884d8" name="Total Transactions" />
              <Bar dataKey="fraudulent" fill="#82ca9d" name="Fraudulent Transactions" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      <Grid item xs={12} md={4}>
        <Paper elevation={3} className="dashboard-card">
          <Typography variant="h6" gutterBottom>Risk Distribution</Typography>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskData}
                cx="50%"
                cy="50%"
                labelLine={false}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {riskData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} className="dashboard-card">
          <Typography variant="h6" gutterBottom>Recent Alerts</Typography>
          <List>
            {recentAlerts.map((alert, index) => (
              <ListItem key={index}>
                <ListItemIcon>
                  {alert.severity === 'high' ? <Warning color="error" /> : 
                   alert.severity === 'medium' ? <TrendingUp color="secondary" /> : 
                   <TrendingDown color="primary" />}
                </ListItemIcon>
                <ListItemText 
                  primary={alert.message} 
                  secondary={`Severity: ${alert.severity}`} 
                />
              </ListItem>
            ))}
          </List>
          <Button variant="contained" color="primary" onClick={simulateFraudulentActivity}>
            Simulate Fraudulent Activity
          </Button>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} className="dashboard-card">
          <Typography variant="h6" gutterBottom>Live Transaction Feed</Typography>
          <List className="alert-list">
            {transactions.map((transaction, index) => (
              <ListItem key={index} className={`alert-item ${transaction.status === 'Flagged' ? 'high' : 'low'}`}>
                <ListItemText
                  primary={`$${transaction.amount} - ${transaction.merchant}`}
                  secondary={`Risk Score: ${transaction.riskScore}`}
                />
                <Chip 
                  label={transaction.status} 
                  color={transaction.status === 'Approved' ? 'primary' : 'secondary'}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Dashboard;