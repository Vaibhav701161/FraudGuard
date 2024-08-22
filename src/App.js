import React from 'react';
import { BrowserRouter as Router, Route, Routes, NavLink } from 'react-router-dom';
import './App.css';
import Dashboard from './components/Dashboard';  

function Transactions() {
  return <h2>Transactions</h2>;
}

function Settings() {
  return <h2>Settings</h2>;
}

function App() {
  return (
    <Router>
      <div className="App">
        <header>
          <h1>FraudGuard</h1>
        </header>
        <nav>
          <ul>
            <li><NavLink to="/" end>Dashboard</NavLink></li>
            <li><NavLink to="/transactions">Transactions</NavLink></li>
            <li><NavLink to="/settings">Settings</NavLink></li>
          </ul>
        </nav>
        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/transactions" element={<Transactions />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;