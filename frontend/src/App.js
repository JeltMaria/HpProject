import React, { useState } from 'react';
import Login from './components/Login';
import PregnancyCardForm from './components/PregnancyCardForm';
import ChildImageGenerator from './components/ChildImageGenerator';
import Dialog from './components/Dialog';
import './App.css';

const App = () => {
  const [user, setUser] = useState(null);

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">HpProject</h1>
      {!user ? (
        <Login setUser={setUser} />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <PregnancyCardForm userId={user.id} />
          <ChildImageGenerator />
          <Dialog />
        </div>
      )}
    </div>
  );
};

export default App;