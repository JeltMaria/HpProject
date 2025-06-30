import React, { useState } from 'react';
import axios from 'axios';

const Login = ({ setUser }) => {
  const [formData, setFormData] = useState({ username: '', password: '' });
  const [message, setMessage] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/api/login-or-register', formData);
      setMessage(response.data.message);
      if (response.data.message === 'User created successfully') {
        setUser({ id: formData.username, username: formData.username });
      }
    } catch (error) {
      setMessage(error.response?.data?.detail || 'Ошибка при входе/регистрации');
    }
  };

  return (
    <div className="border p-4 rounded max-w-md mx-auto">
      <h2 className="text-xl font-semibold mb-2">Вход/ Vision AI</h2>
      <form onSubmit={handleSubmit} className="space-y-2">
        <input
          type="text"
          name="username"
          placeholder="Имя пользователя"
          value={formData.username}
          onChange={handleChange}
          className="border p-2 w-full"
        />
        <input
          type="password"
          name="password"
          placeholder="Пароль"
          value={formData.password}
          onChange={handleChange}
          className="border p-2 w-full"
        />
        <button type="submit" className="bg-blue-500 text-white p-2 rounded w-full">
          Войти/Зарегистрироваться
        </button>
      </form>
      {message && <p className="mt-2">{message}</p>}
    </div>
  );
};

export default Login;