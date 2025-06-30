import React, { useState } from 'react';
import axios from 'axios';

const PregnancyCardForm = ({ userId }) => {
  const [formData, setFormData] = useState({
    full_name: '',
    date_of_birth: '',
    menstruation_start_age: '',
    current_pregnancy_last_menstruation_start: ''
  });
  const [message, setMessage] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/api/create-pregnancy-card', { data: { ...formData, user_id: userId } });
      setMessage(`Карта создана, ID: ${response.data.card_id}`);
    } catch (error) {
      setMessage(error.response?.data?.detail || 'Ошибка при создании карты');
    }
  };

  return (
    <div className="border p-4 rounded">
      <h2 className="text-xl font-semibold mb-2">Создать карту беременности</h2>
      <form onSubmit={handleSubmit} className="space-y-2">
        <input
          type="text"
          name="full_name"
          placeholder="ФИО"
          value={formData.full_name}
          onChange={handleChange}
          className="border p-2 w-full"
        />
        <input
          type="date"
          name="date_of_birth"
          placeholder="Дата рождения"
          value={formData.date_of_birth}
          onChange={handleChange}
          className="border p-2 w-full"
        />
        <input
          type="number"
          name="menstruation_start_age"
          placeholder="Возраст начала менструации"
          value={formData.menstruation_start_age}
          onChange={handleChange}
          className="border p-2 w-full"
        />
        <input
          type="date"
          name="current_pregnancy_last_menstruation_start"
          placeholder="Дата последней менструации"
          value={formData.current_pregnancy_last_menstruation_start}
          onChange={handleChange}
          className="border p-2 w-full"
        />
        <button type="submit" className="bg-blue-500 text-white p-2 rounded w-full">
          Создать карту
        </button>
      </form>
      {message && <p className="mt-2">{message}</p>}
    </div>
  );
};

export default PregnancyCardForm;