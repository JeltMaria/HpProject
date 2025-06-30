import React, { useState } from 'react';
import axios from 'axios';

const Dialog = () => {
  const [message, setMessage] = useState('');
  const [image, setImage] = useState(null);
  const [response, setResponse] = useState('');
  const [action, setAction] = useState('dialog');

  const handleMessageChange = (e) => {
    setMessage(e.target.value);
  };

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setAction('upload_image');
  };

  const handleActionChange = (e) => {
    setAction(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();
    data.append('message', JSON.stringify({ action, content: message }));
    if (image) {
      data.append('image', image);
    }
    try {
      const response = await axios.post('http://localhost:8000/api/dialog', data);
      setResponse(response.data.response);
    } catch (error) {
      setResponse(error.response?.data?.detail || 'Ошибка при обработке запроса');
    }
  };

  return (
    <div className="border p-4 rounded">
      <h2 className="text-xl font-semibold mb-2">Диалог с ИИ</h2>
      <form onSubmit={handleSubmit} className="space-y-2">
        <select value={action} onChange={handleActionChange} className="border p-2 w-full">
          <option value="dialog">Обычный вопрос</option>
          <option value="important_question">Важный вопрос</option>
          <option value="upload_image">Загрузить изображение</option>
        </select>
        <textarea
          value={message}
          onChange={handleMessageChange}
          placeholder="Введите ваш вопрос или описание"
          className="border p-2 w-full h-24"
        />
        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="border p-2 w-full"
          disabled={action !== 'upload_image'}
        />
        <button type="submit" className="bg-blue-500 text-white p-2 rounded w-full">
          Отправить
        </button>
      </form>
      {response && (
        <div className="mt-2">
          <h3 className="font-semibold">Ответ:</h3>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
};

export default Dialog;