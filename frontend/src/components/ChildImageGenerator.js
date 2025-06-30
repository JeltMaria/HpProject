import React, { useState } from 'react';
import axios from 'axios';

const ChildImageGenerator = () => {
  const [formData, setFormData] = useState({ parent1_image: null, parent2_image: null, age_category: 'child' });
  const [image, setImage] = useState(null);
  const [message, setMessage] = useState('');

  const handleFileChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.files[0] });
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();
    data.append('parent1_image', formData.parent1_image);
    data.append('parent2_image', formData.parent2_image);
    data.append('age_category', formData.age_category);
    try {
      const response = await axios.post('http://localhost:8000/api/generate-child-image', data, {
        responseType: 'blob'
      });
      setImage(URL.createObjectURL(response.data));
      setMessage('Изображение успешно сгенерировано');
    } catch (error) {
      setMessage(error.response?.data?.detail || 'Ошибка при генерации изображения');
    }
  };

  return (
    <div className="border p-4 rounded">
      <h2 className="text-xl font-semibold mb-2">Генерация изображения ребенка</h2>
      <form onSubmit={handleSubmit} className="space-y-2">
        <input
          type="file"
          name="parent1_image"
          accept="image/*"
          onChange={handleFileChange}
          className="border p-2 w-full"
        />
        <input
          type="file"
          name="parent2_image"
          accept="image/*"
          onChange={handleFileChange}
          className="border p-2 w-full"
        />
        <select
          name="age_category"
          value={formData.age_category}
          onChange={handleChange}
          className="border p-2 w-full"
        >
          <option value="toddler">Малыш (0-4 года)</option>
          <option value="child">Ребенок (5-12 лет)</option>
          <option value="teen">Подросток (13-18 лет)</option>
          <option value="adult">Взрослый (18+)</option>
        </select>
        <button type="submit" className="bg-blue-500 text-white p-2 rounded w-full">
          Сгенерировать изображение
        </button>
      </form>
      {message && <p className="mt-2">{message}</p>}
      {image && <img src={image} alt="Generated Child" className="mt-2 max-w-full h-auto" />}
    </div>
  );
};

export default ChildImageGenerator;