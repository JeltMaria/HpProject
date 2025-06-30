HpProject
API на базе FastAPI и фронтенд на React для поддержки планирования беременности, включая аутентификацию пользователей, создание карт беременности, генерацию изображений детей и обработку медицинских документов с помощью ИИ.
Структура проекта

backend/: FastAPI-приложение с эндпоинтами для аутентификации, управления картами беременности, генерации изображений и обработки медицинских документов.
frontend/: React-фронтенд для взаимодействия с API.
docker-compose.yaml: Конфигурация Docker для запуска бэкенда, фронтенда и базы данных PostgreSQL.
requirements.txt: Зависимости Python.

Инструкции по установке

Клонируйте репозиторий:
git clone <repository-url>
cd HpProject


Установите зависимости:

Бэкенд:pip install -r backend/requirements.txt


Фронтенд:cd frontend
npm install




Настройте переменные окружения:

Создайте файл .env в директории backend/ с содержимым:DATABASE_URL=postgresql://hp_user:1234@db:5432/hp_db
GOOGLE_API_KEY=<your-google-api-key>
HUGGINGFACE_TOKEN=токен HF




Запустите с помощью Docker:
docker-compose up --build


Доступ к приложению:

API бэкенда: http://localhost:8000
Фронтенд: http://localhost:3000
Документация API: http://localhost:8000/docs



Использование

Эндпоинты бэкенда:

POST /login-or-register: Регистрация или вход пользователя.
POST /create-pregnancy-card: Создание карты беременности.
GET /pregnancy-card/{user_id}: Получение данных карты беременности.
POST /generate-child-image: Генерация изображения ребенка на основе фотографий родителей.
POST /dialog: Обработка текстовых запросов и медицинских изображений с ИИ.
GET /model_status: Проверка состояния моделей ИИ.


Фронтенд:

Регистрация/вход пользователей.
Создание и просмотр карт беременности.
Генерация изображений детей.
Интерактивный диалог с ИИ для анализа медицинских документов.



Примечания

Убедитесь, что база данных PostgreSQL hp_db инициализирована с таблицами users и pregnancy_card.
Файл copy.txt должен быть размещен в /home/user/HpProject/documents/ внутри контейнера бэкенда для работы RAG.
Для работы генерации изображений требуется доступ к модели stabilityai/stable-diffusion-xl-base-1.0 и google/medgemma-4b-it через Hugging Face.
