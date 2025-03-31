# Heart Failure Prediction System

A machine learning-based system for predicting heart failure using the UCI Cleveland Heart Disease dataset.

## Project Structure

```
heart-ecg/
├── data/               # Dataset storage
├── model/             # ML model implementation
├── backend/           # Flask API
├── frontend/          # Web interface
└── utility/           # Utility scripts
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the dataset:
```bash
python utility/data_preparation.py
```

4. Train the model:
```bash
python utility/train_model.py
```

5. Run the application:
```bash
python backend/app.py
```

The application will be available at `http://localhost:5000`

## Features

- Data preprocessing and cleaning
- Random Forest Classifier with hyperparameter tuning
- Feature importance visualization
- RESTful API endpoints
- User-friendly web interface
- Comprehensive model evaluation metrics

## Model Parameters

The model takes 13 clinical parameters as input:
1. Age (years)
2. Sex (1: male, 0: female)
3. Chest Pain Type (1-4)
4. Resting Blood Pressure (mm Hg)
5. Serum Cholesterol (mg/dl)
6. Fasting Blood Sugar (> 120 mg/dl: 1, ≤ 120 mg/dl: 0)
7. Resting ECG Results (0-2)
8. Maximum Heart Rate Achieved
9. Exercise-Induced Angina (1: yes, 0: no)
10. ST Depression Induced by Exercise
11. Slope of Peak Exercise ST Segment (1-3)
12. Number of Major Vessels (0-3)
13. Thalassemia (3: normal, 6: fixed defect, 7: reversible defect)

## Model Performance

The model provides:
- Binary prediction (0: no heart disease, 1: heart disease)
- Probability score for heart disease
- Feature importance visualization
- Model evaluation metrics (accuracy, AUC, sensitivity, specificity)

## API Endpoints

- POST `/api/predict`: Get heart disease prediction
- GET `/api/feature-importance`: Get feature importance data

## Error Handling

The system includes comprehensive error handling for:
- Invalid input parameters
- Missing data
- Model loading errors
- API request failures
- Data validation

# Heart Disease Prediction Application

This application provides heart disease risk prediction based on various health metrics.

## Deployment Instructions

### Prerequisites
- Node.js (v14 or higher)
- npm (Node Package Manager)
- Python 3.8 or higher (for the backend)
- pip (Python Package Manager)

### Frontend Deployment

1. Install dependencies:
```bash
npm install
```

2. Configure environment variables:
- Copy `.env.example` to `.env`
- Update the variables as needed:
  - `PORT`: The port for the frontend server (default: 3000)
  - `API_URL`: The URL of your backend API

3. Build and start the application:
```bash
npm run deploy
```

The frontend will be available at `http://localhost:3000` (or your configured PORT)

### Backend Deployment

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend server:
```bash
python app.py
```

The backend API will be available at `http://localhost:5001`

### Production Deployment

For production deployment, you have several options:

1. **Heroku Deployment**:
   - Install Heroku CLI
   - Login to Heroku: `heroku login`
   - Create a new app: `heroku create your-app-name`
   - Push to Heroku: `git push heroku main`

2. **Docker Deployment**:
   - Build the Docker image: `docker build -t heart-prediction .`
   - Run the container: `docker run -p 3000:3000 heart-prediction`

3. **Manual VPS/Cloud Deployment**:
   - Set up a VPS (e.g., DigitalOcean, AWS EC2)
   - Install Node.js and Python
   - Clone the repository
   - Follow the frontend and backend deployment steps
   - Use PM2 or similar for process management
   - Set up Nginx as a reverse proxy

### Environment Variables

Create a `.env` file with the following variables:

```env
PORT=3000
API_URL=http://your-backend-url:5001
```

### Security Considerations

1. Enable HTTPS in production
2. Set up proper CORS configuration
3. Implement rate limiting
4. Add API authentication if needed
5. Regular security updates

### Monitoring

1. Set up logging (e.g., Winston, Morgan)
2. Monitor server health
3. Track API performance
4. Set up error tracking (e.g., Sentry)

## Support

For deployment issues or questions, please open an issue in the repository. 