import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';
const API_KEY = 'sg-dev-key-2026';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'X-API-Key': API_KEY,
    'Content-Type': 'application/json'
  }
});

export const predictDemand = async (jobData) => {
  // Convert comma separated tags to array
  const tags = typeof jobData.tags === 'string' 
    ? jobData.tags.split(',').map(t => t.strip()).filter(t => t.length > 0)
    : jobData.tags;
    
  const response = await api.post('/predict', { ...jobData, tags });
  return response.data;
};

export const getInsights = async () => {
  const response = await api.get('/insights');
  return response.data;
};

export const getMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

export const getHistory = async () => {
  const response = await api.get('/history');
  return response.data;
};

export default api;
