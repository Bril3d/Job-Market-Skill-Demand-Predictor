import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
});

export const getHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const getMetrics = async () => {
  try {
    const response = await api.get('/metrics');
    return response.data;
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return null;
  }
};

export const getInsights = async () => {
  try {
    const response = await api.get('/insights');
    return response.data;
  } catch (error) {
    console.error("Error fetching insights:", error);
    return null;
  }
};

export const predictSalary = async (jobData) => {
  const response = await api.post('/predict', jobData);
  return response.data;
};

export default api;
