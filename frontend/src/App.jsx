import React, { useState, useEffect } from 'react';
import PredictionForm from './components/PredictionForm';
import Dashboard from './components/Dashboard';

const App = () => {
  const [activeTab, setActiveTab] = useState('predictor');
  const [darkMode, setDarkMode] = useState(true);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-[#0f1014] text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      {/* Header */}
      <nav className="border-b border-gray-800/50 backdrop-blur-xl sticky top-0 z-50 bg-opacity-80">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg bg-gradient-to-tr ${darkMode ? 'from-blue-600 to-cyan-500' : 'from-blue-500 to-indigo-600'}`}>
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <span className="text-xl font-bold tracking-tight">
              SkillDemand <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">AI</span>
            </span>
          </div>

          <div className="flex items-center gap-8">
            <div className="flex gap-1 p-1 bg-gray-900/50 rounded-xl border border-gray-800">
              <button
                onClick={() => setActiveTab('predictor')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'predictor' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-gray-400 hover:text-white'}`}
              >
                Predictor
              </button>
              <button
                onClick={() => setActiveTab('insights')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'insights' ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-gray-400 hover:text-white'}`}
              >
                Insights
              </button>
            </div>

            <button
              onClick={() => setDarkMode(!darkMode)}
              className="p-2.5 rounded-xl border border-gray-800 hover:bg-gray-800/50 transition-colors"
            >
              {darkMode ? '☀️' : '🌙'}
            </button>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        {activeTab === 'predictor' ? <PredictionForm darkMode={darkMode} /> : <Dashboard darkMode={darkMode} />}
      </main>

      <footer className="mt-auto border-t border-gray-800/30 py-8 px-6 text-center text-sm text-gray-500">
        <p>© 2026 SkillDemand AI Engine • Advanced Labor Market Analytics</p>
      </footer>
    </div>
  );
};

export default App;
