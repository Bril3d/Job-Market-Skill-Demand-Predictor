import { useState } from 'react'
import Dashboard from './components/Dashboard'
import PredictionForm from './components/PredictionForm'
import { LayoutDashboard, Target, Moon, Sun, Github } from 'lucide-react'

function App() {
  const [activeTab, setActiveTab] = useState('insights')
  const [darkMode, setDarkMode] = useState(true)

  return (
    <div className={`min-h-screen transition-colors duration-300 ${darkMode ? 'bg-slate-950 text-white' : 'bg-gray-50 text-gray-900'}`}>
      {/* Navigation */}
      <nav className={`border-b sticky top-0 z-50 backdrop-blur-md ${darkMode ? 'border-white/5 bg-slate-900/80' : 'border-gray-200 bg-white/80'}`}>
        <div className="container mx-auto px-4 flex items-center justify-between py-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-sky-400 to-indigo-500 rounded-xl flex items-center justify-center shadow-lg shadow-sky-500/20">
              <LayoutDashboard className="text-white" size={20} />
            </div>
            <div>
              <span className="text-xl font-bold">SalaryGenius <span className="text-sky-400">AI</span></span>
              <span className="text-xs text-slate-500 ml-2 hidden sm:inline">v2.0</span>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className={`flex p-1 rounded-lg ${darkMode ? 'bg-slate-800/50' : 'bg-gray-100'}`}>
              <button 
                onClick={() => setActiveTab('insights')}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${activeTab === 'insights' ? 'bg-sky-500 text-white shadow-md shadow-sky-500/30' : darkMode ? 'text-slate-400 hover:text-white' : 'text-gray-500 hover:text-gray-900'}`}
              >
                <LayoutDashboard size={16} /> Insights
              </button>
              <button 
                onClick={() => setActiveTab('predictor')}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${activeTab === 'predictor' ? 'bg-sky-500 text-white shadow-md shadow-sky-500/30' : darkMode ? 'text-slate-400 hover:text-white' : 'text-gray-500 hover:text-gray-900'}`}
              >
                <Target size={16} /> Predictor
              </button>
            </div>

            <button 
              onClick={() => setDarkMode(!darkMode)}
              className={`p-2 rounded-lg transition-colors ${darkMode ? 'hover:bg-slate-800 text-slate-400' : 'hover:bg-gray-200 text-gray-600'}`}
              title={darkMode ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </div>
      </nav>

      <main className="min-h-[calc(100vh-120px)]">
        {activeTab === 'insights' ? <Dashboard darkMode={darkMode} /> : <PredictionForm darkMode={darkMode} />}
      </main>

      <footer className={`border-t py-6 mt-12 ${darkMode ? 'border-white/5' : 'border-gray-200'}`}>
        <div className="container mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p className={`text-sm ${darkMode ? 'text-slate-500' : 'text-gray-500'}`}>
            © 2026 SalaryGenius AI — FastAPI + React + ML Pipeline
          </p>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className={`text-sm flex items-center gap-1 ${darkMode ? 'text-slate-500 hover:text-white' : 'text-gray-500 hover:text-gray-900'}`}>
            <Github size={14} /> Source Code
          </a>
        </div>
      </footer>
    </div>
  )
}

export default App
