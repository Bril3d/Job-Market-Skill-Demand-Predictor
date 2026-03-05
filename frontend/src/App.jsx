import { useState } from 'react'
import Dashboard from './components/Dashboard'
import PredictionForm from './components/PredictionForm'
import { LayoutDashboard, Target } from 'lucide-react'

function App() {
  const [activeTab, setActiveTab] = useState('insights')

  return (
    <div className="min-h-screen">
      {/* Sidebar / Nav */}
      <nav className="border-b border-white/5 bg-slate-900/50 backdrop-blur-md sticky top-0 z-50">
        <div className="container flex items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 bg-sky-500 rounded-xl flex items-center justify-center">
              <LayoutDashboard className="text-white" />
            </div>
            <span className="text-xl font-bold">SalaryGenius <span className="text-sky-400">AI</span></span>
          </div>
          
          <div className="flex bg-slate-800/50 p-1 rounded-lg">
            <button 
              onClick={() => setActiveTab('insights')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${activeTab === 'insights' ? 'bg-sky-500 text-white' : 'text-slate-400 hover:text-white'}`}
            >
              <LayoutDashboard size={16} /> Insights
            </button>
            <button 
              onClick={() => setActiveTab('predictor')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${activeTab === 'predictor' ? 'bg-sky-500 text-white' : 'text-slate-400 hover:text-white'}`}
            >
              <Target size={16} /> Predictor
            </button>
          </div>
        </div>
      </nav>

      <main>
        {activeTab === 'insights' ? <Dashboard /> : <PredictionForm />}
      </main>

      <footer className="border-t border-white/5 py-8 mt-12">
        <div className="container text-center text-slate-500 text-sm">
          <p>© 2026 SalaryGenius AI. Built with FastAPI & React.</p>
        </div>
      </footer>
    </div>
  )
}

export default App
