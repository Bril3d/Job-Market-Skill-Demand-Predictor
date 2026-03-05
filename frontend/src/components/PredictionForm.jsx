import { useState } from 'react';
import { predictSalary } from '../services/api';
import { Brain, Loader2, Sparkles, Target, Clock, Download } from 'lucide-react';

const PredictionForm = ({ darkMode }) => {
  const [formData, setFormData] = useState({
    title: '',
    seniority: 'Mid-Level',
    category: 'Backend',
    geo_tier: 'Tier 1',
    tags: ''
  });

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await predictSalary(formData);
      setResult(data);
      setHistory(prev => [{ ...data, title: formData.title, timestamp: new Date().toLocaleTimeString() }, ...prev].slice(0, 10));
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  const exportHistory = () => {
    if (history.length === 0) return;
    const rows = [["Title", "Prediction", "Probability", "Threshold", "Time"]];
    history.forEach(h => rows.push([h.title, h.label, h.probability.toFixed(4), h.threshold.toFixed(4), h.timestamp]));
    const csv = rows.map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "prediction_history.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const cardBg = darkMode ? 'glass-card' : 'bg-white rounded-2xl p-6 shadow-md border border-gray-100';
  const inputBg = darkMode ? 'input-field' : 'bg-gray-50 border border-gray-200 text-gray-900 px-4 py-3 rounded-lg w-full mb-4 outline-none focus:border-sky-400 transition-colors';

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form Section */}
        <div className={cardBg}>
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <Brain className="text-sky-400" /> Salary Predictor
          </h2>
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className={`text-sm mb-2 block ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Job Title</label>
                <input 
                  type="text" 
                  placeholder="e.g. Senior ML Engineer"
                  className={inputBg}
                  value={formData.title}
                  onChange={(e) => setFormData({...formData, title: e.target.value})}
                  required
                />
              </div>
              <div>
                <label className={`text-sm mb-2 block ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Category</label>
                <select className={inputBg} value={formData.category} onChange={(e) => setFormData({...formData, category: e.target.value})}>
                  <option>Backend</option>
                  <option>Data Science</option>
                  <option>ML Engineering</option>
                  <option>DevOps</option>
                  <option>Frontend</option>
                  <option>Management</option>
                  <option>Others</option>
                </select>
              </div>
              <div>
                <label className={`text-sm mb-2 block ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Seniority</label>
                <select className={inputBg} value={formData.seniority} onChange={(e) => setFormData({...formData, seniority: e.target.value})}>
                  <option>Junior</option>
                  <option>Mid-Level</option>
                  <option>Senior</option>
                  <option>Staff/Principal</option>
                </select>
              </div>
              <div>
                <label className={`text-sm mb-2 block ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Geo Tier</label>
                <select className={inputBg} value={formData.geo_tier} onChange={(e) => setFormData({...formData, geo_tier: e.target.value})}>
                  <option>Tier 1</option>
                  <option>Tier 2</option>
                </select>
              </div>
            </div>
            <div className="mt-4">
              <label className={`text-sm mb-2 block ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>Tags (comma separated)</label>
              <textarea 
                className={`${inputBg} h-24 resize-none`}
                placeholder="python, pytorch, aws, kubernetes..."
                value={formData.tags}
                onChange={(e) => setFormData({...formData, tags: e.target.value})}
              />
            </div>
            <button 
              type="submit" 
              className="primary-btn w-full mt-6 flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={loading}
            >
              {loading ? <Loader2 className="animate-spin" size={20} /> : <Sparkles size={20} />}
              {loading ? 'Analyzing...' : 'Predict High Salary Potential'}
            </button>
          </form>
        </div>

        {/* Results Section */}
        <div className="flex flex-col gap-6">
          <div className={`${cardBg} flex-1 flex flex-col items-center justify-center text-center`}>
            {!result ? (
              <div className="py-12">
                <Target size={64} className="text-sky-500/20 mx-auto mb-4" />
                <h3 className={`text-xl ${darkMode ? 'text-slate-400' : 'text-gray-400'}`}>Awaiting Input</h3>
                <p className={`text-sm mt-2 ${darkMode ? 'text-slate-500' : 'text-gray-400'}`}>Enter job details to see AI predictions</p>
              </div>
            ) : (
              <div className="w-full">
                <div className={`text-6xl font-black mb-4 ${result.prediction === 1 ? 'text-emerald-400' : 'text-sky-400'}`}>
                  {result.prediction === 1 ? '✓ YES' : '✗ NO'}
                </div>
                <h3 className="text-2xl font-bold mb-2">{result.label}</h3>
                <div className={`flex items-center justify-center gap-2 mb-6 ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>
                  <Target size={18} /> Confidence: <span className={darkMode ? 'text-white' : 'text-gray-900'}>{(result.probability * 100).toFixed(1)}%</span>
                </div>
                
                {/* Confidence Meter */}
                <div className={`w-full max-w-xs h-3 rounded-full overflow-hidden mx-auto mb-6 ${darkMode ? 'bg-slate-800 border border-white/5' : 'bg-gray-200'}`}>
                  <div 
                    className={`h-full rounded-full transition-all duration-1000 ease-out ${result.prediction === 1 ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' : 'bg-gradient-to-r from-sky-500 to-sky-400'}`}
                    style={{ width: `${result.probability * 100}%` }}
                  ></div>
                </div>
                
                {result.experience_detected > 0 && (
                  <p className="text-sm text-amber-400 mb-2">
                    📋 Detected: {result.experience_detected}+ years experience requirement
                  </p>
                )}
                <p className={`text-sm italic ${darkMode ? 'text-slate-500' : 'text-gray-400'}`}>
                  Threshold: {result.threshold.toFixed(4)}
                </p>
              </div>
            )}
          </div>

          {/* Prediction History */}
          {history.length > 0 && (
            <div className={cardBg}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold flex items-center gap-2">
                  <Clock size={18} className="text-sky-400" /> Recent Predictions
                </h3>
                <button onClick={exportHistory} className={`text-xs flex items-center gap-1 ${darkMode ? 'text-slate-400 hover:text-white' : 'text-gray-500 hover:text-gray-900'}`}>
                  <Download size={14} /> Export
                </button>
              </div>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {history.map((h, i) => (
                  <div key={i} className={`flex items-center justify-between py-2 px-3 rounded-lg text-sm ${darkMode ? 'bg-slate-800/50' : 'bg-gray-50'}`}>
                    <span className="truncate max-w-[180px]">{h.title}</span>
                    <div className="flex items-center gap-3">
                      <span className={h.prediction === 1 ? 'text-emerald-400' : 'text-sky-400'}>
                        {(h.probability * 100).toFixed(0)}%
                      </span>
                      <span className={`text-xs ${darkMode ? 'text-slate-500' : 'text-gray-400'}`}>{h.timestamp}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;
