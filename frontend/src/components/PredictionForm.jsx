import React, { useState } from 'react';
import { predictDemand } from '../services/api';

const PredictionForm = ({ darkMode }) => {
  const [formData, setFormData] = useState({
    title: '',
    seniority: 'Mid-Level',
    category: 'Backend',
    geo_tier: 'Tier 1',
    tags: ''
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await predictDemand(formData);
      setResult(data);
      setHistory(prev => [{ ...data, ...formData, timestamp: new Date().toLocaleTimeString() }, ...prev].slice(0, 5));
    } catch (error) {
      console.error("Prediction failed", error);
    } finally {
      setLoading(false);
    }
  };

  const categories = ["Backend", "Frontend", "Data Engineering", "Data Science", "ML Engineering", "DevOps", "Management", "Others"];
  const seniorityLevels = ["Junior", "Mid-Level", "Senior", "Staff/Principal"];
  const tiers = ["Tier 1", "Tier 2"];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      {/* Form Section */}
      <div className="lg:col-span-7 bg-[#1a1c23] border border-gray-800/50 rounded-3xl p-8 shadow-2xl">
        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-2">Skill Demand Predictor</h2>
          <p className="text-gray-400">Deep analysis of tech stack demand and market relevance.</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-400 px-1">Job Title</label>
            <input
              type="text"
              required
              className="w-full bg-gray-900/50 border border-gray-700/50 rounded-2xl px-5 py-4 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 outline-none transition-all"
              placeholder="e.g. Senior Machine Learning Engineer"
              value={formData.title}
              onChange={e => setFormData({ ...formData, title: e.target.value })}
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-400 px-1">Seniority</label>
              <select
                className="w-full bg-gray-900/50 border border-gray-700/50 rounded-2xl px-5 py-4 focus:ring-2 focus:ring-blue-500/50 outline-none appearance-none"
                value={formData.seniority}
                onChange={e => setFormData({ ...formData, seniority: e.target.value })}
              >
                {seniorityLevels.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-400 px-1">Category</label>
              <select
                className="w-full bg-gray-900/50 border border-gray-700/50 rounded-2xl px-5 py-4 focus:ring-2 focus:ring-blue-500/50 outline-none appearance-none"
                value={formData.category}
                onChange={e => setFormData({ ...formData, category: e.target.value })}
              >
                {categories.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-400 px-1">Geographic Tier</label>
            <div className="flex gap-4 p-1 bg-gray-900/50 rounded-2xl border border-gray-700/50">
              {tiers.map(t => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setFormData({ ...formData, geo_tier: t })}
                  className={`flex-1 py-3 rounded-xl text-sm font-medium transition-all ${formData.geo_tier === t ? 'bg-gray-800 text-white shadow-lg' : 'text-gray-500 hover:text-gray-300'}`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-400 px-1">Technology Tags (Comma Separated)</label>
            <textarea
              className="w-full bg-gray-900/50 border border-gray-700/50 rounded-2xl px-5 py-4 h-32 focus:ring-2 focus:ring-blue-500/50 outline-none transition-all resize-none"
              placeholder="e.g. Python, PyTorch, AWS, Docker, NLP"
              value={formData.tags}
              onChange={e => setFormData({ ...formData, tags: e.target.value })}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-500 hover:to-cyan-400 text-white py-5 rounded-2xl font-bold text-lg shadow-xl shadow-blue-500/20 transition-all active:scale-[0.98] disabled:opacity-50"
          >
            {loading ? 'Analyzing Complexity...' : 'Generate Demand Prediction'}
          </button>
        </form>
      </div>

      {/* Result Section */}
      <div className="lg:col-span-5 space-y-6">
        {result ? (
          <div className="bg-gradient-to-br from-gray-900 to-black border border-blue-500/30 rounded-3xl p-8 shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/10 blur-3xl -mr-16 -mt-16 group-hover:bg-blue-500/20 transition-all"></div>
            
            <span className="text-blue-400 text-xs font-bold uppercase tracking-widest block mb-4">Analysis Complete</span>
            <h3 className="text-3xl font-bold mb-6">{result.label}</h3>
            
            <div className="space-y-6 mb-8">
              <div className="flex justify-between items-end">
                <span className="text-gray-400">Demand Confidence</span>
                <span className="text-2xl font-mono text-cyan-400">{(result.probability * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-blue-600 to-cyan-400 transition-all duration-1000 ease-out"
                  style={{ width: `${result.probability * 100}%` }}
                ></div>
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800/30 p-4 rounded-2xl border border-gray-700/30">
                  <span className="text-gray-500 text-xs block mb-1">Demand Score</span>
                  <span className="text-xl font-bold text-white">{result.demand_score.toFixed(2)}</span>
                </div>
                <div className="bg-gray-800/30 p-4 rounded-2xl border border-gray-700/30">
                  <span className="text-gray-500 text-xs block mb-1">Threshold</span>
                  <span className="text-xl font-bold text-white">{result.threshold.toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div className="p-4 bg-blue-500/5 rounded-2xl border border-blue-500/10 text-sm text-blue-200 leading-relaxed italic">
              "This tech stack represents a {result.prediction === 1 ? 'highly competitive' : 'stable'} demand profile in the current market environment."
            </div>
          </div>
        ) : (
          <div className="bg-[#1a1c23] border border-gray-800 border-dashed rounded-3xl p-12 text-center">
            <div className="w-16 h-16 bg-gray-800 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <span className="text-2xl">⚡</span>
            </div>
            <p className="text-gray-500">Enter job details to see the skill demand analysis.</p>
          </div>
        )}

        {/* Recent History Mini-Widget */}
        <div className="bg-[#1a1c23]/50 border border-gray-800/50 rounded-3xl p-6">
          <h4 className="font-bold mb-4 flex items-center gap-2">
            <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
            Recent Analysis
          </h4>
          <div className="space-y-3">
            {history.map((item, i) => (
              <div key={i} className="flex justify-between items-center p-3 hover:bg-gray-800/30 rounded-xl transition-colors cursor-default border border-transparent hover:border-gray-700/30">
                <div className="max-w-[70%]">
                  <p className="text-sm font-medium truncate">{item.title}</p>
                  <p className="text-[10px] text-gray-500 uppercase tracking-tighter">{item.timestamp} • {item.seniority}</p>
                </div>
                <span className={`px-2 py-1 rounded-md text-[10px] font-bold ${item.prediction === 1 ? 'bg-blue-500/10 text-blue-400' : 'bg-gray-700/20 text-gray-500'}`}>
                  {item.prediction === 1 ? 'HIGH' : 'STND'}
                </span>
              </div>
            ))}
            {history.length === 0 && <p className="text-xs text-gray-600 italic">No recent activity yet.</p>}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;
