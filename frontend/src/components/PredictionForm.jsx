import { useState } from 'react';
import { predictSalary } from '../services/api';
import { Brain, Loader2, Sparkles, Target } from 'lucide-react';

const PredictionForm = () => {
  const [formData, setFormData] = useState({
    title: '',
    seniority: 'Mid-Level',
    category: 'Backend',
    geo_tier: 'Tier 1',
    tags: ''
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      const data = await predictSalary(formData);
      setResult(data);
    } catch (error) {
      console.error("Prediction failed:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container py-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form Section */}
        <div className="glass-card">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <Brain className="text-sky-400" /> Salary Predictor
          </h2>
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="text-sm text-secondary mb-2 block">Job Title</label>
                <input 
                  type="text" 
                  placeholder="e.g. Senior ML Engineer"
                  className="input-field"
                  value={formData.title}
                  onChange={(e) => setFormData({...formData, title: e.target.value})}
                  required
                />
              </div>
              <div>
                <label className="text-sm text-secondary mb-2 block">Category</label>
                <select 
                  className="input-field"
                  value={formData.category}
                  onChange={(e) => setFormData({...formData, category: e.target.value})}
                >
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
                <label className="text-sm text-secondary mb-2 block">Seniority</label>
                <select 
                  className="input-field"
                  value={formData.seniority}
                  onChange={(e) => setFormData({...formData, seniority: e.target.value})}
                >
                  <option>Junior</option>
                  <option>Mid-Level</option>
                  <option>Senior</option>
                  <option>Staff/Principal</option>
                </select>
              </div>
              <div>
                <label className="text-sm text-secondary mb-2 block">Geo Tier</label>
                <select 
                  className="input-field"
                  value={formData.geo_tier}
                  onChange={(e) => setFormData({...formData, geo_tier: e.target.value})}
                >
                  <option>Tier 1</option>
                  <option>Tier 2</option>
                </select>
              </div>
            </div>
            <div className="mt-4">
              <label className="text-sm text-secondary mb-2 block">Tags (comma separated)</label>
              <textarea 
                className="input-field h-24"
                placeholder="python, pytorch, aws..."
                value={formData.tags}
                onChange={(e) => setFormData({...formData, tags: e.target.value})}
              />
            </div>
            <button 
              type="submit" 
              className="primary-btn w-full mt-6 flex items-center justify-center gap-2 disabled:opacity-50"
              disabled={loading}
            >
              {loading ? <Loader2 className="animate-spin" /> : <Sparkles size={20} />}
              {loading ? 'Analyzing...' : 'Predict High Salary Potential'}
            </button>
          </form>
        </div>

        {/* Prediction Results Section */}
        <div className="flex flex-col gap-6">
          <div className="glass-card flex-1 flex flex-col items-center justify-center text-center">
            {!result ? (
              <div className="py-12">
                <Target size={64} className="text-sky-500/20 mx-auto mb-4" />
                <h3 className="text-xl text-secondary">Awaiting Input</h3>
                <p className="text-sm text-secondary/60 mt-2">Enter job details to see AI predictions</p>
              </div>
            ) : (
              <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
                <div className={`text-6xl font-black mb-4 ${result.prediction === 1 ? 'text-emerald-400' : 'text-sky-400'}`}>
                   {result.prediction === 1 ? 'YES' : 'NO'}
                </div>
                <h3 className="text-2xl font-bold mb-2">{result.label}</h3>
                <div className="flex items-center justify-center gap-2 text-secondary mb-6">
                  <Target size={18} /> Confidence: <span className="text-white">{(result.probability * 100).toFixed(1)}%</span>
                </div>
                
                {/* Confidence Meter */}
                <div className="w-full max-w-xs bg-bg-accent h-3 rounded-full overflow-hidden mx-auto mb-8 border border-white/5">
                  <div 
                    className={`h-full transition-all duration-1000 ${result.prediction === 1 ? 'bg-emerald-500' : 'bg-sky-500'}`}
                    style={{ width: `${result.probability * 100}%` }}
                  ></div>
                </div>
                
                <p className="text-sm text-secondary italic">
                  Model decision threshold: {result.threshold.toFixed(4)}
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;
