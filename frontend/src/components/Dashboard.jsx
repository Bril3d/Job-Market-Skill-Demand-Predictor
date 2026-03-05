import React, { useState, useEffect } from 'react';
import { getInsights, getMetrics } from '../services/api';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, 
  BarChart, Bar, XAxis, YAxis, Tooltip, 
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  Legend
} from 'recharts';

const Dashboard = ({ darkMode }) => {
  const [insights, setInsights] = useState(null);
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const [i, m] = await Promise.all([getInsights(), getMetrics()]);
        setInsights(i);
        setMetrics(m);
      } catch (err) {
        console.error("Dashboard error", err);
      }
    };
    loadData();
  }, []);

  if (!insights || !metrics) return <div className="text-center py-20 animate-pulse text-gray-500">Compiling Market Insights...</div>;

  const demandData = [
    { name: 'High Demand', value: insights.high_demand_count },
    { name: 'Standard Demand', value: insights.low_demand_count }
  ];

  const catData = Object.entries(insights.categories).map(([name, count]) => ({ name, count }));
  
  const radarData = [
    { subject: 'Accuracy', A: (metrics.accuracy * 100), fullMark: 100 },
    { subject: 'Precision', A: (metrics.precision * 100), fullMark: 100 },
    { subject: 'Recall', A: (metrics.recall * 100), fullMark: 100 },
    { subject: 'F1 Score', A: (metrics.f1_high * 100), fullMark: 100 },
    { subject: 'ROC AUC', A: (metrics.roc_auc * 100), fullMark: 100 },
  ];

  const COLORS = ['#3b82f6', '#1f2937'];

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[
          { label: 'Total Index Jobs', value: insights.total_jobs.toLocaleString(), icon: '📋' },
          { label: 'High Demand Rate', value: `${insights.high_demand_rate}%`, icon: '📈' },
          { label: 'Analytic Precision', value: `${(metrics.precision * 100).toFixed(1)}%`, icon: '🎯' },
          { label: 'Predictions Served', value: insights.predictions_served, icon: '⚡' }
        ].map((stat, i) => (
          <div key={i} className="bg-[#1a1c23] border border-gray-800 p-6 rounded-3xl shadow-lg hover:border-gray-700 transition-colors">
            <div className="text-2xl mb-2">{stat.icon}</div>
            <div className="text-gray-500 text-sm font-medium">{stat.label}</div>
            <div className="text-2xl font-bold mt-1 tracking-tight">{stat.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Demand Dist */}
        <div className="bg-[#1a1c23] border border-gray-800 p-8 rounded-3xl">
          <h3 className="text-lg font-bold mb-8 flex justify-between items-center">
            Skill Demand Distribution
            <span className="text-xs bg-blue-500/10 text-blue-400 px-3 py-1 rounded-full border border-blue-500/20">Market View</span>
          </h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={demandData}
                  innerRadius={80}
                  outerRadius={120}
                  paddingAngle={8}
                  dataKey="value"
                  stroke="none"
                >
                  {demandData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '12px' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Legend layout="vertical" align="right" verticalAlign="middle" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Categories Bar */}
        <div className="bg-[#1a1c23] border border-gray-800 p-8 rounded-3xl">
          <h3 className="text-lg font-bold mb-8">Role Domain Frequency</h3>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={catData} layout="vertical">
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 11, fill: '#9ca3af' }} />
                <Tooltip cursor={{ fill: '#374151', opacity: 0.1 }} contentStyle={{ backgroundColor: '#111827', border: 'none', borderRadius: '12px' }} />
                <Bar dataKey="count" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Radar Metrics */}
        <div className="bg-[#1a1c23] border border-gray-800 p-8 rounded-3xl lg:col-span-2">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
            <div>
              <h3 className="text-lg font-bold">Model Performance Intelligence</h3>
              <p className="text-sm text-gray-400">Holistic view of predictor reliability metrics.</p>
            </div>
            <div className="flex gap-2">
              <button onClick={() => window.print()} className="px-5 py-2.5 bg-gray-800 hover:bg-gray-700 text-xs font-bold rounded-xl transition-all border border-gray-700">Export PDF Report</button>
              <a href="data:text/csv;charset=utf-8,ID,Title,Demand_Score" download="skill_demand_export.csv" className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-xs font-bold rounded-xl transition-all shadow-lg shadow-blue-500/20">Download Dataset</a>
            </div>
          </div>
          
          <div className="h-96 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                <PolarGrid stroke="#374151" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#9ca3af', fontSize: 12 }} />
                <Radar
                  name="Model Confidence"
                  dataKey="A"
                  stroke="#3b82f6"
                  fill="#3b82f6"
                  fillOpacity={0.3}
                />
                <Tooltip contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '12px' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
