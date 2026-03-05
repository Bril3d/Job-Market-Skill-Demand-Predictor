import { useState, useEffect } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, Legend, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import {
    Briefcase, TrendingUp, Activity, BrainCircuit, Download, Users
} from 'lucide-react';
import { getInsights, getMetrics } from '../services/api';

const COLORS = ['#38bdf8', '#818cf8', '#fb7185', '#34d399', '#fbbf24', '#f97316', '#a78bfa'];

const Dashboard = ({ darkMode }) => {
  const [insights, setInsights] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [insightsData, metricsData] = await Promise.all([
          getInsights(),
          getMetrics()
        ]);
        setInsights(insightsData);
        setMetrics(metricsData);
      } catch (err) {
        setError("Could not connect to API. Make sure the backend is running.");
        console.error("Failed to load dashboard data:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const exportCSV = () => {
    if (!insights?.categories) return;
    const rows = [["Category", "Count"]];
    Object.entries(insights.categories).forEach(([k, v]) => rows.push([k, v]));
    const csv = rows.map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "salary_insights.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-sky-400"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto px-4 py-16 text-center">
        <BrainCircuit size={48} className="mx-auto mb-4 text-red-400" />
        <h2 className="text-xl font-semibold mb-2">Connection Error</h2>
        <p className={`${darkMode ? 'text-slate-400' : 'text-gray-600'}`}>{error}</p>
        <button onClick={() => window.location.reload()} className="primary-btn mt-4">Retry</button>
      </div>
    );
  }

  const categoryData = insights?.categories 
    ? Object.entries(insights.categories).map(([name, value]) => ({ name, value }))
    : [];

  const pieData = [
    { name: 'Standard Salary', value: insights?.standard_salary_count || 0 },
    { name: 'High Salary ($120k+)', value: insights?.high_salary_count || 0 }
  ];

  const seniorityData = insights?.seniority_breakdown
    ? Object.entries(insights.seniority_breakdown).map(([name, value]) => ({ name, value }))
    : [];

  const radarData = metrics ? [
    { metric: 'Precision', value: (metrics.precision || 0) * 100 },
    { metric: 'Recall', value: (metrics.recall || 0) * 100 },
    { metric: 'F1', value: (metrics.f1_high || metrics.f1_macro || 0) * 100 },
    { metric: 'ROC-AUC', value: (metrics.roc_auc || 0) * 100 },
    { metric: 'Accuracy', value: (metrics.accuracy || 0) * 100 }
  ] : [];

  const cardBg = darkMode ? 'glass-card' : 'bg-white rounded-2xl p-6 shadow-md border border-gray-100';
  const labelColor = darkMode ? '#94a3b8' : '#6b7280';

  return (
    <div className="container mx-auto px-4 py-8">
      <header className="mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold gradient-text mb-2">Job Market Intelligence</h1>
          <p className={darkMode ? 'text-slate-400' : 'text-gray-500'}>AI-driven insights into high-salary software roles.</p>
        </div>
        <button onClick={exportCSV} className="primary-btn flex items-center gap-2 text-sm">
          <Download size={16} /> Export CSV
        </button>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard icon={<Briefcase />} title="Total Jobs" value={insights?.total_jobs || 0} subtitle="Scraped postings" darkMode={darkMode} />
        <StatCard icon={<TrendingUp />} title="High Salary Rate" value={`${insights?.high_salary_rate || 0}%`} subtitle="Above $120k" darkMode={darkMode} color="emerald" />
        <StatCard icon={<BrainCircuit />} title="ROC-AUC" value={(metrics?.roc_auc || 0).toFixed(2)} subtitle="Model quality" darkMode={darkMode} color="indigo" />
        <StatCard icon={<Users />} title="Predictions Served" value={insights?.predictions_served || 0} subtitle="API usage" darkMode={darkMode} color="amber" />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className={cardBg}>
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Activity className="text-sky-400" /> Salary Distribution
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={65} outerRadius={90} paddingAngle={5} dataKey="value" strokeWidth={0}>
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1e293b' : '#fff', border: 'none', borderRadius: '8px', color: darkMode ? '#fff' : '#000' }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className={cardBg}>
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <TrendingUp className="text-indigo-400" /> Job Categories
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#334155' : '#e5e7eb'} />
                <XAxis dataKey="name" stroke={labelColor} tick={{ fontSize: 11 }} angle={-15} textAnchor="end" />
                <YAxis stroke={labelColor} />
                <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1e293b' : '#fff', border: 'none', borderRadius: '8px', color: darkMode ? '#fff' : '#000' }} />
                <Bar dataKey="value" fill="#38bdf8" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Second Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className={cardBg}>
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Users className="text-emerald-400" /> Seniority Breakdown
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seniorityData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? '#334155' : '#e5e7eb'} />
                <XAxis type="number" stroke={labelColor} />
                <YAxis dataKey="name" type="category" stroke={labelColor} width={100} />
                <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1e293b' : '#fff', border: 'none', borderRadius: '8px', color: darkMode ? '#fff' : '#000' }} />
                <Bar dataKey="value" fill="#818cf8" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className={cardBg}>
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <BrainCircuit className="text-sky-400" /> Model Performance Radar
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid stroke={darkMode ? '#334155' : '#e5e7eb'} />
                <PolarAngleAxis dataKey="metric" stroke={labelColor} tick={{ fontSize: 12 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name="Score %" dataKey="value" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.3} />
                <Tooltip contentStyle={{ backgroundColor: darkMode ? '#1e293b' : '#fff', border: 'none', borderRadius: '8px', color: darkMode ? '#fff' : '#000' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ icon, title, value, subtitle, darkMode, color = "sky" }) => {
  const colorMap = {
    sky: { bg: 'bg-sky-500/10', text: 'text-sky-400' },
    emerald: { bg: 'bg-emerald-500/10', text: 'text-emerald-400' },
    indigo: { bg: 'bg-indigo-500/10', text: 'text-indigo-400' },
    amber: { bg: 'bg-amber-500/10', text: 'text-amber-400' },
  };
  const c = colorMap[color] || colorMap.sky;

  return (
    <div className={darkMode ? 'glass-card flex items-start gap-4' : 'bg-white rounded-2xl p-6 shadow-md border border-gray-100 flex items-start gap-4'}>
      <div className={`p-3 rounded-lg ${c.bg} ${c.text}`}>
        {icon}
      </div>
      <div>
        <h4 className={`text-sm font-medium ${darkMode ? 'text-slate-400' : 'text-gray-500'}`}>{title}</h4>
        <div className="text-2xl font-bold mt-1">{value}</div>
        <p className={`text-xs mt-1 ${c.text} opacity-60`}>{subtitle}</p>
      </div>
    </div>
  );
};

export default Dashboard;
