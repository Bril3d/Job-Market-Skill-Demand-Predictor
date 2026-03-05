import { useState, useEffect } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell, Legend
} from 'recharts';
import {
    Briefcase, TrendingUp, Activity, BrainCircuit, CheckCircle
} from 'lucide-react';
import { getInsights, getMetrics } from '../services/api';

const COLORS = ['#38bdf8', '#818cf8', '#fb7185', '#34d399', '#fbbf24'];

const Dashboard = () => {
  const [insights, setInsights] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [insightsData, metricsData] = await Promise.all([
          getInsights(),
          getMetrics()
        ]);
        setInsights(insightsData);
        setMetrics(metricsData);
      } catch (error) {
        console.error("Failed to load dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-sky-400"></div>
      </div>
    );
  }

  const categoryData = insights?.categories 
    ? Object.entries(insights.categories).map(([name, value]) => ({ name, value }))
    : [];

  const pieData = JSON.parse(JSON.stringify([
    { name: 'Standard Salary', value: insights?.standard_salary_count || 0 },
    { name: 'High Salary ($120k+)', value: insights?.high_salary_count || 0 }
  ]));

  return (
    <div className="container">
      <header className="mb-8">
        <h1 className="text-4xl font-bold gradient-text mb-2">Job Market Intelligence</h1>
        <p className="text-secondary">AI-driven insights into high-salary software roles.</p>
      </header>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard 
          icon={<Briefcase />} 
          title="Total Jobs Scraped" 
          value={insights?.total_jobs || 0} 
          subtitle="Real-time postings"
        />
        <StatCard 
          icon={<TrendingUp />} 
          title="Model Precision" 
          value={`${((metrics?.precision || 0) * 100).toFixed(1)}%`} 
          subtitle="Optimized threshold"
        />
        <StatCard 
          icon={<BrainCircuit />} 
          title="ROC-AUC Score" 
          value={(metrics?.roc_auc || 0).toFixed(2)} 
          subtitle="Prediction quality"
        />
        <StatCard 
          icon={<CheckCircle />} 
          title="ML Recall" 
          value={`${((metrics?.recall || 0) * 100).toFixed(0)}%`} 
          subtitle="High-salary capture"
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="glass-card">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Activity className="text-sky-400" /> Salary Distribution
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="glass-card">
          <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <TrendingUp className="text-indigo-400" /> Categories Distribution
          </h3>
          <div className="h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <YAxis stroke="#94a3b8" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="value" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ icon, title, value, subtitle }) => (
  <div className="glass-card flex items-start gap-4">
    <div className="p-3 bg-sky-500/10 rounded-lg text-sky-400">
      {icon}
    </div>
    <div>
      <h4 className="text-secondary text-sm font-medium">{title}</h4>
      <div className="text-2xl font-bold mt-1">{value}</div>
      <p className="text-xs text-sky-400/60 mt-1">{subtitle}</p>
    </div>
  </div>
);

export default Dashboard;
