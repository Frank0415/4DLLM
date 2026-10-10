import { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ImportPage from './ImportPage';
import ScanPage from './ScanPage';
import TaskPage from './TaskPage';
import SettingsPage from './SettingsPage';
import { api, Job, Status } from './api';
import './style.css';

type Page = 'import' | 'scans' | 'tasks' | 'settings';
const pages: { id: Page; icon: string; name: string }[] = [
  { id: 'import', icon: '↧', name: '导入扫描' }, { id: 'scans', icon: '◈', name: '扫描与分析' },
  { id: 'tasks', icon: '☷', name: '任务记录' }, { id: 'settings', icon: '⚙', name: '设置' },
];

function App() {
  const [page, setPage] = useState<Page>('import'); const [error, setError] = useState('');
  const [status, setStatus] = useState<Status>(); const [jobs, setJobs] = useState<Job[]>([]);
  const [refresh, setRefresh] = useState(0); const completed = useRef('');
  const onError = useCallback((problem: unknown) => setError(problem instanceof Error ? problem.message : String(problem)), []);
  const loadStatus = useCallback(() => { api<Status>('/status').then(setStatus).catch(onError); }, [onError]);
  const loadJobs = useCallback(() => {
    api<Job[]>('/jobs').then(data => {
      setJobs(data); const stamp = data.filter(job => job.status === 'succeeded').map(job => job.id).join(',');
      if (completed.current !== stamp) { completed.current = stamp; setRefresh(value => value + 1); }
    }).catch(onError);
  }, [onError]);
  useEffect(() => { loadStatus(); loadJobs(); const timer = setInterval(loadJobs, 2500); return () => clearInterval(timer); }, [loadJobs, loadStatus]);
  const onTask = () => { setPage('tasks'); loadJobs(); };
  const active = jobs.filter(job => ['running', 'queued'].includes(job.status)).length;
  const props = { onError, onTask, refresh };
  return <div className="app-shell"><aside className="sidebar">
    <div className="brand"><div className="brand-symbol"><span /><span /><span /><span /><span /></div><div><strong>4DLLM</strong><span>RESEARCH WORKBENCH</span></div></div>
    <div className="nav-label">工作空间</div><nav>{pages.map(item => <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => { setPage(item.id); setError(''); }}><span className="nav-icon">{item.icon}</span>{item.name}{item.id === 'tasks' && active > 0 && <span className="nav-count">{active}</span>}</button>)}</nav>
    <div className="sidebar-bottom"><div className="local-mark"><span />本机工作空间</div><p>4D-STEM 数据处理与解释</p><small>v0.2.0</small></div>
  </aside><div className="main-shell"><header className="topbar"><div><span className="breadcrumb">工作台</span><span className="breadcrumb-separator">/</span><strong>{pages.find(item => item.id === page)?.name}</strong></div><div className="connections"><span><i className={status?.database_ready ? 'online' : 'offline'} />数据库{status?.database_ready ? '已连接' : '待配置'}</span><span><i className={status?.llm_ready ? 'online' : 'offline'} />LLM{status?.llm_ready ? '已配置' : '可选'}</span></div></header>
    <main>{error && <div role="alert" className="error-banner"><span>{error}</span><button aria-label="关闭提示" onClick={() => setError('')}>×</button></div>}
      {status && !status.database_ready && page !== 'settings' && <div className="setup-banner"><div><strong>连接数据库后即可开始</strong><span>在设置中填写 PostgreSQL 连接信息。</span></div><button className="secondary" onClick={() => setPage('settings')}>打开设置 →</button></div>}
      {page === 'import' && <ImportPage {...props} />}{page === 'scans' && <ScanPage {...props} />}{page === 'tasks' && <TaskPage {...props} jobs={jobs} />}{page === 'settings' && <SettingsPage {...props} onSaved={() => { loadStatus(); setRefresh(value => value + 1); }} />}
    </main><footer><span>4DLLM / 4D-STEM Workbench</span><span title={status?.storage_directory}>{status?.storage_directory}</span></footer>
  </div></div>;
}

createRoot(document.getElementById('root')!).render(<App />);
