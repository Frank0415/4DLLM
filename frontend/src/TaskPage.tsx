import { useEffect, useState } from 'react';
import { api, date, Job, PageProps } from './api';

const states: Record<string, string> = { queued: '排队中', running: '运行中', succeeded: '已完成', failed: '失败', cancelled: '已取消' };
const kinds: Record<string, string> = { import: '扫描导入', cluster: '聚类分析', llm: 'LLM 分析' };

export default function TaskPage({ jobs, onError, onTask }: PageProps & { jobs: Job[] }) {
  const [selected, setSelected] = useState<string>(); const [log, setLog] = useState('');
  useEffect(() => {
    if (!selected) return;
    let active = true;
    const load = () => api<{ text: string }>(`/jobs/${selected}/log`).then(data => { if (active) setLog(data.text); }).catch(onError);
    void load(); const timer = setInterval(load, 3000);
    return () => { active = false; clearInterval(timer); };
  }, [selected]);
  const action = async (job: Job, kind: string) => {
    try { await api(`/jobs/${job.id}/${kind}`, {}); onTask(); } catch (e) { onError(e); }
  };
  return <><div className="page-heading"><div><span className="eyebrow">03 / PROCESSING QUEUE</span><h1>任务记录</h1><p>按顺序执行耗时任务，关闭页面后继续处理。</p></div><span className="pill">{jobs.filter(job => ['running', 'queued'].includes(job.status)).length} 个待完成</span></div>
    {!jobs.length ? <div className="panel empty"><div className="empty-orbit">↗</div><h3>任务会记录在这里</h3><p>导入数据或运行分析后，可以查看进度和结果。</p></div> : <div className="task-list">{jobs.map(job => <article className="panel task-card" key={job.id}>
      <div className="task-header"><div className={`task-symbol ${job.status}`}>{job.kind === 'import' ? '↧' : job.kind === 'cluster' ? '◈' : '✦'}</div><div><h3>{kinds[job.kind]}<span>{String(job.parameters.scan_name || job.result?.scan_name || (job.parameters.path ? String(job.parameters.path).split('/').pop() : `Scan #${job.parameters.scan_id}`))}</span></h3><small>{date(job.created_at)} · {job.id.slice(0, 8)}</small></div><span className={`status-badge ${job.status}`}>{job.status === 'running' && job.cancel_requested ? '已请求取消' : states[job.status]}</span></div>
      {job.status === 'running' && <><div className="progress-line"><span>{job.message}</span><strong>{Math.round(job.progress * 100)}%</strong></div><progress max="1" value={job.progress} /></>}
      {job.error && <p className="task-error">{job.error}</p>}
      {job.result?.status === 'partial' && <p className="task-error">部分 LLM 分析未完成：{job.result.errors?.join('；')}</p>}
      <div className="task-actions">{['running', 'queued'].includes(job.status) && <button className="text-button" disabled={job.cancel_requested} onClick={() => void action(job, 'cancel')}>{job.cancel_requested ? '等待当前步骤结束' : '取消任务'}</button>}{['failed', 'cancelled'].includes(job.status) && <button className="secondary" onClick={() => void action(job, 'retry')}>重新提交</button>}<button className="text-button" onClick={() => { setSelected(selected === job.id ? undefined : job.id); setLog(''); }}>{selected === job.id ? '收起日志' : '查看日志'}</button>{job.status === 'succeeded' && <a href={`/api/jobs/${job.id}/report`}>下载记录 ↗</a>}</div>
      {selected === job.id && <pre className="job-log">{log || '尚无日志。'}</pre>}
    </article>)}</div>}
  </>;
}
