import { useEffect, useRef, useState } from 'react';
import { api, categoryNames, clusterColor, date, Cluster, PageProps, Run, Scan } from './api';

function ClusterGrid({ run, onSelect }: { run: number; onSelect: (row: number, col: number) => void }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [shape, setShape] = useState<number[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    api<{ shape: number[]; labels: number[] }>(`/runs/${run}/grid`).then(data => {
      if (!active || !canvas.current) return;
      const [rows, cols] = data.shape; setShape(data.shape);
      const element = canvas.current; element.width = cols; element.height = rows;
      const context = element.getContext('2d')!; const image = context.createImageData(cols, rows);
      data.labels.forEach((label, i) => {
        const color = label < 0 ? '#e2e8f0' : clusterColor(label);
        image.data[i * 4] = parseInt(color.slice(1, 3), 16); image.data[i * 4 + 1] = parseInt(color.slice(3, 5), 16);
        image.data[i * 4 + 2] = parseInt(color.slice(5, 7), 16); image.data[i * 4 + 3] = 255;
      });
      context.putImageData(image, 0, 0);
    }).catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, [run]);
  return <>{error && <p className="hint">{error}</p>}<canvas ref={canvas} className="scan-grid" aria-label="聚类空间图，点击查看图案" onClick={e => {
    const bounds = e.currentTarget.getBoundingClientRect();
    if (shape.length) onSelect(Math.min(shape[0], Math.floor((e.clientY - bounds.top) / bounds.height * shape[0]) + 1), Math.min(shape[1], Math.floor((e.clientX - bounds.left) / bounds.width * shape[1]) + 1));
  }} /></>;
}

function ClusterCard({ cluster, run, onError, reload }: { cluster: Cluster; run: number; onError: PageProps['onError']; reload: () => void }) {
  const [category, setCategory] = useState(cluster.manual_category || '');
  const [note, setNote] = useState(cluster.manual_note || '');
  const [busy, setBusy] = useState(false);
  const details = cluster.llm_detailed_features;
  const description = details?.consensus_description || details?.samples?.[0]?.description;
  return <article className="cluster-card"><div className="cluster-title"><span className="color-dot" style={{ background: clusterColor(cluster.cluster_index) }} /><h3>Cluster {cluster.cluster_index}</h3><span>{cluster.pattern_count.toLocaleString()} 个图案</span></div>
    <img className="montage" src={`/api/runs/${run}/files/montages/cluster_${cluster.cluster_index}.png`} alt={`Cluster ${cluster.cluster_index} 代表图案`} loading="lazy" />
    <div className="cluster-body"><div className="llm-label"><span>LLM 建议</span><strong>{cluster.llm_assigned_class ? categoryNames[cluster.llm_assigned_class] || cluster.llm_assigned_class : '尚未分析'}</strong></div>
      {description && <p className="description">{description}</p>}
      <label>人工确认<select value={category} onChange={e => setCategory(e.target.value)}><option value="">未确认</option>{Object.entries(categoryNames).filter(([key]) => !['empty', 'mixing'].includes(key)).map(([key, name]) => <option value={key} key={key}>{name}</option>)}</select></label>
      <input aria-label="标注备注" placeholder="标注备注（可选）" value={note} onChange={e => setNote(e.target.value)} />
      <button className="secondary wide" disabled={busy} onClick={async () => {
        setBusy(true); try { await api(`/clusters/${cluster.id}/annotation`, { category: category || null, note }, 'PUT'); reload(); } catch (e) { onError(e); } finally { setBusy(false); }
      }}>{busy ? '保存中…' : '保存标注'}</button>
    </div>
  </article>;
}

export default function ScanPage({ onError, onTask, refresh }: PageProps) {
  const previousScan = useRef<number | undefined>(undefined);
  const [scans, setScans] = useState<Scan[]>([]);
  const [selected, setSelected] = useState<number>();
  const [scan, setScan] = useState<Scan>();
  const [runID, setRunID] = useState<number>();
  const [run, setRun] = useState<Run>();
  const [coordinate, setCoordinate] = useState([1, 1]);
  const [k, setK] = useState(16); const [seed, setSeed] = useState(0); const [device, setDevice] = useState('');
  const [samples, setSamples] = useState(5); const [busy, setBusy] = useState(false);
  const reloadRun = () => { if (runID) api<Run>(`/runs/${runID}`).then(data => setRun(current => current?.id === data.id ? data : current)).catch(onError); };
  useEffect(() => { api<Scan[]>('/scans').then(data => { setScans(data); if (!selected && data.length) setSelected(data[0].id); }).catch(onError); }, [refresh]);
  useEffect(() => {
    if (!selected) return;
    let active = true; setScan(undefined); setRun(undefined); setCoordinate([1, 1]);
    api<Scan>(`/scans/${selected}`).then(data => {
      if (!active) return; setScan(data);
      setRunID(current => previousScan.current === selected && current && data.runs?.some(item => item.id === current) ? current : data.runs?.[0]?.id);
      previousScan.current = selected;
    }).catch(onError);
    return () => { active = false; };
  }, [selected, refresh]);
  useEffect(() => {
    let active = true; setRun(undefined);
    if (runID) api<Run>(`/runs/${runID}`).then(data => { if (active) setRun(data); }).catch(onError);
    return () => { active = false; };
  }, [runID, refresh]);
  const submit = async (kind: 'cluster' | 'llm') => {
    setBusy(true);
    try { await api(`/scans/${selected}/${kind}`, kind === 'cluster' ? { k, seed, device: device || null } : { samples, concurrency: 3 }); onTask(); }
    catch (e) { onError(e); } finally { setBusy(false); }
  };
  return <><div className="page-heading"><div><span className="eyebrow">02 / EXPLORE & ANALYZE</span><h1>扫描与分析</h1><p>从空间分布走进单个衍射图案，保留每次分析结果。</p></div><span className="pill">{scans.length} 个扫描</span></div>
    {!scans.length ? <div className="panel empty"><div className="empty-orbit">◎</div><h3>还没有扫描数据</h3><p>先在导入页添加一份 MIB 数据。</p></div> : <div className="scan-layout"><aside className="panel scan-list">{scans.map(item => <button key={item.id} className={`scan-item ${item.id === selected ? 'active' : ''}`} onClick={() => setSelected(item.id)}><strong>{item.scan_name}</strong><span>{item.pattern_count.toLocaleString()} 个图案 · {item.run_count} 次分析</span><small>{date(item.created_at)}</small></button>)}</aside>
    <div className="scan-content">{scan ? <>
      <section className="panel"><div className="panel-title"><div><h2>{scan.scan_name}</h2><p className="hint">扫描 {scan.scan_shape?.join(' × ')} · 点击空间图查看对应衍射图案</p></div><select aria-label="分析历史" value={runID || ''} onChange={e => setRunID(Number(e.target.value))}><option value="">原始扫描</option>{scan.runs?.map(item => <option value={item.id} key={item.id}>Run #{item.id} · K={item.k_value} · {date(item.run_timestamp)}</option>)}</select></div>
        <div className="preview-pair"><div className="map-box">{runID ? <ClusterGrid key={runID} run={runID} onSelect={(row, col) => setCoordinate([row, col])} /> : <img className="scan-grid" src={`/api/scans/${scan.id}/overview`} alt="扫描总强度图" onClick={e => { const box = e.currentTarget.getBoundingClientRect(); const shape = scan.scan_shape!; setCoordinate([Math.min(shape[0], Math.floor((e.clientY - box.top) / box.height * shape[0]) + 1), Math.min(shape[1], Math.floor((e.clientX - box.left) / box.width * shape[1]) + 1)]); }} />}
          <span className="image-caption">{runID ? '聚类空间图' : '总强度空间图'}</span></div>
          <div className="diffraction-box"><img src={`/api/scans/${scan.id}/pattern?row=${coordinate[0]}&col=${coordinate[1]}`} alt={`第 ${coordinate[0]} 行第 ${coordinate[1]} 列的衍射图案`} /><div className="coordinate-fields"><label>行<input type="number" min="1" max={scan.scan_shape?.[0]} value={coordinate[0]} onChange={e => setCoordinate([Math.max(1, Math.min(scan.scan_shape![0], Number(e.target.value))), coordinate[1]])} /></label><label>列<input type="number" min="1" max={scan.scan_shape?.[1]} value={coordinate[1]} onChange={e => setCoordinate([coordinate[0], Math.max(1, Math.min(scan.scan_shape![1], Number(e.target.value)))])} /></label></div></div>
        </div>
      </section>
      <section className="panel analysis-controls"><div><h2>运行分析</h2><p className="hint">新结果独立保存，原有运行记录继续保留。</p></div><div className="control-line"><label>K<input type="number" min="1" max="256" value={k} onChange={e => setK(Number(e.target.value))} /></label><label>随机种子<input type="number" min="0" value={seed} onChange={e => setSeed(Number(e.target.value))} /></label><label>计算设备<select value={device} onChange={e => setDevice(e.target.value)}><option value="">自动</option><option value="cpu">CPU</option><option value="cuda">CUDA</option></select></label><button className="primary" disabled={busy} onClick={() => void submit('cluster')}>运行聚类 →</button></div>
        <div className="control-line llm-controls"><label>每簇样本数<input type="number" min="1" max="16" value={samples} onChange={e => setSamples(Number(e.target.value))} /></label><button className="secondary" disabled={busy || !scan.runs?.length} onClick={() => void submit('llm')}>分析最新运行的 LLM 建议</button><span className="hint">材料类别与人工确认分别记录。</span></div>
      </section>
      {run && <><div className="section-heading"><h2>Run #{run.id} · 簇与标注</h2><div className="export-links">{run.files?.filter(name => name.endsWith('.npz') || name === 'parameters.json').map(name => <a key={name} href={`/api/runs/${run.id}/files/${name}`}>{name.endsWith('.npz') ? '下载聚类数据 ↗' : '下载参数 ↗'}</a>)}</div></div><div className="cluster-grid">{run.clusters?.map(cluster => <ClusterCard key={`${run.id}-${cluster.id}-${cluster.manual_category}-${cluster.manual_note}`} cluster={cluster} run={run.id} onError={onError} reload={reloadRun} />)}</div><details className="panel parameters"><summary>分析参数</summary><pre>{JSON.stringify(run.parameters, null, 2)}</pre></details></>}
    </> : <div className="panel empty">加载扫描…</div>}</div></div>}
  </>;
}
