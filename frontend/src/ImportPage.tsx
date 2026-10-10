import { useEffect, useState } from 'react';
import { api, bytes, PageProps } from './api';

type Listing = { path: string; roots: string[]; entries: { name: string; path: string; directory: boolean; size_bytes: number }[] };
type Inspection = { path: string; filename: string; size_bytes: number; frame_count: number; bit_depth: number; detector_shape: number[]; scan_shape: number[] | null; hdr_path: string | null };

export default function ImportPage({ onError, onTask }: PageProps) {
  const [listing, setListing] = useState<Listing>();
  const [path, setPath] = useState('');
  const [info, setInfo] = useState<Inspection>();
  const [shape, setShape] = useState(['', '']);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const browse = async (value?: string) => {
    try { const data = await api<Listing>(`/files${value ? `?path=${encodeURIComponent(value)}` : ''}`); setListing(data); setPath(data.path); }
    catch (e) { onError(e); }
  };
  useEffect(() => { void browse(); }, []);
  const inspect = async (source: string) => {
    setBusy(true);
    try {
      const data = await api<Inspection>('/inspect', { path: source });
      setInfo(data); setShape(data.scan_shape?.map(String) || ['', '']); setName(data.filename.replace(/\.mib$/i, ''));
    } catch (e) { onError(e); } finally { setBusy(false); }
  };
  const scanShape = () => shape.every(Boolean) ? shape.map(Number) : undefined;
  const importSelected = async (folder = false) => {
    setBusy(true);
    try {
      if (folder) await api('/import-folders', { path: listing?.path, scan_shape: scanShape() });
      else await api('/imports', { path: info?.path, scan_name: name, scan_shape: scanShape() });
      onTask();
    } catch (e) { onError(e); } finally { setBusy(false); }
  };
  return <>
    <div className="page-heading"><div><span className="eyebrow">01 / DATA INGESTION</span><h1>导入扫描</h1><p>选择本机数据，让读取、转换和入库一次完成。</p></div><span className="pill">MIB · U08 / U16 / U32</span></div>
    <div className="import-layout">
      <section className="panel file-browser"><div className="panel-title"><h2>数据目录</h2><button className="text-button" onClick={() => void browse(listing?.path)}>刷新</button></div>
        <div className="directory-roots">{listing?.roots.map(root => <button key={root} title={root} onClick={() => void browse(root)}>{root.split('/').filter(Boolean).pop()}</button>)}</div>
        <form className="path-form" onSubmit={e => { e.preventDefault(); void browse(path); }}><input aria-label="目录路径" value={path} onChange={e => setPath(e.target.value)} /><button className="secondary">前往</button></form>
        <button className="file-row parent-row" onClick={() => void browse(`${listing?.path}/..`)}><span>↰</span><span>上一级目录</span></button>
        <div className="file-list">{listing?.entries.map(entry => <button className={`file-row ${info?.path === entry.path ? 'selected' : ''}`} key={entry.path} onClick={() => entry.directory ? void browse(entry.path) : void inspect(entry.path)}>
          <span className={`file-icon ${entry.directory ? 'folder' : ''}`}>{entry.directory ? '▱' : '◈'}</span><span className="file-name">{entry.name}</span><span className="file-size">{entry.directory ? '目录' : bytes(entry.size_bytes)}</span>
        </button>)}{listing && !listing.entries.length && <div className="empty small">此目录中没有 MIB 文件或子目录。</div>}</div>
        <div className="panel-footer"><span>批量导入当前目录中的 MIB 文件</span><button className="secondary" disabled={busy || !listing} onClick={() => void importSelected(true)}>导入整个目录</button></div>
      </section>
      <section className="panel inspection"><div className="panel-title"><h2>导入预览</h2><span className="step-mark">A → B</span></div>
        {info ? <><div className="pattern-preview"><img src={`/api/source-preview?path=${encodeURIComponent(info.path)}`} alt="首个衍射图案" /><span>首个衍射图案</span></div>
          <div className="file-summary"><h3>{info.filename}</h3><div className="metric-row"><span>{bytes(info.size_bytes)}</span><span>{info.frame_count.toLocaleString()} 帧</span><span>{info.bit_depth} bit</span><span>{info.detector_shape.join(' × ')}</span></div></div>
          <label>扫描名称<input value={name} onChange={e => setName(e.target.value)} /></label>
          <div className="form-grid"><label>扫描行数<input type="number" min="1" value={shape[0]} onChange={e => setShape([e.target.value, shape[1]])} /></label><label>扫描列数<input type="number" min="1" value={shape[1]} onChange={e => setShape([shape[0], e.target.value])} /></label></div>
          <p className="hint">{info.hdr_path ? '已读取配套 HDR。' : '未发现配套 HDR。'} 行数 × 列数应等于总帧数。</p>
          <button className="primary wide" disabled={busy || !shape.every(Boolean) || Number(shape[0]) * Number(shape[1]) !== info.frame_count || !name.trim()} onClick={() => void importSelected()}>{busy ? '正在提交…' : '导入扫描 →'}</button>
        </> : <div className="empty"><div className="empty-orbit">◈</div><h3>从左侧选择一个 MIB 文件</h3><p>在导入前查看图案、尺寸和文件信息。</p></div>}
      </section>
    </div>
  </>;
}
