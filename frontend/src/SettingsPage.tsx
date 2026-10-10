import { useEffect, useState } from 'react';
import { api, PageProps } from './api';

type Settings = { import_roots: string[]; database: { host: string; port: number; user: string; dbname: string; password_configured: boolean }; llm: { base_url: string; model: string; keys_configured: boolean } };

export default function SettingsPage({ onError, onSaved }: PageProps & { onSaved: () => void }) {
  const [config, setConfig] = useState<Settings>(); const [roots, setRoots] = useState('');
  const [password, setPassword] = useState(''); const [keys, setKeys] = useState(''); const [busy, setBusy] = useState(false); const [saved, setSaved] = useState(false);
  useEffect(() => { api<Settings>('/settings').then(data => { setConfig(data); setRoots(data.import_roots.join('\n')); }).catch(onError); }, []);
  const save = async () => {
    if (!config) return; setBusy(true); setSaved(false);
    try {
      await api('/settings', { import_roots: roots.split('\n').map(line => line.trim()).filter(Boolean),
        database: { ...config.database, password: password || null },
        llm: { ...config.llm, api_keys: keys.trim() ? keys.split('\n').map(line => line.trim()).filter(Boolean) : null } }, 'PUT');
      setPassword(''); setKeys(''); setSaved(true); onSaved();
    } catch (e) { onError(e); } finally { setBusy(false); }
  };
  return <><div className="page-heading"><div><span className="eyebrow">04 / WORKSPACE SETTINGS</span><h1>工作台设置</h1><p>配置数据目录、数据库，以及可选的 LLM 分析。</p></div></div>
    {config ? <form className="settings-layout" onSubmit={e => { e.preventDefault(); void save(); }}>
      <section className="panel settings-card"><div className="panel-title"><h2>本机数据目录</h2><span className="step-mark">01</span></div><label>允许浏览的目录<textarea rows={4} value={roots} onChange={e => setRoots(e.target.value)} placeholder="每行一个绝对目录路径" /></label><p className="hint">选择目录后，后端直接读取文件；无需上传整份扫描数据。</p></section>
      <section className="panel settings-card"><div className="panel-title"><h2>PostgreSQL</h2><span className="step-mark">02</span></div><div className="form-grid"><label>主机<input required value={config.database.host} onChange={e => setConfig({ ...config, database: { ...config.database, host: e.target.value } })} /></label><label>端口<input required type="number" min="1" max="65535" value={config.database.port} onChange={e => setConfig({ ...config, database: { ...config.database, port: Number(e.target.value) } })} /></label><label>用户名<input required value={config.database.user} onChange={e => setConfig({ ...config, database: { ...config.database, user: e.target.value } })} /></label><label>数据库<input required value={config.database.dbname} onChange={e => setConfig({ ...config, database: { ...config.database, dbname: e.target.value } })} /></label></div><label>密码<input type="password" autoComplete="new-password" placeholder={config.database.password_configured ? '已配置，留空保留' : '数据库密码'} value={password} onChange={e => setPassword(e.target.value)} /></label></section>
      <section className="panel settings-card"><div className="panel-title"><h2>LLM · Gemini 兼容接口</h2><span className="pill">可选</span></div><label>API 基础地址<input type="url" value={config.llm.base_url} onChange={e => setConfig({ ...config, llm: { ...config.llm, base_url: e.target.value } })} /></label><label>模型名称<input value={config.llm.model} onChange={e => setConfig({ ...config, llm: { ...config.llm, model: e.target.value } })} /></label><label>API 密钥<textarea rows={3} autoComplete="off" placeholder={config.llm.keys_configured ? '已配置，留空保留；更换时每行一个密钥' : '每行一个 API 密钥'} value={keys} onChange={e => setKeys(e.target.value)} /></label><p className="hint">未配置密钥也可以完成导入、预览和聚类。</p></section>
      <div className="settings-actions">{saved && <span className="saved-note">设置已保存</span>}<button className="primary" disabled={busy}>{busy ? '保存中…' : '保存设置 →'}</button></div>
    </form> : <div className="panel empty">加载配置…</div>}
  </>;
}
