export async function api<T = unknown>(path: string, body?: unknown, method = 'POST'): Promise<T> {
  const response = await fetch(`/api${path}`, body === undefined ? undefined : {
    method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  }
  return response.json();
}

export const bytes = (n: number) => n >= 1024 ** 3 ? `${(n / 1024 ** 3).toFixed(2)} GB` : `${(n / 1024 ** 2).toFixed(1)} MB`;
export const date = (s: string) => new Date(s).toLocaleString('zh-CN', { hour12: false });
export const categoryNames: Record<string, string> = { vacuum: '真空', crystalline: '晶体', amorphous: '非晶', mixed: '混合', empty: '真空', mixing: '混合' };
export const colors = ['#6366f1', '#14b8a6', '#f59e0b', '#f43f5e', '#8b5cf6', '#0ea5e9', '#84cc16', '#ec4899', '#64748b', '#d97706', '#059669', '#7c3aed'];
export function clusterColor(index: number): string {
  if (index < colors.length) return colors[index];
  const hue = (index * 137.508) % 360;
  const channel = (n: number) => {
    const k = (n + hue / 30) % 12;
    const value = 0.54 - 0.285 * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(value * 255).toString(16).padStart(2, '0');
  };
  return `#${channel(0)}${channel(8)}${channel(4)}`;
}
export type Job = { id: string; kind: string; status: string; cancel_requested?: boolean; progress: number; stage: string; message: string; error?: string; created_at: string; updated_at: string; parameters: Record<string, unknown>; result?: { scan_id?: number; scan_name?: string; clustering_run_id?: number; status?: string; errors?: string[] } };
export type Scan = { id: number; scan_name: string; created_at: string; pattern_count: number; run_count: number; metadata: { scan_shape?: number[]; detector_shape?: number[]; bit_depth?: number }; scan_shape?: number[]; runs?: Run[] };
export type Run = { id: number; k_value: number; run_timestamp: string; parameters: Record<string, unknown>; files?: string[]; clusters?: Cluster[] };
export type Cluster = { id: number; cluster_index: number; pattern_count: number; manual_category: string | null; manual_note: string | null; llm_assigned_class: string | null; llm_detailed_features: { dominant_classification_code?: number; samples?: { description?: string; phase_type?: string; confidence?: string }[]; consensus_description?: string } | null };
export type Status = { database_ready: boolean; database_error: string | null; llm_ready: boolean; storage_directory: string };
export type PageProps = { onError: (error: unknown) => void; onTask: () => void; refresh: number };
