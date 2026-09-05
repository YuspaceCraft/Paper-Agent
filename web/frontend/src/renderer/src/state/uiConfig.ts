/**
 * uiConfig.ts — 通用配置（外观/对话偏好/数据），前端本地持久化，即时生效。
 *
 * 持久化在 localStorage `demo_ui_config`；applyUIConfig 把可即时生效项写进 DOM：
 *   - theme → documentElement[data-theme]（配合 index.css 的暗色调色板）
 *   - zoom  → documentElement.style.zoom（Chromium/Electron 全量缩放，含 px 布局）
 * 对话偏好（时间戳/步骤展开/密度）由 MessageList 等消费端读取。
 */

export type Theme = 'light' | 'dark';
export type Density = 'comfortable' | 'compact';

export interface UIConfig {
  /** 亮/暗主题。 */
  theme: Theme;
  /** 界面缩放倍数：0.9 / 1 / 1.1 / 1.25。 */
  zoom: number;
  /** 消息气泡下显示时间戳。 */
  showTimestamps: boolean;
  /** 工具调用步骤卡片默认展开。 */
  stepsExpanded: boolean;
  /** 消息密度（紧凑=更小间距/字号）。 */
  density: Density;
  /** 配置中心最后停留的板块。 */
  configTab: string;
}

export const UI_CONFIG_KEY = 'demo_ui_config';

export const defaultUIConfig: UIConfig = {
  theme: 'light',
  zoom: 1,
  showTimestamps: false,
  stepsExpanded: false,
  density: 'comfortable',
  configTab: 'general',
};

export function loadUIConfig(): UIConfig {
  try {
    const raw = localStorage.getItem(UI_CONFIG_KEY);
    if (raw) return { ...defaultUIConfig, ...JSON.parse(raw) };
  } catch { /* ignore */ }
  return { ...defaultUIConfig };
}

export function saveUIConfig(cfg: UIConfig): void {
  try {
    localStorage.setItem(UI_CONFIG_KEY, JSON.stringify(cfg));
  } catch { /* ignore */ }
}

/** 把可即时生效项写入 DOM（theme/zoom）。调用发生在 App 挂载与每次保存后。 */
export function applyUIConfig(cfg: UIConfig): void {
  const root = document.documentElement;
  root.dataset.theme = cfg.theme;
  root.style.zoom = String(cfg.zoom);
  root.style.colorScheme = cfg.theme;
}