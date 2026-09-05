/**
 * appContext.ts — 应用级 Context（state + dispatch + uiConfig）。
 * 独立模块避免 App.tsx ↔ 深层组件（MessageList 等）的循环 import。
 */
import { createContext, useContext } from 'react';
import type { Action, AppState } from '../state';
import { initialAppState } from '../state';
import type { UIConfig } from '../state/uiConfig';
import { defaultUIConfig } from '../state/uiConfig';

export interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<Action>;
  /** 配置中心「通用配置」——消费端（MessageList 等）读取对话显示偏好。 */
  uiConfig: UIConfig;
}

export const AppCtx = createContext<AppContextType>({
  state: initialAppState,
  dispatch: () => {},
  uiConfig: defaultUIConfig,
});

export const useApp = () => useContext(AppCtx);