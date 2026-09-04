/// <reference types="vite/client" />

// react-syntax-highlighter v16 ships no type declarations; keep it loosely typed.
declare module 'react-syntax-highlighter';
declare module 'react-syntax-highlighter/dist/esm/styles/prism/one-light';

interface Window {
  electronAPI?: {
    getBackendPort: () => Promise<number | null>;
    getBackendStatus: () => Promise<string>;
    restartBackend: () => Promise<string>;
    onBackendStatus: (cb: (data: { status: string; port?: number; error?: string }) => void) => () => void;
    onBackendLog: (cb: (line: string) => void) => () => void;
    /** 原生「选择文件夹」对话框 → 绝对路径或 null（取消）。 */
    selectDirectory: (defaultPath?: string) => Promise<string | null>;
  };
}
