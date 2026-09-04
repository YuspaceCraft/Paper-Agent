import { contextBridge, ipcRenderer } from 'electron'

export interface BackendStatus {
  status: string
  port?: number
  error?: string
}

contextBridge.exposeInMainWorld('electronAPI', {
  getBackendPort: () => ipcRenderer.invoke('get-backend-port') as Promise<number | null>,
  getBackendStatus: () => ipcRenderer.invoke('get-backend-status') as Promise<string>,
  restartBackend: () => ipcRenderer.invoke('backend-restart') as Promise<string>,
  selectDirectory: (defaultPath?: string) =>
    ipcRenderer.invoke('dialog:open-directory', defaultPath) as Promise<string | null>,
  onBackendStatus: (callback: (data: BackendStatus) => void) => {
    const handler = (_event: unknown, data: BackendStatus) => callback(data)
    ipcRenderer.on('backend-status', handler)
    return () => ipcRenderer.removeListener('backend-status', handler)
  },
  onBackendLog: (callback: (line: string) => void) => {
    const handler = (_event: unknown, line: string) => callback(line)
    ipcRenderer.on('backend-log', handler)
    return () => ipcRenderer.removeListener('backend-log', handler)
  },
})
