import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import path from 'path'
import { PythonBackend } from './backend'

app.setName('Demo Agent')

let mainWindow: BrowserWindow | null = null
let backend: PythonBackend | null = null

const isDev = !app.isPackaged
const VITE_DEV_URL = 'http://localhost:5173'

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 900,
    minHeight: 600,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  if (isDev) {
    // Renderer is served by the Vite dev server (started by `npm run dev:renderer`).
    mainWindow.loadURL(VITE_DEV_URL)
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'))
  }

  mainWindow.once('ready-to-show', () => mainWindow?.show())
  mainWindow.on('closed', () => { mainWindow = null })
}

function setupIPC() {
  ipcMain.handle('get-backend-port', () => backend?.getPort() ?? null)
  ipcMain.handle('get-backend-status', () => backend?.getStatus() ?? 'stopped')
  ipcMain.handle('shell:open-path', async (_e, path?: string) => {
    // 配置中心「Skills → 打开目录」：在系统文件管理器中显示该目录。
    if (!path) return false
    const r = await shell.openPath(String(path))
    return r === '' // 空字符串 = 打开成功
  })
  ipcMain.handle('dialog:open-directory', async (_e, defaultPath?: string) => {
    // 原生「选择文件夹」对话框（与上传 PDF 的原生文件对话框同类）。
    // 返回绝对路径或 null（取消）。cwd 提示来自调用方当前值。
    if (!mainWindow) return null
    const r = await dialog.showOpenDialog(mainWindow, {
      title: '选择文件夹',
      properties: ['openDirectory', 'createDirectory'],
      ...(defaultPath ? { defaultPath } : {}),
    })
    return r.canceled || r.filePaths.length === 0 ? null : r.filePaths[0]
  })

  ipcMain.handle('backend-restart', async () => {
    // Retry after a startup failure (e.g. :8001 already in use) without
    // relaunching Electron. The result flows back via the usual
    // `ready`/`error` backend-status events.
    if (!backend) return 'stopped'
    const status = backend.getStatus()
    if (status === 'ready' || status === 'starting') return status
    await backend.start()
    return backend.getStatus()
  })
}

function startBackend() {
  backend = new PythonBackend()
  backend.on('ready', (port: number) => {
    console.log(`[backend] ready on ${port}`)
    mainWindow?.webContents.send('backend-status', { status: 'ready', port })
  })
  backend.on('log', (line: string) => {
    console.log(`[backend] ${line}`)
    mainWindow?.webContents.send('backend-log', line)
  })
  backend.on('error', (message: string) => {
    console.error(`[backend] ${message}`)
    mainWindow?.webContents.send('backend-status', { status: 'error', error: message })
  })
  void backend.start()
}

// ponytail: single-instance lock; a second launch focuses the existing window.
if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.show()
      mainWindow.focus()
    }
  })

  app.whenReady().then(() => {
    setupIPC()
    createWindow()
    startBackend()

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow()
    })
  })
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => backend?.stop())
