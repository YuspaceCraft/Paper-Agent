import { spawn, type ChildProcess } from 'child_process'
import { EventEmitter } from 'events'
import path from 'path'
import http from 'http'

// ponytail: fixed port 8001 (avoid the dev web console's 8000). Add pickPort /
// fallback ports only if a collision actually happens.
const BACKEND_PORT = 8001

/**
 * PythonBackend — spawn `python -m uvicorn web.api.main:app` as a child and
 * probe /api/health until ready. Python is NOT bundled (see package.json
 * build): it resolves from DEMO_PYTHON, else `python`/`python3` on PATH.
 */
export class PythonBackend extends EventEmitter {
  private process: ChildProcess | null = null
  private status: 'stopped' | 'starting' | 'ready' | 'error' = 'stopped'
  private port = BACKEND_PORT

  getPort(): number {
    return this.port
  }

  getStatus(): string {
    return this.status
  }

  private findPython(): string {
    if (process.env.DEMO_PYTHON) return process.env.DEMO_PYTHON
    return process.platform === 'win32' ? 'python' : 'python3'
  }

  // Backend writes relative paths (checkpoints.db, pdf_pipeline/output), so it
  // must run with the project root as cwd. Compiled main lives at
  // dist/main/index.js → 4 levels up is the repo root.
  private projectRoot(): string {
    return path.resolve(__dirname, '../../../..')
  }

  async start(): Promise<void> {
    if (this.status === 'ready' || this.status === 'starting') return
    this.status = 'starting'

    const python = this.findPython()
    const args = [
      '-m', 'uvicorn', 'web.api.main:app',
      '--host', '127.0.0.1', '--port', String(this.port),
    ]
    this.emit('log', `Starting backend: ${python} ${args.join(' ')} (cwd=${this.projectRoot()})`)

    this.process = spawn(python, args, {
      cwd: this.projectRoot(),
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        // builtin agent tools (search_papers/fetch_content/…) call the local API
        // via AGENT_API_BASE; default is :8000 but we bind :8001 — point them here.
        AGENT_API_BASE: `http://127.0.0.1:${this.port}`,
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    const onOutput = (data: Buffer) => {
      for (const line of data.toString().split('\n')) {
        if (line.trim()) this.emit('log', line)
      }
    }
    this.process.stdout?.on('data', onOutput)
    this.process.stderr?.on('data', onOutput)

    this.process.on('exit', (code) => {
      const wasReady = this.status === 'ready'
      this.status = 'stopped'
      this.emit('log', `Backend exited (code ${code})`)
      if (!wasReady && code !== 0) {
        this.status = 'error'
        this.emit('error', `Backend exited during startup (code ${code})`)
      }
    })
    this.process.on('error', (err) => {
      this.status = 'error'
      this.emit('error', `Failed to start Python: ${err.message}`)
    })

    await this.waitForReady()
  }

  private waitForReady(): Promise<void> {
    return new Promise((resolve) => {
      const deadline = Date.now() + 30_000
      const check = () => {
        if (this.status !== 'starting') return resolve()
        if (Date.now() >= deadline) {
          this.status = 'error'
          this.emit('error', 'Backend timed out after 30s')
          return resolve()
        }
        const req = http.get(`http://127.0.0.1:${this.port}/api/health`, (res) => {
          res.resume() // drain the body so the keep-alive socket is freed
          if (res.statusCode === 200) {
            this.status = 'ready'
            this.emit('ready', this.port)
            resolve()
          } else {
            setTimeout(check, 1000)
          }
        })
        req.on('error', () => setTimeout(check, 1000))
        req.setTimeout(2000, () => {
          req.destroy()
          setTimeout(check, 1000)
        })
      }
      setTimeout(check, 2000)
    })
  }

  stop(): void {
    const proc = this.process
    if (proc) {
      let exited = false
      proc.once('exit', () => { exited = true })
      proc.kill('SIGTERM')
      // SIGTERM first; force-kill a stuck process after a grace period.
      setTimeout(() => { if (!exited) proc.kill('SIGKILL') }, 5000)
      this.process = null
    }
    this.status = 'stopped'
  }
}
