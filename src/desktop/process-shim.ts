/**
 * Some modules bundled for the Electron renderer (notably the `path`
 * package, imported by file-path.service.ts) reference the bare Node
 * global `process` at module-evaluation time. Electron always defines
 * `process` in the renderer, but a plain Tauri WebView does not, so the
 * main bundle throws `ReferenceError: Can't find variable: process`
 * before Angular can bootstrap. Define the minimal shape those modules
 * touch. This module must be imported before any module that may pull
 * in Node-flavoured packages.
 */
const globalScope = window as unknown as { process?: unknown };

if (typeof globalScope.process === 'undefined') {
  const userAgent = navigator.userAgent;
  globalScope.process = {
    platform: userAgent.includes('Windows')
      ? 'win32'
      : userAgent.includes('Macintosh')
        ? 'darwin'
        : 'linux',
    env: {},
    cwd: (): string => '/',
    nextTick: (callback: (...args: unknown[]) => void, ...args: unknown[]): void => {
      setTimeout(() => callback(...args), 0);
    },
  };
}
