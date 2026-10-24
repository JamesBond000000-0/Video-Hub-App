/* SystemJS module definition */
declare var nodeModule: NodeModule;
interface NodeModule {
  id: string;
}

interface DesktopIpcBridge {
  sendToMain: (channel: string, ...args: unknown[]) => void;
  receiveFromMain: (channel: string, callback: (...args: any[]) => void) => void;
  setZoomFactor: (zoomFactor: number) => void;
  clearCache: () => void;
}

interface DesktopFileBridge {
  getPathForFile: (file: File) => string;
}

interface Window {
  process: any;
  require: any;
  myElectron: DesktopIpcBridge;
  myAPI: DesktopFileBridge;
}
