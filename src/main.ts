import { enableProdMode } from '@angular/core';
import { platformBrowserDynamic } from '@angular/platform-browser-dynamic';

import { AppModule } from './app/app.module';
import { AppConfig } from './environments/environment';
import { installTauriBridge } from './desktop/tauri-bridge';

// Electron exposes this API from node/preload.js. Tauri installs the same
// contract here so the Angular application does not need runtime-specific calls.
installTauriBridge();

if (AppConfig.production) {
  enableProdMode();
}

platformBrowserDynamic()
  .bootstrapModule(AppModule, {
    preserveWhitespaces: false
  })
  .catch(err => console.error(err));
