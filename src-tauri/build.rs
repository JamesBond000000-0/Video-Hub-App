fn main() {
  // Workaround for a missing DEP_TAURI_DEV in some CI unit graphs.
  //
  // tauri_build::is_dev() reads the `cargo:dev` value emitted by the `tauri`
  // crate's build script through the DEP_TAURI_DEV env var, and panics when
  // the variable is absent. Cargo is supposed to run ta's build script first
  // and pass its links metadata to this build script, but in the Windows and
  // macOS jobs of the tauri-v3.3.0-test.3 release this build script ran
  // before ta's build script ever executed, so the variable was never set
  // (the same tree built fine on Linux and with the smaller test.2 graph).
  //
  // When the variable is absent, derive the same value from the build
  // profile: the Tauri CLI runs dev builds with the debug profile (no
  // custom-protocol, dev = true) and production builds with the release
  // profile (custom-protocol enabled, dev = false). When cargo does
  // provide the variable, this fallback is a no-op.
  if std::env::var_os("DEP_TAURI_DEV").is_none() {
    let dev = std::env::var("PROFILE").map(|p| p == "debug").unwrap_or(false);
    // SAFETY: this runs at the very start of the build script's main, before
    // tauri_build spawns any threads, so no other thread can be reading the
    // environment concurrently.
    unsafe {
      std::env::set_var("DEP_TAURI_DEV", if dev { "true" } else { "false" });
    }
  }
  tauri_build::build()
}
