// App shell available offline: network first (always the latest app when the box is reachable), cache fallback.
// Only the shell is ever cached: API responses (patient data) never go to the browser cache, which is not
// encrypted — patient data lives only in the encrypted IndexedDB.
const C = "dayone-v3", SHELL = ["./", "index.html", "app.js", "i18n.js", "schema.json", "lifecycle.json", "manifest.webmanifest"];
const SHELL_PATHS = new Set(SHELL.map(p => new URL(p, self.registration.scope).pathname));
self.addEventListener("install", e => e.waitUntil(caches.open(C).then(c => c.addAll(SHELL)).then(() => self.skipWaiting())));
self.addEventListener("activate", e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== C).map(k => caches.delete(k))))));
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== self.location.origin || !SHELL_PATHS.has(url.pathname)) return;
  e.respondWith(fetch(e.request).then(r => {
    if (r.ok) { const copy = r.clone(); caches.open(C).then(c => c.put(url.pathname, copy)); }
    return r;
  }).catch(() => caches.match(url.pathname)));
});
