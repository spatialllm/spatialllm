/* spatialllm.org — service worker
   Strategies:
     - Pre-cache app shell on install.
     - Cache-first for hashed / versioned assets under /assets/*.
     - Stale-while-revalidate for HTML navigations (fast + auto-refreshing).
     - Network-only for everything else.
*/

const VERSION = "v1.0.0";
const CACHE_STATIC = `spatialllm-static-${VERSION}`;
const CACHE_PAGES  = `spatialllm-pages-${VERSION}`;

const APP_SHELL = [
  "/",
  "/assets/css/site.css",
  "/assets/css/katex.min.css",
  "/assets/js/site.js",
  "/assets/img/logo.svg",
  "/assets/img/favicon.svg",
  "/assets/img/icon-maskable.svg",
  "/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_STATIC).then((cache) =>
      cache.addAll(APP_SHELL.map((u) => new Request(u, { cache: "reload" })))
        .catch(() => undefined)
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== CACHE_STATIC && k !== CACHE_PAGES)
          .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  // HTML navigations: stale-while-revalidate.
  if (req.mode === "navigate" || (req.headers.get("accept") || "").includes("text/html")) {
    event.respondWith(staleWhileRevalidate(req, CACHE_PAGES));
    return;
  }

  // Static assets: cache-first.
  if (url.pathname.startsWith("/assets/") || url.pathname === "/manifest.webmanifest") {
    event.respondWith(cacheFirst(req, CACHE_STATIC));
    return;
  }

  // Default: network with cache fallback.
  event.respondWith(
    fetch(req).catch(() => caches.match(req))
  );
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response && response.ok) cache.put(request, response.clone());
    return response;
  } catch (err) {
    return cached || Response.error();
  }
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  const network = fetch(request)
    .then((response) => {
      if (response && response.ok) cache.put(request, response.clone());
      return response;
    })
    .catch(() => null);
  return cached || (await network) || cache.match("/") || Response.error();
}
