/* spatialllm.org — client-side enhancements
   - copy-to-clipboard for code blocks
   - mobile nav toggle
   - togglable task-list checkboxes with localStorage persistence
   - mermaid (lazy-loaded only if needed)
   - smooth scroll respecting sticky header offset
*/

(function () {
  "use strict";

  // ---------- Mobile nav toggle ----------
  const toggle = document.querySelector(".nav-toggle");
  const nav = document.getElementById("primary-nav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.getAttribute("data-open") === "true";
      nav.setAttribute("data-open", String(!open));
      toggle.setAttribute("aria-expanded", String(!open));
    });
    // Close on Escape
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && nav.getAttribute("data-open") === "true") {
        nav.setAttribute("data-open", "false");
        toggle.setAttribute("aria-expanded", "false");
        toggle.focus();
      }
    });
  }

  // ---------- Theme toggle (light / dark, persisted) ----------
  const THEME_KEY = "spatialllm:theme";
  const themeBtn = document.querySelector("[data-theme-toggle]");

  function currentTheme() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "light" || explicit === "dark") return explicit;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (themeBtn) {
      themeBtn.setAttribute(
        "aria-label",
        theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
      );
    }
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#0c141f" : "#1668c4");
  }

  applyTheme(currentTheme());

  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = currentTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem(THEME_KEY, next); } catch (_) {}
    });
  }

  // Follow the OS while the visitor has not made an explicit choice.
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onSchemeChange = (e) => {
      let stored = null;
      try { stored = localStorage.getItem(THEME_KEY); } catch (_) {}
      if (stored === "light" || stored === "dark") return;
      applyTheme(e.matches ? "dark" : "light");
    };
    if (mq.addEventListener) mq.addEventListener("change", onSchemeChange);
    else if (mq.addListener) mq.addListener(onSchemeChange);
  }

  // ---------- Diagram figures: download + full-screen ----------
  // Progressive enhancement — the figure is complete without these controls.
  const PAINT_PROPS = ["fill", "stroke", "stroke-width", "stroke-dasharray", "font-size", "font-weight", "font-family", "opacity", "text-anchor"];

  function themedSvgSource(svg) {
    // Bake the *rendered* colours into a standalone copy so a downloaded file
    // matches the theme on screen (the page CSS is not part of the download).
    const clone = svg.cloneNode(true);
    const originals = svg.querySelectorAll("*");
    const copies = clone.querySelectorAll("*");
    for (let i = 0; i < originals.length && i < copies.length; i++) {
      const cs = getComputedStyle(originals[i]);
      for (const prop of PAINT_PROPS) {
        const value = cs.getPropertyValue(prop);
        if (value && value !== "none" && value !== "normal") copies[i].setAttribute(prop, value.trim());
      }
    }
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + new XMLSerializer().serializeToString(clone);
  }

  function slugForFigure(fig, index) {
    const title = fig.querySelector("svg title");
    const base = (title ? title.textContent : "diagram-" + (index + 1)) || "diagram";
    return base.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "diagram";
  }

  document.querySelectorAll("figure.diagram").forEach((fig, index) => {
    const svg = fig.querySelector("svg");
    if (!svg) return;

    const tools = document.createElement("div");
    tools.className = "figure-tools";

    const dl = document.createElement("button");
    dl.type = "button";
    dl.textContent = "Download SVG";
    dl.addEventListener("click", () => {
      try {
        const blob = new Blob([themedSvgSource(svg)], { type: "image/svg+xml;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = slugForFigure(fig, index) + ".svg";
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 2000);
      } catch (_) {
        dl.textContent = "Download unavailable";
        setTimeout(() => { dl.textContent = "Download SVG"; }, 1800);
      }
    });

    const fs = document.createElement("button");
    fs.type = "button";
    fs.textContent = "Full screen";
    fs.setAttribute("aria-expanded", "false");
    const setExpanded = (on) => {
      fig.classList.toggle("is-fullscreen", on);
      fs.setAttribute("aria-expanded", String(on));
      fs.textContent = on ? "Exit full screen" : "Full screen";
    };
    fs.addEventListener("click", () => setExpanded(!fig.classList.contains("is-fullscreen")));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && fig.classList.contains("is-fullscreen")) {
        setExpanded(false);
        fs.focus();
      }
    });

    tools.appendChild(dl);
    tools.appendChild(fs);
    fig.appendChild(tools);
  });

  // ---------- Copy buttons for code blocks ----------
  document.querySelectorAll(".code-block").forEach((wrap) => {
    const btn = wrap.querySelector(".copy-btn");
    const pre = wrap.querySelector("pre");
    if (!btn || !pre) return;
    btn.addEventListener("click", async () => {
      const text = pre.innerText.replace(/\n$/, "");
      try {
        await navigator.clipboard.writeText(text);
      } catch (_) {
        // Fallback: select & copy via execCommand
        const range = document.createRange();
        range.selectNodeContents(pre);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        try {
          document.execCommand("copy");
        } catch (_) {}
        sel.removeAllRanges();
      }
      btn.textContent = "Copied";
      btn.setAttribute("data-state", "copied");
      setTimeout(() => {
        btn.textContent = "Copy";
        btn.removeAttribute("data-state");
      }, 1500);
    });
  });

  // ---------- Task-list checkboxes (interactive + persisted) ----------
  const storageKey = "spatialllm:checks:" + location.pathname;
  let persisted = {};
  try { persisted = JSON.parse(localStorage.getItem(storageKey) || "{}"); } catch (_) {}

  const taskBoxes = document.querySelectorAll(".task-list-item input[type='checkbox']");
  taskBoxes.forEach((cb, idx) => {
    cb.disabled = false;
    cb.removeAttribute("disabled");
    const key = String(idx);
    if (persisted[key]) {
      cb.checked = true;
      cb.closest(".task-list-item")?.classList.add("is-checked");
    }
    cb.addEventListener("change", () => {
      const li = cb.closest(".task-list-item");
      if (li) li.classList.toggle("is-checked", cb.checked);
      persisted[key] = cb.checked;
      try { localStorage.setItem(storageKey, JSON.stringify(persisted)); } catch (_) {}
    });
  });

  // ---------- Mermaid (lazy load only if needed) ----------
  if (document.querySelector(".mermaid")) {
    const script = document.createElement("script");
    script.src = "/assets/js/mermaid.min.js";
    script.defer = true;
    script.onload = () => {
      if (!window.mermaid) return;
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.mermaid.initialize({
        startOnLoad: false,
        theme: "base",
        fontFamily: "ui-sans-serif, system-ui, sans-serif",
        themeVariables: {
          primaryColor: "#eaf2ff",
          primaryTextColor: "#0e1b2c",
          primaryBorderColor: "#1f7ae0",
          lineColor: "#7c3aed",
          secondaryColor: "#fdf6ec",
          tertiaryColor: "#f3fbf7",
          fontSize: "14px",
        },
        flowchart: { curve: reduceMotion ? "linear" : "basis", htmlLabels: true },
        securityLevel: "strict",
      });
      window.mermaid.run({ querySelector: ".mermaid" }).catch(() => {});
    };
    document.head.appendChild(script);
  }

  // ---------- Header nav panels (topics under each section) ----------
  (function navPanels() {
    const toggles = Array.from(document.querySelectorAll(".nav-panel-toggle"));
    if (!toggles.length) return;

    function panelFor(btn) {
      return document.getElementById(btn.getAttribute("aria-controls"));
    }
    function close(btn) {
      const panel = panelFor(btn);
      if (panel) panel.hidden = true;
      btn.setAttribute("aria-expanded", "false");
    }
    function closeAll(except) {
      toggles.forEach((btn) => { if (btn !== except) close(btn); });
    }

    toggles.forEach((btn) => {
      btn.addEventListener("click", () => {
        const open = btn.getAttribute("aria-expanded") === "true";
        closeAll(btn);
        const panel = panelFor(btn);
        if (panel) panel.hidden = open;
        btn.setAttribute("aria-expanded", open ? "false" : "true");
      });
    });

    document.addEventListener("click", (e) => {
      if (!e.target.closest(".nav-item")) closeAll(null);
    });
    document.addEventListener("keydown", (e) => {
      if (e.key !== "Escape") return;
      const open = toggles.find((b) => b.getAttribute("aria-expanded") === "true");
      if (!open) return;
      close(open);
      open.focus();
    });
  })();

  // ---------- Smooth in-page scroll honouring sticky header ----------
  function getHeaderOffset() {
    const h = document.querySelector(".site-header");
    return (h ? h.getBoundingClientRect().height : 64) + 12;
  }
  document.querySelectorAll('a[href^="#"]').forEach((a) => {
    const href = a.getAttribute("href");
    if (!href || href === "#") return;
    a.addEventListener("click", (e) => {
      const id = decodeURIComponent(href.slice(1));
      const target = document.getElementById(id);
      if (!target) return;
      e.preventDefault();
      const top = target.getBoundingClientRect().top + window.scrollY - getHeaderOffset();
      const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({ top, behavior: reduceMotion ? "auto" : "smooth" });
      history.pushState(null, "", href);
      target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: true });
    });
  });

  // ---------- Card / hero fade-in (respects reduced-motion) ----------
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches && "IntersectionObserver" in window) {
    const items = document.querySelectorAll(".card, .overview__hero, .hero__title, .section__title");
    items.forEach((el) => { el.style.opacity = "0"; el.style.transform = "translateY(8px)"; });
    const io = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.transition = "opacity 400ms ease, transform 400ms ease";
          entry.target.style.opacity = "1";
          entry.target.style.transform = "translateY(0)";
          io.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.05 });
    items.forEach((el) => io.observe(el));
  }
})();
