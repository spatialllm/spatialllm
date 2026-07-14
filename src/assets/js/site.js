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
