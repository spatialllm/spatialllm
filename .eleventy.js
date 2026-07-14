const markdownIt = require("markdown-it");
const markdownItAnchor = require("markdown-it-anchor");
const markdownItAttrs = require("markdown-it-attrs");
const markdownItTaskLists = require("markdown-it-task-lists");
const markdownItKatex = require("@iktakahiro/markdown-it-katex");
const syntaxHighlight = require("@11ty/eleventy-plugin-syntaxhighlight");
const slugify = require("slugify");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");

module.exports = function (eleventyConfig) {
  // ---- Passthrough copies ----
  eleventyConfig.addPassthroughCopy({ "src/assets": "assets" });
  eleventyConfig.addPassthroughCopy({ "src/manifest.webmanifest": "manifest.webmanifest" });
  eleventyConfig.addPassthroughCopy({ "src/sw.js": "sw.js" });
  eleventyConfig.addPassthroughCopy({ "src/robots.txt": "robots.txt" });
  // IndexNow key verification file.
  eleventyConfig.addPassthroughCopy({ "src/a775779e180eeae80a89ebd663e6decd.txt": "a775779e180eeae80a89ebd663e6decd.txt" });
  // Cloudflare Pages metadata at output root.
  eleventyConfig.addPassthroughCopy({ "src/_headers": "_headers" });
  // Ship KaTeX CSS + fonts locally (no CDN per project requirements). If the
  // packages are unavailable at build time, these copies are skipped (we ship
  // an empty stub via eleventy.before in that case).
  eleventyConfig.addPassthroughCopy({ "node_modules/katex/dist/katex.min.css": "assets/css/katex.min.css" });
  eleventyConfig.addPassthroughCopy({ "node_modules/katex/dist/fonts": "assets/css/fonts" });
  // Mermaid (local copy)
  eleventyConfig.addPassthroughCopy({ "node_modules/mermaid/dist/mermaid.min.js": "assets/js/mermaid.min.js" });

  // ---- Watch targets ----
  eleventyConfig.addWatchTarget("./src/assets/");

  // ---- Plugins ----
  eleventyConfig.addPlugin(syntaxHighlight);

  // ---- Markdown ----
  const md = markdownIt({
    html: true,
    linkify: true,
    typographer: true,
    breaks: false,
  });

  md.use(markdownItAnchor, {
    slugify: (s) => slugify(s, { lower: true, strict: true }),
    permalink: false,
  });
  md.use(markdownItAttrs);
  md.use(markdownItTaskLists, { enabled: true, label: true, labelAfter: true });
  md.use(markdownItKatex, { throwOnError: false, errorColor: "#cc0000" });

  // Override fence to: (a) emit mermaid blocks as <pre class="mermaid">, (b) wrap others with copy button
  const defaultFence =
    md.renderer.rules.fence ||
    function (tokens, idx, options, env, self) {
      return self.renderToken(tokens, idx, options);
    };

  md.renderer.rules.fence = function (tokens, idx, options, env, self) {
    const token = tokens[idx];
    const info = (token.info || "").trim();
    const lang = info.split(/\s+/)[0];

    if (lang === "mermaid") {
      const content = token.content;
      return `<div class="mermaid-wrap"><pre class="mermaid">${escapeHtml(content)}</pre></div>`;
    }

    let rendered = defaultFence(tokens, idx, options, env, self);
    // Make the (horizontally scrollable) <pre> keyboard-focusable so the code
    // region is reachable without a mouse (WCAG scrollable-region-focusable).
    rendered = rendered.replace(/<pre(\s|>)/, '<pre tabindex="0"$1');
    return `<div class="code-block" data-lang="${escapeHtml(lang || "")}">
  <button type="button" class="copy-btn" data-copy aria-label="Copy code to clipboard">Copy</button>
  ${rendered}
</div>`;
  };

  // Wrap tables in a horizontally scrollable container.
  const defaultTableOpen =
    md.renderer.rules.table_open ||
    function (tokens, idx, options, env, self) {
      return self.renderToken(tokens, idx, options);
    };
  const defaultTableClose =
    md.renderer.rules.table_close ||
    function (tokens, idx, options, env, self) {
      return self.renderToken(tokens, idx, options);
    };
  md.renderer.rules.table_open = function (tokens, idx, options, env, self) {
    return '<div class="table-scroll">' + defaultTableOpen(tokens, idx, options, env, self);
  };
  md.renderer.rules.table_close = function (tokens, idx, options, env, self) {
    return defaultTableClose(tokens, idx, options, env, self) + "</div>";
  };

  eleventyConfig.setLibrary("md", md);

  // ---- Collections ----
  eleventyConfig.addCollection("topics", (api) =>
    api
      .getAll()
      .filter((item) => {
        const url = item.url || "";
        if (!url || url === "/") return false;
        const segments = url.replace(/^\/|\/$/g, "").split("/");
        return segments.length === 1 && item.data.layout === "overview.njk";
      })
      .sort((a, b) => (a.data.order || 0) - (b.data.order || 0))
  );

  eleventyConfig.addCollection("articles", (api) =>
    api.getAll().filter((item) => {
      const layout = item.data.layout;
      return layout === "overview.njk" || layout === "article.njk";
    })
  );

  // ---- Filters ----

  // Cache-bust immutable assets by appending a short content hash as ?v=<hash>.
  // Candidate paths are tried in order; the first that exists is used.
  // This covers both site-owned assets (src/assets/…) and third-party assets
  // copied from node_modules (e.g. katex.min.css, mermaid.min.js).
  const nodeModulesMap = {
    "assets/css/katex.min.css": "node_modules/katex/dist/katex.min.css",
    "assets/js/mermaid.min.js": "node_modules/mermaid/dist/mermaid.min.js",
  };
  eleventyConfig.addFilter("assetHash", (url) => {
    try {
      const rel = url.replace(/^\//, "").split("?")[0];
      const candidates = [
        path.join(__dirname, "src", rel),
        nodeModulesMap[rel] ? path.join(__dirname, nodeModulesMap[rel]) : null,
        path.join(__dirname, "_site", rel),
      ].filter(Boolean);
      const filePath = candidates.find((p) => fs.existsSync(p));
      if (!filePath) return url;
      const hash = crypto
        .createHash("md5")
        .update(fs.readFileSync(filePath))
        .digest("hex")
        .slice(0, 8);
      return `${url}?v=${hash}`;
    } catch {
      return url;
    }
  });

  eleventyConfig.addFilter("breadcrumb", function (url, articles) {
    if (!url || url === "/") return [];
    const segments = url.replace(/^\/|\/$/g, "").split("/");
    const crumbs = [];
    let acc = "";
    for (const seg of segments) {
      acc += "/" + seg;
      const accUrl = acc + "/";
      const match = (articles || []).find((a) => a.url === accUrl);
      crumbs.push({
        url: accUrl,
        title: match ? match.data.title : seg.replace(/-/g, " "),
      });
    }
    return crumbs;
  });

  eleventyConfig.addFilter("siblings", function (current, articles) {
    if (!current || !articles) return [];
    const url = current.url || "";
    const parts = url.replace(/^\/|\/$/g, "").split("/");
    if (parts.length < 1) return [];
    const parentParts = parts.slice(0, -1);
    const parentPrefix = parentParts.length ? "/" + parentParts.join("/") + "/" : "/";
    return articles.filter((a) => {
      if (a.url === url) return false;
      const aParts = a.url.replace(/^\/|\/$/g, "").split("/");
      if (aParts.length !== parts.length) return false;
      const aParentPrefix =
        aParts.length > 1 ? "/" + aParts.slice(0, -1).join("/") + "/" : "/";
      return aParentPrefix === parentPrefix;
    });
  });

  eleventyConfig.addFilter("children", function (current, articles) {
    if (!current || !articles) return [];
    const url = current.url || "/";
    return articles.filter((a) => {
      if (a.url === url) return false;
      if (!a.url.startsWith(url)) return false;
      const rest = a.url.slice(url.length).replace(/\/$/, "");
      return rest.length > 0 && !rest.includes("/");
    });
  });

  eleventyConfig.addFilter("parentUrl", function (url) {
    if (!url || url === "/") return null;
    const parts = url.replace(/^\/|\/$/g, "").split("/");
    if (parts.length <= 1) return "/";
    return "/" + parts.slice(0, -1).join("/") + "/";
  });

  eleventyConfig.addFilter("excerpt", function (content, words = 30) {
    if (!content) return "";
    const text = content
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    const parts = text.split(" ");
    if (parts.length <= words) return text;
    return parts.slice(0, words).join(" ") + "…";
  });

  // Generate a meta-description-safe summary from raw markdown content:
  // strip front matter, take the first prose paragraph, strip markdown
  // syntax, normalize whitespace and trim to ~155 chars on a word boundary.
  eleventyConfig.addFilter("metaDescription", function (raw, maxLen = 155) {
    if (!raw) return "";
    let body = String(raw);
    if (body.startsWith("---")) {
      const end = body.indexOf("\n---", 3);
      if (end !== -1) body = body.slice(end + 4);
    }
    // Skip leading blank lines / headings / code fences and grab the first prose paragraph.
    const paragraphs = body.split(/\n{2,}/);
    let para = "";
    for (const p of paragraphs) {
      const trimmed = p.trim();
      if (!trimmed) continue;
      if (trimmed.startsWith("#")) continue;
      if (trimmed.startsWith("```")) continue;
      para = trimmed;
      break;
    }
    if (!para) return "";
    const stripped = para
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")    // [text](url) → text
      .replace(/`([^`]+)`/g, "$1")                // `code` → code
      .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, "$1") // **bold** / *italic* → text
      .replace(/<[^>]+>/g, " ")                   // raw HTML
      .replace(/&[a-zA-Z#0-9]+;/g, " ")           // entities
      .replace(/\s+/g, " ")
      .trim();
    if (stripped.length <= maxLen) return stripped;
    const cut = stripped.slice(0, maxLen);
    const lastSpace = cut.lastIndexOf(" ");
    return (lastSpace > maxLen * 0.6 ? cut.slice(0, lastSpace) : cut).replace(/[,.;:\-—]+$/, "") + "…";
  });

  eleventyConfig.addFilter("dateISO", () => new Date().toISOString());

  eleventyConfig.addFilter("currentYear", () => new Date().getFullYear());

  // ---- Transforms ----
  // The page title is rendered in the header (<h1 class="...__title">) from the
  // derived `title`. Markdown bodies also open with their own "# ..." heading,
  // which markdown-anchor turns into <h1 id="..." tabindex="-1">…</h1>. Strip
  // that first body-level <h1> so each page has exactly one visible title and
  // no duplicated heading.
  eleventyConfig.addTransform("dropBodyH1", function (content, outputPath) {
    if (!outputPath || !outputPath.endsWith(".html")) return content;
    return content.replace(/\s*<h1\b[^>]*\bid="[^"]*"[^>]*>[\s\S]*?<\/h1>/, "");
  });

  // ---- Layout aliases ----
  eleventyConfig.addLayoutAlias("base", "base.njk");
  eleventyConfig.addLayoutAlias("overview", "overview.njk");
  eleventyConfig.addLayoutAlias("article", "article.njk");

  return {
    dir: {
      input: "src",
      includes: "_includes",
      data: "_data",
      output: "_site",
    },
    markdownTemplateEngine: false,
    htmlTemplateEngine: "njk",
    templateFormats: ["njk", "md", "html", "11ty.js"],
  };
};

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
