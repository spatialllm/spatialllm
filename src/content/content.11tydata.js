// Directory-data for everything under src/content/.
// Sets a default layout & tag, and computes the public URL by stripping
// the "/content/" prefix and any trailing "/index" segment.
const fs = require("fs");
const path = require("path");

function deriveTitle(raw) {
  if (!raw) return "";
  let body = String(raw);
  if (body.startsWith("---")) {
    const end = body.indexOf("\n---", 3);
    if (end !== -1) body = body.slice(end + 4);
  }
  // First level-1 ATX heading ("# ...") is the page title.
  const lines = body.split(/\r?\n/);
  let heading = "";
  for (const line of lines) {
    const m = /^#\s+(.+?)\s*#*\s*$/.exec(line.trim());
    if (m) {
      heading = m[1];
      break;
    }
  }
  if (!heading) return "";
  return heading
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function deriveDescription(raw, maxLen = 155) {
  if (!raw) return "";
  let body = String(raw);
  if (body.startsWith("---")) {
    const end = body.indexOf("\n---", 3);
    if (end !== -1) body = body.slice(end + 4);
  }
  const paragraphs = body.split(/\n{2,}/);
  let para = "";
  for (const p of paragraphs) {
    const trimmed = p.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("#") || trimmed.startsWith("```")) continue;
    para = trimmed;
    break;
  }
  if (!para) return "";
  const stripped = para
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/&[a-zA-Z#0-9]+;/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (stripped.length <= maxLen) return stripped;
  const cut = stripped.slice(0, maxLen);
  const lastSpace = cut.lastIndexOf(" ");
  const tail = lastSpace > maxLen * 0.6 ? cut.slice(0, lastSpace) : cut;
  return tail.replace(/[,.;:\-—]+$/, "") + "…";
}

module.exports = {
  layout: "article.njk",
  tags: ["article"],
  eleventyComputed: {
    title: (data) => {
      // Respect an explicit front-matter title; otherwise derive it from the
      // first level-1 markdown heading in the page body.
      if (data.title) return data.title;
      const file = data.page && data.page.inputPath;
      if (!file) return "";
      try {
        const raw = fs.readFileSync(path.resolve(file), "utf8");
        return deriveTitle(raw);
      } catch (_) {
        return "";
      }
    },
    permalink: (data) => {
      const stem = data.page.filePathStem || "";
      let clean = stem.replace(/^\/content\//, "/").replace(/\/index$/, "");
      if (!clean.endsWith("/")) clean += "/";
      return clean;
    },
    description: (data) => {
      // If the page already declared a description in its own front matter,
      // respect that. Otherwise derive one from the markdown body.
      if (data.description) return data.description;
      const file = data.page && data.page.inputPath;
      if (!file) return "";
      try {
        const raw = fs.readFileSync(path.resolve(file), "utf8");
        return deriveDescription(raw, 155);
      } catch (_) {
        return "";
      }
    },
  },
};
