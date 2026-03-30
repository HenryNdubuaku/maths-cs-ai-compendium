import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { readdir, readFile } from "node:fs/promises";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

// Resolve compendium root: env var > relative to this script (mcp/src/index.ts → ../../)
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.COMPENDIUM_ROOT || join(__dirname, "..", "..");

// --- Filesystem helpers ---

const CHAPTER_RE = /^chapter (\d{2}): (.+)$/;
const SECTION_RE = /^(\d{2})\. (.+)\.md$/;

interface Chapter {
  number: number;
  name: string;
  path: string;
}

interface Section {
  number: number;
  name: string;
  path: string;
}

async function getChapters(): Promise<Chapter[]> {
  const entries = await readdir(ROOT);
  return entries
    .map((entry) => {
      const match = entry.match(CHAPTER_RE);
      if (!match) return null;
      return { number: parseInt(match[1], 10), name: match[2], path: join(ROOT, entry) };
    })
    .filter((ch): ch is Chapter => ch !== null)
    .sort((a, b) => a.number - b.number);
}

async function getSections(chapterPath: string): Promise<Section[]> {
  const entries = await readdir(chapterPath);
  return entries
    .map((entry) => {
      const match = entry.match(SECTION_RE);
      if (!match) return null;
      return { number: parseInt(match[1], 10), name: match[2], path: join(chapterPath, entry) };
    })
    .filter((s): s is Section => s !== null)
    .sort((a, b) => a.number - b.number);
}

// --- llms.txt parser for recommend tool ---

interface SectionMeta {
  chapter: number;
  chapterName: string;
  section: number;
  sectionName: string;
  description: string;
}

async function parseLlmsTxt(): Promise<SectionMeta[]> {
  const content = await readFile(join(ROOT, "llms.txt"), "utf-8");
  const results: SectionMeta[] = [];
  let currentChapter = 0;
  let currentChapterName = "";

  for (const line of content.split("\n")) {
    const chapterMatch = line.match(/^### Chapter (\d+): (.+)$/);
    if (chapterMatch) {
      currentChapter = parseInt(chapterMatch[1], 10);
      currentChapterName = chapterMatch[2];
      continue;
    }

    const sectionMatch = line.match(/^- \[(.+?)\]\(.+?\): (.+)$/);
    if (sectionMatch && currentChapter > 0) {
      results.push({
        chapter: currentChapter,
        chapterName: currentChapterName,
        section: results.filter((r) => r.chapter === currentChapter).length + 1,
        sectionName: sectionMatch[1],
        description: sectionMatch[2],
      });
    }
  }
  return results;
}

const STOP_WORDS = new Set([
  "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one",
  "our", "out", "has", "how", "its", "may", "who", "did", "get", "got", "let", "say", "she",
  "too", "use", "what", "why", "when", "where", "which", "with", "would", "could", "should",
  "about", "after", "been", "being", "between", "both", "does", "doing", "during", "each",
  "from", "have", "into", "just", "know", "like", "make", "more", "most", "much", "need",
  "only", "other", "over", "some", "such", "take", "than", "that", "them", "then", "there",
  "these", "they", "this", "very", "want", "well", "were", "will", "work", "your",
  "understand", "learn", "explain", "tell", "help",
]);

// --- MCP Server ---

const server = new McpServer({
  name: "compendium",
  version: "1.0.0",
});

// Tool: list_topics
server.registerTool(
  "list_topics",
  {
    description: "List all chapters and sections in the compendium, or filter to a specific chapter",
    inputSchema: { chapter: z.number().optional().describe("Filter to a specific chapter number (1-20)") },
  },
  async ({ chapter }) => {
    const chapters = await getChapters();
    const filtered = chapter ? chapters.filter((ch) => ch.number === chapter) : chapters;

    if (filtered.length === 0) {
      return { content: [{ type: "text", text: `Chapter ${chapter} not found. Valid chapters: 1-${chapters.length}.` }] };
    }

    const lines: string[] = [];
    for (const ch of filtered) {
      const sections = await getSections(ch.path);
      lines.push(`\n## Chapter ${ch.number}: ${ch.name}`);
      for (const sec of sections) {
        lines.push(`  ${sec.number}. ${sec.name}`);
      }
    }

    return { content: [{ type: "text", text: lines.join("\n").trim() }] };
  },
);

// Tool: read_section
server.registerTool(
  "read_section",
  {
    description: "Read the full content of a specific section from the compendium",
    inputSchema: {
      chapter: z.number().describe("Chapter number (1-20)"),
      section: z.number().describe("Section number (typically 0-7, varies by chapter)"),
    },
  },
  async ({ chapter, section }) => {
    const chapters = await getChapters();
    const ch = chapters.find((c) => c.number === chapter);
    if (!ch) {
      return { content: [{ type: "text", text: `Chapter ${chapter} not found. Valid chapters: 1-${chapters.length}.` }] };
    }

    const sections = await getSections(ch.path);
    const sec = sections.find((s) => s.number === section);
    if (!sec) {
      const valid = sections.map((s) => s.number).join(", ");
      return { content: [{ type: "text", text: `Section ${section} not found in Chapter ${chapter}: ${ch.name}. Valid sections: ${valid}.` }] };
    }

    const content = await readFile(sec.path, "utf-8");
    return { content: [{ type: "text", text: `# Chapter ${ch.number}: ${ch.name} — ${sec.name}\n\n${content}` }] };
  },
);

// Tool: search
server.registerTool(
  "search",
  {
    description: "Search across all compendium sections for a term or phrase",
    inputSchema: { query: z.string().describe("Search term or phrase to find across all sections") },
  },
  async ({ query }) => {
    const chapters = await getChapters();
    const results: string[] = [];
    const lowerQuery = query.toLowerCase();

    for (const ch of chapters) {
      const sections = await getSections(ch.path);
      for (const sec of sections) {
        const content = await readFile(sec.path, "utf-8");
        const lines = content.split("\n");
        const matches: string[] = [];

        for (let i = 0; i < lines.length; i++) {
          if (lines[i].toLowerCase().includes(lowerQuery)) {
            const start = Math.max(0, i - 2);
            const end = Math.min(lines.length - 1, i + 2);
            const excerpt = lines.slice(start, end + 1).join("\n");
            matches.push(`  Line ${i + 1}:\n${excerpt}`);
          }
        }

        if (matches.length > 0) {
          results.push(`### Chapter ${ch.number}: ${ch.name} — ${sec.name}\n${matches.slice(0, 3).join("\n\n")}`);
        }
      }

      if (results.length >= 20) break;
    }

    if (results.length === 0) {
      return { content: [{ type: "text", text: `No results found for "${query}".` }] };
    }

    return { content: [{ type: "text", text: `Found matches in ${results.length} sections:\n\n${results.join("\n\n")}` }] };
  },
);

// Tool: recommend
server.registerTool(
  "recommend",
  {
    description: "Given a learning goal or question, recommend the most relevant compendium sections in suggested reading order",
    inputSchema: { query: z.string().describe("A learning goal or question, e.g. 'How do transformers work?' or 'What math do I need for ML?'") },
  },
  async ({ query }) => {
    const meta = await parseLlmsTxt();
    const keywords = query
      .toLowerCase()
      .split(/\W+/)
      .filter((w) => w.length > 2 && !STOP_WORDS.has(w));

    if (keywords.length === 0) {
      return { content: [{ type: "text", text: "Could not extract meaningful keywords from your query. Try using specific technical terms." }] };
    }

    const scored = meta.map((entry) => {
      const descLower = entry.description.toLowerCase();
      const nameLower = `${entry.chapterName} ${entry.sectionName}`.toLowerCase();

      let score = 0;
      for (const kw of keywords) {
        if (descLower.includes(kw)) score += 2;
        if (nameLower.includes(kw)) score += 3;
      }
      return { ...entry, score };
    });

    const matches = scored
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score || a.chapter - b.chapter)
      .slice(0, 15);

    if (matches.length === 0) {
      return { content: [{ type: "text", text: `No relevant sections found for "${query}". Try broader terms or use search for exact matches.` }] };
    }

    // Group by chapter and sort by chapter order for reading sequence
    const byChapter = new Map<number, typeof matches>();
    for (const m of matches) {
      if (!byChapter.has(m.chapter)) byChapter.set(m.chapter, []);
      byChapter.get(m.chapter)!.push(m);
    }

    const lines: string[] = ["Recommended sections (in suggested reading order):\n"];
    for (const [chNum, sections] of [...byChapter.entries()].sort((a, b) => a[0] - b[0])) {
      const ch = sections[0];
      sections.sort((a, b) => a.section - b.section);
      lines.push(`## Chapter ${chNum}: ${ch.chapterName}`);
      for (const sec of sections) {
        lines.push(`  ${sec.section}. ${sec.sectionName} — ${sec.description}`);
      }
      lines.push("");
    }

    return { content: [{ type: "text", text: lines.join("\n").trim() }] };
  },
);

// Tool: get_examples
server.registerTool(
  "get_examples",
  {
    description: "Extract code examples from the compendium, optionally filtered by topic or language. Returns implementation code with surrounding explanation.",
    inputSchema: {
      query: z.string().optional().describe("Topic to find examples for, e.g. 'attention mechanism' or 'CUDA kernel'"),
      language: z.string().optional().describe("Filter by programming language, e.g. 'python', 'cpp', 'bash'"),
      chapter: z.number().optional().describe("Filter to a specific chapter number (1-20)"),
    },
  },
  async ({ query, language, chapter }) => {
    const chapters = await getChapters();
    const filtered = chapter ? chapters.filter((ch) => ch.number === chapter) : chapters;
    const lowerQuery = query?.toLowerCase();
    const results: string[] = [];

    for (const ch of filtered) {
      const sections = await getSections(ch.path);
      for (const sec of sections) {
        const content = await readFile(sec.path, "utf-8");
        const lines = content.split("\n");

        for (let i = 0; i < lines.length; i++) {
          const openMatch = lines[i].match(/^```(\w*)$/);
          if (!openMatch) continue;

          const lang = openMatch[1] || "text";
          if (language && lang !== language) continue;

          // Find closing fence
          let end = i + 1;
          while (end < lines.length && lines[end] !== "```") end++;

          const code = lines.slice(i + 1, end).join("\n");
          if (!code.trim()) continue;

          // Get context: up to 3 lines before the code block for explanation
          const ctxStart = Math.max(0, i - 3);
          const context = lines.slice(ctxStart, i).filter((l) => l.trim()).join("\n");

          // Check if query matches the code or surrounding context
          if (lowerQuery) {
            const searchable = `${context} ${code}`.toLowerCase();
            if (!searchable.includes(lowerQuery)) continue;
          }

          results.push(
            `### Chapter ${ch.number}: ${ch.name} — ${sec.name}\n` +
              (context ? `${context}\n\n` : "") +
              `\`\`\`${lang}\n${code}\n\`\`\``,
          );

          if (results.length >= 10) break;
          i = end; // Skip past this block
        }
        if (results.length >= 10) break;
      }
      if (results.length >= 10) break;
    }

    if (results.length === 0) {
      const filters = [query && `topic "${query}"`, language && `language "${language}"`, chapter && `chapter ${chapter}`].filter(Boolean).join(", ");
      return { content: [{ type: "text", text: `No code examples found for ${filters || "the given filters"}.` }] };
    }

    return { content: [{ type: "text", text: `Found ${results.length} code examples:\n\n${results.join("\n\n---\n\n")}` }] };
  },
);

// --- Start ---

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Compendium MCP server running on stdio");
}

main().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
