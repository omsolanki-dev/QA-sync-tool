import * as fs from 'fs';
import * as path from 'path';
import * as mammoth from 'mammoth';
import * as dotenv from 'dotenv';
import * as crypto from 'crypto';
import { GoogleGenerativeAI } from '@google/generative-ai';

dotenv.config();

export interface Requirement {
  id: string; // e.g. "R1"
  title: string;
  description: string;
  goal?: string;
  expectedResult?: string;
  rawText: string;
  fingerprint: string;
}

export function generateRequirementFingerprint(req: Omit<Requirement, 'fingerprint'>): string {
  const normField = (s?: string) =>
    (s || '')
      .toLowerCase()
      .replace(/[^a-z0-9]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  const combined = [
    normField(req.title),
    normField(req.goal),
    normField(req.description),
    normField(req.expectedResult)
  ].join('|');
  return crypto.createHash('sha256').update(combined).digest('hex');
}

export function generateTestFingerprint(fullText: string): string {
  const cleaned = fullText
    .toLowerCase()
    .replace(/[^a-z0-9]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  return crypto.createHash('sha256').update(cleaned).digest('hex');
}

export interface BlockStructure {
  type: 'requirement' | 'generic';
  requirementId?: string;
  requirementIds?: string[];
  code: string;
}

export interface TestCaseStructure {
  title: string;
  fullText: string;
  header: string;
  blocks: BlockStructure[];
  footer: string;
  startLine: number;
  endLine: number;
  matchIndex?: number;
}

export type FileSegment =
  | { type: 'text'; content: string }
  | { type: 'testCase'; structure: TestCaseStructure };

export interface ParsedTest {
  title: string;
  requirementId: string | null;
  requirementIds: string[];
  fullText: string;
  startLine: number;
  endLine: number;
  filePath: string;
  hasHardcodedUrl: boolean;
  hasHardcodedLocator: boolean;
  hasHardcodedSecret: boolean;
  detectedUrls: string[];
  detectedLocators: string[];
  detectedSecrets: { selector: string; value: string }[];
  requirementBlocks: { id: string; code: string }[];
  fingerprint: string;
}

/**
 * Extracts raw text from .docx files using mammoth.
 */
/**
 * Extracts content from .docx files, converting tables to Markdown pipe-format
 * so the local parser can handle merged cells, multi-column layouts, etc.
 * Falls back to raw text extraction if HTML conversion fails.
 */
async function readDocxContent(filePath: string): Promise<string> {
  try {
    // Convert to HTML to preserve table structure
    const htmlResult = await mammoth.convertToHtml({ path: filePath });
    const html = htmlResult.value;

    // Convert HTML tables → Markdown pipe tables
    const tableToMarkdown = (tableHtml: string): string => {
      const rows: string[][] = [];
      const rowMatches = tableHtml.match(/<tr[^>]*>([\s\S]*?)<\/tr>/gi) || [];
      for (const row of rowMatches) {
        const cells: string[] = [];
        const cellMatches = row.match(/<t[dh][^>]*>([\s\S]*?)<\/t[dh]>/gi) || [];
        for (const cell of cellMatches) {
          const text = cell
            .replace(/<br\s*\/?>/gi, ' ')
            .replace(/<[^>]+>/g, '')
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&nbsp;/g, ' ')
            .replace(/&quot;/g, '"')
            .replace(/&#39;/g, "'")
            .replace(/\s+/g, ' ')
            .trim();
          cells.push(text);
        }
        if (cells.length > 0) rows.push(cells);
      }
      if (rows.length === 0) return '';
      const maxCols = Math.max(...rows.map(r => r.length));
      const lines: string[] = [];
      rows.forEach((row, idx) => {
        // Pad row to maxCols
        while (row.length < maxCols) row.push('');
        lines.push('| ' + row.join(' | ') + ' |');
        if (idx === 0) {
          lines.push('| ' + Array(maxCols).fill('---').join(' | ') + ' |');
        }
      });
      return lines.join('\n');
    };

    // Replace each <table> block with a markdown table
    let markdown = html.replace(/<table[^>]*>([\s\S]*?)<\/table>/gi, (match) => {
      return '\n' + tableToMarkdown(match) + '\n';
    });

    // Convert headings
    markdown = markdown.replace(/<h([1-6])[^>]*>([\s\S]*?)<\/h[1-6]>/gi, (_, level, content) => {
      const text = content.replace(/<[^>]+>/g, '').trim();
      return '\n' + '#'.repeat(Number(level)) + ' ' + text + '\n';
    });

    // Convert list items
    markdown = markdown.replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, (_, content) => {
      const text = content.replace(/<[^>]+>/g, '').trim();
      return '- ' + text + '\n';
    });

    // Convert paragraphs and breaks
    markdown = markdown
      .replace(/<p[^>]*>([\s\S]*?)<\/p>/gi, (_, content) => {
        const text = content.replace(/<br\s*\/?>/gi, '\n').replace(/<[^>]+>/g, '').trim();
        return text ? text + '\n' : '';
      })
      .replace(/<br\s*\/?>/gi, '\n')
      .replace(/<strong[^>]*>([\s\S]*?)<\/strong>/gi, '$1')
      .replace(/<em[^>]*>([\s\S]*?)<\/em>/gi, '$1')
      .replace(/<[^>]+>/g, '') // strip remaining tags
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&nbsp;/g, ' ')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/[ \t]+/g, ' ')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    return markdown.length > 50 ? markdown : (await mammoth.extractRawText({ path: filePath })).value;
  } catch (e: any) {
    // Fallback to plain text extraction
    try {
      const result = await mammoth.extractRawText({ path: filePath });
      return result.value;
    } catch (e2: any) {
      console.error(`Error reading docx file ${filePath}:`, e2.message || e2);
      return '';
    }
  }
}

export function getFilesRecursively(dir: string, filterFunc: (name: string, isDir: boolean) => boolean, skipDirs: string[] = []): string[] {
  const results: string[] = [];
  try {
    const list = fs.readdirSync(dir, { withFileTypes: true });
    for (const item of list) {
      const fullPath = path.join(dir, item.name);
      if (item.isDirectory()) {
        if (!skipDirs.includes(item.name)) {
          results.push(...getFilesRecursively(fullPath, filterFunc, skipDirs));
        }
      } else if (item.isFile() && filterFunc(item.name, false)) {
        results.push(fullPath);
      }
    }
  } catch (e) {
    // ignore read errors
  }
  return results;
}

function extractTextFromDoc(buffer: Buffer): string {
  // Try to decode both ascii and utf16le, and pick the one with most printable words
  let asciiText = '';
  for (let i = 0; i < buffer.length; i++) {
    const char = buffer[i];
    if ((char >= 32 && char <= 126) || char === 10 || char === 13 || char === 9) {
      asciiText += String.fromCharCode(char);
    } else {
      asciiText += ' ';
    }
  }

  let utf16Text = '';
  for (let i = 0; i < buffer.length - 1; i += 2) {
    const code = buffer.readUInt16LE(i);
    if ((code >= 32 && code <= 126) || code === 10 || code === 13 || code === 9) {
      utf16Text += String.fromCharCode(code);
    } else if (code >= 0x00A0 && code <= 0x00FF) {
      utf16Text += String.fromCharCode(code);
    } else {
      utf16Text += ' ';
    }
  }

  const countWords = (text: string) => (text.match(/\b[a-zA-Z]{3,}\b/g) || []).length;
  const asciiWordCount = countWords(asciiText);
  const utf16WordCount = countWords(utf16Text);

  const rawText = utf16WordCount > asciiWordCount ? utf16Text : asciiText;

  // Clean up excessive spaces and return
  return rawText.replace(/[ \t]+/g, ' ').replace(/\n\s*\n/g, '\n').trim();
}

/**
 * Automatically detects the requirement document in the project.
 * Strictly prefers Word (.docx or .doc) files. Only falls back to .md/.txt if NO Word is found.
 * Looks in project root and subdirectories recursively.
 */
export async function findRequirementDocument(projectRoot: string): Promise<{ filePath: string; content: string } | null> {
  const skipDirs = ['node_modules', 'dist', 'build', '.git', '.github', 'test-results', 'playwright-report', 'qa-sync-tool', 'scratch'];
  const candidates = getFilesRecursively(
    projectRoot,
    (name) => {
      if (name.startsWith('~$')) return false;
      const ext = path.extname(name).toLowerCase();
      const lowerName = name.toLowerCase();
      return lowerName === 'requirement.docs' || ext === '.docx' || ext === '.doc' || ext === '.md' || ext === '.txt';
    },
    skipDirs
  );

  // Sort: Word documents with "requirement" in name first, then any Word, then .md/.txt fallbacks
  candidates.sort((a, b) => {
    const aExt = path.extname(a).toLowerCase();
    const bExt = path.extname(b).toLowerCase();
    const aIsWord = aExt === '.docx' || aExt === '.doc';
    const bIsWord = bExt === '.docx' || bExt === '.doc';

    // Word always beats non-Word
    if (aIsWord && !bIsWord) return -1;
    if (!aIsWord && bIsWord) return 1;

    // Within same extension, files with "requirement" in name come first
    const aReq = a.toLowerCase().includes('requirement');
    const bReq = b.toLowerCase().includes('requirement');
    if (aReq && !bReq) return -1;
    if (!aReq && bReq) return 1;

    return a.localeCompare(b);
  });

  // If we found any Word file, drop all non-Word candidates entirely
  const hasWord = candidates.some(c => {
    const ext = path.extname(c).toLowerCase();
    return ext === '.docx' || ext === '.doc';
  });
  const finalCandidates = hasWord
    ? candidates.filter(c => {
      const ext = path.extname(c).toLowerCase();
      return ext === '.docx' || ext === '.doc';
    })
    : candidates;

  if (finalCandidates.length > 0) {
    const filePath = finalCandidates[0];
    const ext = path.extname(filePath).toLowerCase();
    let content = '';

    if (ext === '.docx') {
      content = await readDocxContent(filePath);
      console.log(`  ✓ Reading DOCX document line-by-line (mammoth extraction)...`);
    } else if (ext === '.doc') {
      const buffer = fs.readFileSync(filePath);
      content = extractTextFromDoc(buffer);
      console.log(`  ✓ Reading binary DOC document line-by-line (heuristic text extraction)...`);
    } else {
      content = fs.readFileSync(filePath, 'utf-8');
    }

    return {
      filePath,
      content,
    };
  }

  return null;
}

/**
 * Extracts requirements from the document content.
 * Looks for IDs matching R1, R2, TC-XXX, etc. and groups their titles and descriptions.
 *
 * FIX (root cause of false REMOVE deletions / brittle parsing):
 * Docx -> markdown extraction (mammoth/pandoc) frequently wraps headings in
 * "**bold**" markers and prefixes bullet lines with multi-codepoint emoji
 * (e.g. "🔹"). The old regex tried to swallow all of that in a single
 * optional prefix group, which is brittle the moment a heading mixes
 * markers in an unexpected order (e.g. "**## Search **&** Listing
 * Requirements**" could be partially matched and pollute results, or a
 * subtly different bullet style could silently fail to match at all).
 *
 * The fix: strip known markdown/emoji/bullet noise from a CLONED line
 * before testing it against a much simpler, stricter header pattern that
 * must match starting at position 0. The original line is preserved
 * untouched in `rawText` so nothing about the source document is lost.
 */
async function callWithRetry<T>(fn: () => Promise<T>, maxRetries = 3, delayMs = 1000): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (error: any) {
      attempt++;
      const is404 = error.status === 404 ||
        (error.message && (error.message.includes('404') || error.message.toLowerCase().includes('not found') || error.message.toLowerCase().includes('not supported')));
      if (is404) {
        throw error;
      }
      const msgLower = (error.message || '').toLowerCase();
      const isQuotaLimit = error.status === 429 ||
        msgLower.includes('quota') ||
        msgLower.includes('freetier') ||
        msgLower.includes('billing') ||
        msgLower.includes('limit: 20') ||
        msgLower.includes('limit: 0') ||
        msgLower.includes('too many requests');
      if (isQuotaLimit) {
        throw error;
      }
      if (attempt >= maxRetries) {
        throw error;
      }
      await new Promise(resolve => setTimeout(resolve, delayMs));
      delayMs *= 2;
    }
  }
}

/**
 * Extracts requirements from a plain-text document using purely local heuristics.
 *
 * Supports ALL requirement document formats:
 *   PATH A — Explicit IDs    : FR-01, R1, R-01, TC-01 with title + body
 *           — Table rows      : | ID | Title | Description | cells
 *   PATH B — Structured nums  : I. Title, 1. Title, A. Title headings (+ sub-bullets as description)
 *   PATH C — Pure prose       : blank-line-separated paragraphs → each = one TEMP requirement
 *   Mixed  : any combination handled automatically
 *
 * Key rules:
 *   - Section headings (ALL-CAPS, ends-with-colon) are excluded as standalone requirements.
 *   - 'Expected Result' subsections are merged verbatim into the parent requirement description.
 *   - Sub-bullets within a heading section belong to that heading's description.
 *   - Output is deterministic: identical input → identical output.
 */
/**
 * Normalizes a list/bullet prefix to detect the list type.
 * Returns { type, marker } where type is 'roman'|'alpha'|'numeric'|'bullet'|null.
 */
function detectListPrefix(trimmed: string): { type: string; label: string; rest: string } | null {
  // Ignore compound sub-prefixes like a.i, 1.a, 1.1, a.1, etc.
  if (/^[A-Za-z0-9]+[.)][A-Za-z0-9]/i.test(trimmed)) {
    return null;
  }

  // Roman numerals (upper or lower): I. II. IV. i. ii. iv.
  const romanUpper = /^((?:M{0,4})(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))[.)\s]+(.+)$/i;
  const romanMatch = trimmed.match(romanUpper);
  if (romanMatch && romanMatch[1].toUpperCase() !== romanMatch[1].toLowerCase()) {
    return { type: 'roman', label: romanMatch[1].toUpperCase(), rest: romanMatch[2].trim() };
  }
  // Alphabetic: A. B. a. b. A) B)
  const alphaMatch = trimmed.match(/^([A-Za-z])[.)\s]+(.+)$/);
  if (alphaMatch && !/^[IVXLCDMivxlcdm]+$/.test(alphaMatch[1])) {
    return { type: 'alpha', label: alphaMatch[1].toUpperCase(), rest: alphaMatch[2].trim() };
  }
  // Numeric: 1. 2. 1) 2) 1.1 2.3
  const numMatch = trimmed.match(/^(\d+(?:\.\d+)*)[.)\s]+(.+)$/);
  if (numMatch) {
    return { type: 'numeric', label: numMatch[1], rest: numMatch[2].trim() };
  }
  // Bullets: - * • ◦ ▪ ▸ ➤ ➔ → ✓ ✗ ✔ ►
  const bulletMatch = trimmed.match(/^[-*•◦▪▸➤➔→✓✗✔►]\s+(.+)$/);
  if (bulletMatch) {
    return { type: 'bullet', label: '-', rest: bulletMatch[1].trim() };
  }
  return null;
}

/**
 * Strips markdown bold markers, emoji, and bullet/list prefixes from a line.
 */
function stripFormatting(line: string): string {
  let cleaned = line
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}]/gu, '')
    .replace(/^[\s\-*•◦▪▸➤➔→✓✗✔►#]+/, '')
    .trim();
  // Strip compound sub-indexes like "a.i ", "1.1 ", "a.1 "
  cleaned = cleaned.replace(/^[A-Za-z0-9]+[.)][A-Za-z0-9]+[.)\s]*/, '').trim();
  return cleaned;
}

/**
 * Parses a Markdown pipe-table (produced by readDocxContent) into rows of string[].
 * Handles merged cells (empty cells) by inheriting the previous row's value.
 */
function parseMarkdownTable(lines: string[]): string[][] {
  const rows: string[][] = [];
  for (const line of lines) {
    const t = line.trim();
    if (!t.startsWith('|')) continue;
    if (/^\|[\s|:-]+\|$/.test(t)) continue; // separator row
    const cells = t
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map(c => c.trim());
    rows.push(cells);
  }
  // Fill forward for merged cells (empty cell = inherit from previous row)
  for (let r = 1; r < rows.length; r++) {
    for (let c = 0; c < rows[r].length; c++) {
      if (rows[r][c] === '' && rows[r - 1] && rows[r - 1][c] !== undefined) {
        rows[r][c] = rows[r - 1][c];
      }
    }
  }
  return rows;
}

export function localExtractRequirements(content: string): Requirement[] {
  // ── Pre-check: 4-column mammoth cell-by-cell table extraction ──────────────────
  const cleanLines = content.split(/\r?\n/).map(l => l.trim()).filter(Boolean);

  // Flexible 4-column table header detection (case-insensitive, partial match)
  const fourColHeaders = [
    ['section', 'goal', 'requirements', 'expected'],
    ['id', 'title', 'requirement', 'expected'],
    ['req', 'title', 'description', 'result'],
    ['feature', 'goal', 'requirement', 'expected'],
  ];
  const isFourColFlatTable = cleanLines.length >= 4 && fourColHeaders.some(headers =>
    headers.every((h, i) => cleanLines[i] && cleanLines[i].toLowerCase().includes(h))
  );

  if (isFourColFlatTable) {
    const tableRequirements: Omit<Requirement, 'fingerprint'>[] = [];
    let seqIndex = 1;
    for (let i = 4; i < cleanLines.length; i += 4) {
      const title = cleanLines[i] || '';
      const goal = cleanLines[i + 1] || '';
      const description = cleanLines[i + 2] || '';
      const expectedResult = cleanLines[i + 3] || '';

      if (!title && !description) continue;

      const paddedIndex = seqIndex.toString().padStart(2, '0');
      tableRequirements.push({
        id: `FR-${paddedIndex}`,
        title,
        description: description,
        goal,
        expectedResult,
        rawText: `Section: ${title}\nGoal: ${goal}\nRequirements: ${description}\nExpected Result: ${expectedResult}`
      });
      seqIndex++;
    }
    return tableRequirements.map(req => ({
      ...req,
      fingerprint: generateRequirementFingerprint(req)
    }));
  }

  const reqIdRegex = /\b(FR-\d+|R-?\d+|TC-\d+|TS-\d+)\b/i;
  const lines = content.split(/\r?\n/);
  const requirements: Omit<Requirement, 'fingerprint'>[] = [];

  // ── Helpers ────────────────────────────────────────────────────────────────
  const isTableRow = (line: string): boolean => {
    const t = line.trim();
    return t.startsWith('|') && t.endsWith('|') && t.length > 2;
  };
  const isTableSeparator = (line: string): boolean =>
    /^\|?\s*[-:|]+(\|\s*[-:|]+\s*)+\|?$/.test(line.trim());

  // Section headings (ALL-CAPS short lines, or lines ending with ':') are NOT requirements.
  const isSectionHeading = (title: string): boolean => {
    const t = title.trim();
    if (t === t.toUpperCase() && t.length > 2 && t.length < 80 && /[A-Z]/.test(t)) return true;
    if (t.endsWith(':') && t.split(' ').length < 8) return true;
    return false;
  };

  const stripBullet = (line: string): string => stripFormatting(line);

  // ── PATH: Markdown pipe-table (from readDocxContent or plain markdown) ─────
  const tableLines = lines.filter(l => isTableRow(l) || isTableSeparator(l));
  const hasMarkdownTable = tableLines.length >= 2;

  if (hasMarkdownTable) {
    // Group contiguous table lines into blocks
    const tableBlocks: string[][] = [];
    let currentTableBlock: string[] = [];
    for (const rawLine of lines) {
      if (isTableRow(rawLine) || isTableSeparator(rawLine)) {
        currentTableBlock.push(rawLine);
      } else {
        if (currentTableBlock.length >= 2) tableBlocks.push([...currentTableBlock]);
        currentTableBlock = [];
      }
    }
    if (currentTableBlock.length >= 2) tableBlocks.push(currentTableBlock);

    for (const block of tableBlocks) {
      const rows = parseMarkdownTable(block);
      if (rows.length < 1) continue;

      const header = rows[0].map(h => h.toLowerCase().trim());
      const dataRows = rows.slice(1);

      // Map column indices for known column names
      const colIdx = (names: string[]): number =>
        header.findIndex(h => names.some(n => h.includes(n)));

      const idCol = colIdx(['id', 'req', 'fr-', 'requirement id', 'no.', '#']);
      const titleCol = colIdx(['title', 'section', 'feature', 'name', 'heading']);
      const goalCol = colIdx(['goal', 'purpose', 'objective']);
      const descCol = colIdx(['requirement', 'description', 'details', 'statement', 'spec']);
      const expectedCol = colIdx(['expected', 'result', 'outcome', 'acceptance']);

      let seqIndex = requirements.length + 1;
      for (const row of dataRows) {
        const getCell = (idx: number) => (idx >= 0 && row[idx] ? row[idx].trim() : '');

        let id = idCol >= 0 ? getCell(idCol) : '';
        const title = titleCol >= 0 ? getCell(titleCol) : '';
        const goal = goalCol >= 0 ? getCell(goalCol) : '';
        const description = descCol >= 0 ? getCell(descCol) : '';
        const expectedResult = expectedCol >= 0 ? getCell(expectedCol) : '';

        // If no dedicated id column, derive from cell that matches ID pattern
        if (!id) {
          const idMatch = row.join(' ').match(reqIdRegex);
          id = idMatch ? idMatch[1].toUpperCase() : `FR-${seqIndex.toString().padStart(2, '0')}`;
        } else {
          const idMatch = id.match(reqIdRegex);
          id = idMatch ? idMatch[1].toUpperCase() : `FR-${seqIndex.toString().padStart(2, '0')}`;
        }

        // Use the most content-rich column as description fallback
        const bestDesc = description || row
          .filter((_, i) => i !== idCol && i !== titleCol && i !== goalCol && i !== expectedCol)
          .join(' ').trim();

        const rowTitle = title || bestDesc.split(/[.!?]/)[0].slice(0, 80).trim();
        if (!rowTitle && !bestDesc) continue;

        requirements.push({
          id,
          title: rowTitle,
          description: bestDesc,
          goal: goal || undefined,
          expectedResult: expectedResult || undefined,
          rawText: row.join(' | ')
        });
        seqIndex++;
      }
    }
  }

  // ── Format detection for non-table content ─────────────────────────────────
  const nonTableLines = lines.filter(l => !isTableRow(l) && !isTableSeparator(l));

  let hasExplicitIds = false;
  let hasBulletList = false;
  let hasRomanHeadings = false;
  let hasArabicDotHead = false;
  let hasArabicParenHead = false;
  let hasAlphaUpperHead = false;
  let hasAlphaLowerHead = false;

  for (const line of nonTableLines) {
    const t = line.trim();
    if (!t) continue;
    if (reqIdRegex.test(t)) hasExplicitIds = true;
    const prefix = detectListPrefix(t);
    if (prefix) {
      if (prefix.type === 'roman') hasRomanHeadings = true;
      else if (prefix.type === 'numeric') {
        if (/^\d+\.\s/.test(t)) hasArabicDotHead = true;
        else if (/^\d+\)\s/.test(t)) hasArabicParenHead = true;
        else hasArabicDotHead = true;
      } else if (prefix.type === 'alpha') {
        if (/^[A-Z]\.\s/.test(t)) hasAlphaUpperHead = true;
        else hasAlphaLowerHead = true;
      } else if (prefix.type === 'bullet') {
        hasBulletList = true;
      }
    }
  }

  const hasStructuredHeadings = hasRomanHeadings || hasArabicDotHead ||
    hasArabicParenHead || hasAlphaUpperHead || hasAlphaLowerHead;

  // ── PATH A: Explicit IDs (FR-01, R1, R-01, TC-01…) ────────────────────────
  if (hasExplicitIds) {
    const processedContent = content.replace(
      /(?<=.)\s*(?=[\[\(\{\s]*\b(R-?\d+|FR-\d+|TS-\d+|TC-[A-Z0-9-]+)\b)/gi, '\n'
    );
    const processedLines = processedContent.split(/\r?\n/);

    const cleanLine = (line: string): string =>
      line
        .replace(/\*\*/g, '')
        .replace(/^[\s\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\-*\u2022#]+/u, '')
        .trim();

    const reqHeaderRegex =
      /^\s*[\-*•#\s]*[\[\(\{\s]*\b(R-?\d+|FR-\d+|TS-\d+|TC-[A-Z0-9-]+)\b\s*[\]\)\}\s]*[:\-]?\s*(.+)$/i;
    let currentReq: Omit<Requirement, 'fingerprint'> | null = null;

    for (const rawLine of processedLines) {
      if (isTableSeparator(rawLine) || isTableRow(rawLine)) continue;

      const cleaned = cleanLine(rawLine);
      const match = cleaned.match(reqHeaderRegex);

      if (match) {
        if (currentReq) requirements.push(currentReq);
        currentReq = {
          id: match[1].toUpperCase(),
          title: match[2].trim(),
          description: '',
          rawText: rawLine
        };
      } else if (currentReq) {
        if (cleaned) {
          const expMatch = cleaned.match(/^expected\s+result[s]?[:\-]?\s*(.*)/i);
          const goalMatch = cleaned.match(/^goal[s]?[:\-]?\s*(.*)/i);
          if (expMatch) {
            currentReq.expectedResult = (currentReq.expectedResult ? currentReq.expectedResult + '\n' : '') + expMatch[1].trim();
          } else if (goalMatch) {
            currentReq.goal = (currentReq.goal ? currentReq.goal + '\n' : '') + goalMatch[1].trim();
          } else {
            currentReq.description += (currentReq.description ? '\n' : '') + cleaned;
          }
        }
        currentReq.rawText += '\n' + rawLine;
      }
    }
    if (currentReq) requirements.push(currentReq);

    // ── PATH B: Bullet/list-only format (all items are bullets without IDs) ────
  } else if (hasBulletList && !hasStructuredHeadings) {
    // Each bullet item is treated as its own requirement
    let currentReq: Omit<Requirement, 'fingerprint'> | null = null;
    let seqIndex = requirements.length + 1;

    for (const rawLine of nonTableLines) {
      const trimmed = rawLine.trim();
      if (!trimmed) {
        if (currentReq) { requirements.push(currentReq); currentReq = null; }
        continue;
      }

      const prefix = detectListPrefix(trimmed);
      const isIndented = rawLine.length > 0 && /^\s{2,}/.test(rawLine);

      if (prefix && prefix.type === 'bullet' && !isIndented) {
        if (currentReq) requirements.push(currentReq);
        const paddedIndex = seqIndex.toString().padStart(2, '0');
        seqIndex++;
        currentReq = {
          id: `FR-${paddedIndex}`,
          title: prefix.rest,
          description: '',
          rawText: rawLine
        };
      } else if (currentReq) {
        // Sub-bullets become description
        const cleaned = stripBullet(trimmed);
        if (cleaned) {
          const expMatch = cleaned.match(/^expected\s+result[s]?[:\-]?\s*(.*)/i);
          if (expMatch) {
            currentReq.expectedResult = (currentReq.expectedResult ? currentReq.expectedResult + '\n' : '') + expMatch[1].trim();
          } else {
            currentReq.description += (currentReq.description ? '\n' : '') + cleaned;
          }
        }
        currentReq.rawText += '\n' + rawLine;
      }
    }
    if (currentReq) requirements.push(currentReq);

    // ── PATH C: Structured numbering (roman / arabic / alpha) ──────────────────
  } else if (hasStructuredHeadings) {
    let currentReq: Omit<Requirement, 'fingerprint'> | null = null;
    let seqIndex = requirements.length + 1;

    const isTopLevelPrefix = (trimmed: string): { label: string; rest: string } | null => {
      const p = detectListPrefix(trimmed);
      if (!p) return null;
      // Treat as top-level if not indented and type matches detected heading types
      if (p.type === 'roman' && hasRomanHeadings) return p;
      if (p.type === 'numeric' && (hasArabicDotHead || hasArabicParenHead)) return p;
      if (p.type === 'alpha' && (hasAlphaUpperHead || hasAlphaLowerHead)) return p;
      return null;
    };

    for (const rawLine of nonTableLines) {
      const trimmed = rawLine.trim();
      const isIndented = rawLine.length > 0 && /^\s{2,}/.test(rawLine);

      if (!trimmed) continue;

      const topLevel = !isIndented ? isTopLevelPrefix(trimmed) : null;

      if (topLevel) {
        const title = topLevel.rest;
        if (isSectionHeading(title)) continue;
        if (currentReq) requirements.push(currentReq);
        const paddedIndex = seqIndex.toString().padStart(2, '0');
        seqIndex++;
        currentReq = { id: `FR-${paddedIndex}`, title, description: '', rawText: rawLine };
      } else if (currentReq) {
        const prefix = detectListPrefix(trimmed);
        const cleaned = prefix ? prefix.rest : stripBullet(trimmed);
        if (cleaned) {
          const expMatch = cleaned.match(/^expected\s+result[s]?[:\-]?\s*(.*)/i);
          const goalMatch = cleaned.match(/^goal[s]?[:\-]?\s*(.*)/i);
          if (expMatch) {
            currentReq.expectedResult = (currentReq.expectedResult ? currentReq.expectedResult + '\n' : '') + expMatch[1].trim();
          } else if (goalMatch) {
            currentReq.goal = (currentReq.goal ? currentReq.goal + '\n' : '') + goalMatch[1].trim();
          } else {
            currentReq.description += (currentReq.description ? '\n' : '') + cleaned;
          }
        }
        currentReq.rawText += '\n' + rawLine;
      }
    }
    if (currentReq) requirements.push(currentReq);

    // ── PATH D: Pure prose / paragraph mode ────────────────────────────────────
  } else {
    let seqIndex = requirements.length + 1;

    const paragraphBlocks: string[][] = [];
    let currentBlock: string[] = [];
    for (const rawLine of nonTableLines) {
      if (rawLine.trim() === '') {
        if (currentBlock.length > 0) { paragraphBlocks.push(currentBlock); currentBlock = []; }
      } else {
        currentBlock.push(rawLine);
      }
    }
    if (currentBlock.length > 0) paragraphBlocks.push(currentBlock);

    for (const block of paragraphBlocks) {
      const firstLine = block[0]?.trim() ?? '';
      const isMarkdownHead = firstLine.startsWith('#');
      const isShortTitleLine =
        firstLine.length > 3 &&
        /^[A-Z]/.test(firstLine) &&
        firstLine.length < 80 &&
        !/^[\-*•]/.test(firstLine) &&
        ((firstLine === firstLine.toUpperCase() && /[A-Z]/.test(firstLine)) || block.length > 1);

      if (isMarkdownHead || isShortTitleLine) {
        const cleanTitle = firstLine.replace(/^[#\*\s]+|[#\*\s]+$/g, '');
        if (isSectionHeading(cleanTitle)) continue;

        const bodyLines = block.slice(1);
        const description = bodyLines
          .map((l: string) => stripBullet(l.trim()))
          .filter((l: string) => l.length > 0)
          .join('\n');
        requirements.push({
          id: `TEMP-${seqIndex++}`,
          title: cleanTitle,
          description,
          rawText: block.join('\n')
        });
      } else {
        const nonEmptyLines = block
          .map((l: string) => stripBullet(l.trim()))
          .filter((l: string) => l.length > 0);
        if (nonEmptyLines.length > 0) {
          const id = `TEMP-${seqIndex++}`;
          requirements.push({
            id,
            title: 'Requirement ' + id,
            description: nonEmptyLines.join('\n'),
            rawText: block.join('\n')
          });
        }
      }
    }
  }

  return requirements.map(req => ({
    ...req,
    fingerprint: generateRequirementFingerprint(req)
  }));
}




async function revalidateRequirementsWithAI(
  apiKey: string,
  content: string,
  localReqs: Requirement[]
): Promise<Requirement[]> {
  const genAI = new GoogleGenerativeAI(apiKey);

  const systemInstruction = [
    "You are an expert QA requirements diff engine.",
    "Your task: given a raw Requirement Document and a locally-parsed requirements list, produce the FINAL CORRECTED list that exactly matches the raw document — nothing more, nothing less.",
    "",
    "═══ YOUR RESPONSIBILITIES ═══",
    "1. ADD: If the raw document contains a requirement, section, or bullet that the local parser missed, add it.",
    "2. UPDATE: If a requirement's title, description, or any word was changed/added/removed in the raw document compared to the local list, update it to exactly match the raw document text.",
    "3. REMOVE: If a requirement or any of its lines are no longer present in the raw document, remove them from the output. NEVER retain content that is absent from the raw document.",
    "4. PRESERVE IDs: Keep the same IDs as the local list if the requirement still exists. Only renumber if the sequence changed due to additions or removals.",
    "   CRITICAL: ALWAYS output IDs in strict 'FR-XX' format (e.g., FR-01, FR-02, FR-03). If the document uses Roman numerals (I, II, III), plain numbers (1, 2, 3), or letters (A, B, C) as section numbers, convert them to FR-01, FR-02, FR-03 etc. NEVER output 'I', 'II', 'III', 'A', 'B', '1', '2' as IDs — always use FR-XX format.",
    "5. ORDER: Output requirements in the same order they appear in the raw document.",
    "",
    "═══ CRITICAL ANTI-HALLUCINATION RULES ═══",
    "- NEVER invent requirements that are not present in the raw document.",
    "- NEVER restore deleted requirements from the local list that are absent from the raw document.",
    "- NEVER recreate text that has been removed from the raw document.",
    "- NEVER assume a requirement still exists because it existed before. Only the raw document is truth.",
    "- If a bullet point, sentence, or line was removed from a requirement in the raw document, it must NOT appear in the description of the output.",
    "- Do not merge multiple requirements into one, and do not split one into many.",
    "",
    "═══ FORMAT HANDLING ═══",
    "Handle ALL requirement document formats without exception:",
    "  - Plain text paragraphs (no formatting)",
    "  - Bullet lists (-, *, •, ◦, ▪, ▸, ➤, ✓, ✗, ► etc.)",
    "  - Numbered lists (1. 2. 3. or 1) 2) 3))",
    "  - Alphabetic lists (A. B. C. or a. b. c. or A) B) C))",
    "  - Roman numeral lists (I. II. III. IV. or i. ii. iii.)",
    "  - Nested lists (any combination of the above)",
    "  - Mixed layouts (different formats in the same document)",
    "  - Tables with header rows and data rows",
    "  - Tables with merged cells (a cell repeated for multiple rows)",
    "  - 4-column tables (Section | Goal | Requirements | Expected Result)",
    "  - 2-column tables (ID | Requirement or Title | Description)",
    "  - N-column tables with any column header names",
    "  - Documents with explicit IDs (FR-01, R1, R-01, TC-01)",
    "  - Documents with no explicit IDs (generate sequential FR-XX IDs)",
    "  - Markdown documents (# headings, ## subheadings, --- separators)",
    "  - DOCX documents rendered as HTML-converted markdown",
    "For 4-column tables: the 'title' field MUST contain the Section. The 'description' field MUST contain ONLY the verbatim text of the 'Requirements' column. The 'goal' field MUST contain the Goal column value. The 'expectedResult' field MUST contain the Expected Result column value. NEVER merge these sections together; extract them independently into their respective fields.",
    "For tables with merged cells: a blank cell in a data row inherits the value from the same column in the previous row.",
    "For other document formats: map heading/title to 'title', requirement body to 'description', goal to 'goal' (if specified, else empty), and expected result to 'expectedResult' (if specified, else empty).",
    "For bullet-only documents: each top-level bullet item is a separate requirement. Nested/indented sub-bullets become the description of the parent requirement.",
    "For numbered/roman/alpha lists: each numbered item is a separate requirement. Indented sub-items are the description.",
    "",
    "═══ DETERMINISM ═══",
    "Given identical inputs, always produce identical outputs.",
    "Do not reorder, renumber, or restructure requirements unless the raw document itself changed."
  ].join("\n");

  const prompt = [
    `═══ RAW REQUIREMENT DOCUMENT (source of truth) ═══`,
    content,
    ``,
    `═══ LOCAL PARSER OUTPUT (may be incomplete or stale) ═══`,
    JSON.stringify(localReqs, null, 2),
    ``,
    `═══ TASK ═══`,
    `Produce the final corrected requirements list that EXACTLY matches the raw document.`,
    `Every requirement present in the raw document must appear in the output.`,
    `Every requirement NOT present in the raw document must be ABSENT from the output.`,
    `If a line was removed from a requirement description, it must NOT appear in the output.`,
    `Do NOT invent, restore, or recreate any content not found verbatim in the raw document.`
  ].join('\n');

  const schema: any = {
    type: "object",
    properties: {
      requirements: {
        type: "array",
        description: "Final corrected list of requirements matching the raw document exactly, in document order.",
        items: {
          type: "object",
          properties: {
            id: { type: "string", description: "Requirement ID (from document or generated FR-XX)." },
            title: { type: "string", description: "Short heading phrase or Section name for this requirement." },
            description: { type: "string", description: "Verbatim text of the 'Requirements' column or the body of the requirement statement." },
            goal: { type: "string", description: "Goal/Purpose of the requirement (if available, otherwise empty string)." },
            expectedResult: { type: "string", description: "Expected result or assertion criteria (if available, otherwise empty string)." },
            rawText: { type: "string", description: "Exact raw text block from the document for this requirement." }
          },
          required: ["id", "title", "description", "goal", "expectedResult", "rawText"]
        }
      }
    },
    required: ["requirements"]
  };

  const modelName = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction,
    generationConfig: {
      temperature: 0.0,
      responseMimeType: "application/json",
      responseSchema: schema
    }
  });

  try {
    const responseText = await callWithRetry(async () => {
      const result = await model.generateContent(prompt);
      return result.response.text();
    }, 2, 2000);

    const parsed = JSON.parse(responseText);
    if (parsed && Array.isArray(parsed.requirements) && parsed.requirements.length > 0) {
      return parsed.requirements;
    }
    throw new Error("Invalid or empty re-validated requirements list returned by Gemini.");
  } catch (error: any) {
    console.error(`  ❌ Model ${modelName} failed to re-validate requirements: ${error.message || error}`);
    throw error;
  }
}

export async function extractRequirements(content: string): Promise<Requirement[]> {
  const localRequirements = localExtractRequirements(content).map(req => ({
    ...req,
    id: normalizeRequirementId(req.id)
  }));

  const cleanLines = content.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const isFourColTable = cleanLines.length >= 4 &&
    cleanLines[0].toLowerCase() === 'section' &&
    cleanLines[1].toLowerCase() === 'goal' &&
    cleanLines[2].toLowerCase() === 'requirements' &&
    cleanLines[3].toLowerCase() === 'expected result';

  const geminiKey = process.env.GEMINI_API_KEY;

  if (geminiKey && geminiKey.trim() !== '') {
    try {
      const isVerbose = process.env.QA_SYNC_VERBOSE === 'true';
      if (isVerbose) {
        console.log("  → Validating and refining requirements with Gemini AI...");
      }
      const validatedReqs = await revalidateRequirementsWithAI(geminiKey, content, localRequirements);
      if (validatedReqs && validatedReqs.length > 0) {
        const uniqueIds = new Set<string>();
        const validRequirements: Requirement[] = [];

        // Pattern that matches recognized requirement ID formats
        const validIdPattern = /^(FR|R|TC|TS)-?\d+$/i;

        // Build a local ID lookup for fallback: position → FR-XX from local parser
        const localIdByPos = new Map<number, string>();
        localRequirements.forEach((r, idx) => { localIdByPos.set(idx, r.id); });

        let aiSeqIndex = 1;
        for (const req of validatedReqs) {
          if (!req.id || !req.title) continue;

          let normId = normalizeRequirementId(req.id);

          // If the AI returned a non-standard ID (Roman numeral, plain number, single letter),
          // convert it to sequential FR-XX using the local parser's list as source of truth
          if (!validIdPattern.test(normId)) {
            // Try to find matching local requirement by title similarity
            const matchingLocal = localRequirements.find(lr =>
              lr.title.toLowerCase().trim() === req.title.toLowerCase().trim()
            );
            if (matchingLocal) {
              normId = matchingLocal.id;
            } else {
              // Fallback: use sequential FR-XX
              normId = `FR-${aiSeqIndex.toString().padStart(2, '0')}`;
            }
          }

          aiSeqIndex++;

          if (!uniqueIds.has(normId)) {
            uniqueIds.add(normId);
            const reqTemp = {
              id: normId,
              title: req.title.trim(),
              description: req.description.trim(),
              goal: req.goal ? req.goal.trim() : undefined,
              expectedResult: req.expectedResult ? req.expectedResult.trim() : undefined,
              rawText: req.rawText
            };
            validRequirements.push({
              ...reqTemp,
              fingerprint: generateRequirementFingerprint(reqTemp)
            });
          }
        }
        if (validRequirements.length > 0) {
          if (isVerbose) {
            console.log(`  ✓ Successfully validated and extracted ${validRequirements.length} requirements.`);
          }
          return validRequirements;
        }
      }
    } catch (error: any) {
      console.warn(`  ⚠️ Gemini AI requirement re-validation failed (${error.message || error}). Falling back to local parser.`);
    }
  }

  if (localRequirements.length > 0) {
    return localRequirements;
  }

  throw new Error("No requirements found matching pattern, and local parser returned empty results.");
}


/**
 * Structure-based executable atomic extractor.
 *
 * Extracts ONLY the executable atomic statements from a requirement's description.
 * Uses the Requirement Document structure — NOT verb heuristics — to decide what is executable.
 *
 * INCLUDED as executable (requirement body content):
 *   - Every bullet point (•, -, *) in the requirement body
 *   - Every numbered sub-item (1., a., i.) in the requirement body
 *   - Every plain sentence in the requirement body
 *
 * EXCLUDED from coverage analysis (non-executable structure):
 *   - The requirement Title line itself
 *   - Lines starting with: Goal:, Goals:, Purpose:, Overview:, Note:, Notes:, Context:
 *   - Expected Result: / Expected Results: header lines (but CONTENT below IS executable)
 *   - ALL-CAPS section headings (e.g. NAVIGATION REQUIREMENTS)
 *   - Lines shorter than 10 characters
 *   - Lines that are purely numeric
 */
export function extractExecutableAtomics(req: Requirement): string[] {
  if (!req.description || !req.description.trim()) return [];

  const lines = req.description.split(/\r?\n/);
  const atomics: string[] = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.length < 10) continue;

    // Skip purely numeric lines (page numbers, reference numbers)
    if (/^\d+$/.test(line)) continue;

    // Skip non-executable structural headers
    const isNonExecHeader =
      /^(goal|goals|purpose|overview|note|notes|context)\s*:/i.test(line) ||
      /^expected\s+results?\s*:/i.test(line);

    if (isNonExecHeader) continue;

    // Skip ALL-CAPS section headings (3+ uppercase letters, no lowercase, short enough to be a heading)
    if (line === line.toUpperCase() && /[A-Z]{3,}/.test(line) && line.length < 80 && !/[.?!]$/.test(line)) continue;

    // Split line into individual sentences to prevent merging multiple requirement statements
    const sentences = line.split(/(?<=[.?!])\s+/);
    for (const sentence of sentences) {
      const cleaned = sentence
        .replace(/^[\s\-\*•\d\.\)\(]+/, '')
        .trim();

      if (cleaned.length >= 10) {
        atomics.push(cleaned);
      }
    }
  }

  return atomics;
}

/**
 * Finds all Playwright test files recursively in the project.
 */
export function findTestScripts(projectRoot: string): string[] {
  const skipDirs = ['node_modules', 'dist', 'build', '.git', '.github', 'test-results', 'playwright-report', 'qa-sync-tool', 'scratch'];
  return getFilesRecursively(
    projectRoot,
    (name) => name.endsWith('.spec.ts'),
    skipDirs
  );
}

/**
 * Scans test spec files and extracts individual test cases, titles, and mappings.
 * Uses a brace-matching parser to correctly identify multi-line test blocks.
 */
export function scanTestFiles(filePaths: string[]): ParsedTest[] {
  const parsedTests: ParsedTest[] = [];

  for (const filePath of filePaths) {
    const content = fs.readFileSync(filePath, 'utf-8');
    const testsInFile = parseSpecFileContent(content, filePath);
    parsedTests.push(...testsInFile);
  }

  return parsedTests;
}

export function parseTestBodyToBlocks(bodyContent: string): BlockStructure[] {
  // Fix Issue 9: R-?\d+ covers both R01 and R-01 (hyphenated) forms
  const reqCommentRegex = /(\/\/\s*[^\w\s]*\s*(?:R-?\d+|FR-\d+|TS-\d+|TC\-[A-Z0-9\-]+)\b.*|\/\*\s*[^\w\s]*\s*(?:R-?\d+|FR-\d+|TS-\d+|TC\-[A-Z0-9\-]+)\b[\s\S]*?\*\/)/gi;
  const parts = bodyContent.split(reqCommentRegex);
  const blocks: BlockStructure[] = [];

  if (parts[0] !== undefined && parts[0].trim() !== '') {
    blocks.push({
      type: 'generic',
      code: parts[0]
    });
  }

  for (let i = 1; i < parts.length; i += 2) {
    const comment = parts[i];
    const code = parts[i + 1] || '';

    const ids: string[] = [];
    // Fix Issue 9: R-?\d+ covers R-01 (hyphenated) as well as R01
    const regex = /\b(R-?\d+|FR-\d+|TS-\d+|TC-[A-Z0-9-]+)\b/gi;
    let m;
    while ((m = regex.exec(comment)) !== null) {
      ids.push(m[1].toUpperCase());
    }

    blocks.push({
      type: ids.length > 0 ? 'requirement' : 'generic',
      requirementId: ids[0],
      requirementIds: ids,
      code: comment + code
    });
  }

  return blocks;
}

export function parseTestFileStructure(content: string, filePath: string): FileSegment[] {
  const segments: FileSegment[] = [];
  const testStartRegex = /\b(test|test\.(?:only|skip))\s*\(/g;

  let match;
  let lastIdx = 0;

  interface TestCaseRange {
    matchIndex: number;
    blockEndIdx: number;
    title: string;
    bodyStartIdx: number;
    closingBraceIdx: number;
  }

  const ranges: TestCaseRange[] = [];

  while ((match = testStartRegex.exec(content)) !== null) {
    const matchIndex = match.index;
    const { title, titleEndIdx } = parseTestTitle(content, matchIndex + match[0].length);
    if (title === null) continue;

    let bodyStartIdx = -1;
    const arrowIdx = content.indexOf('=>', titleEndIdx);
    if (arrowIdx !== -1) {
      bodyStartIdx = content.indexOf('{', arrowIdx);
    } else {
      bodyStartIdx = content.indexOf('{', titleEndIdx);
    }
    if (bodyStartIdx === -1) continue;

    const blockEndIdx = findMatchingClosingBrace(content, bodyStartIdx);
    if (blockEndIdx === -1) continue;

    let closingBraceIdx = -1;
    for (let i = blockEndIdx - 1; i >= bodyStartIdx; i--) {
      if (content[i] === '}') {
        closingBraceIdx = i;
        break;
      }
    }
    if (closingBraceIdx === -1) continue;

    ranges.push({
      matchIndex,
      blockEndIdx,
      title,
      bodyStartIdx,
      closingBraceIdx
    });

    testStartRegex.lastIndex = blockEndIdx;
  }

  for (const range of ranges) {
    if (range.matchIndex > lastIdx) {
      segments.push({
        type: 'text',
        content: content.substring(lastIdx, range.matchIndex)
      });
    }

    const testFullText = content.substring(range.matchIndex, range.blockEndIdx);
    const header = content.substring(range.matchIndex, range.bodyStartIdx + 1);
    const footer = content.substring(range.closingBraceIdx, range.blockEndIdx);
    const bodyContent = content.substring(range.bodyStartIdx + 1, range.closingBraceIdx);
    const blocks = parseTestBodyToBlocks(bodyContent);

    const startLine = content.substring(0, range.matchIndex).split('\n').length;
    const endLine = content.substring(0, range.blockEndIdx).split('\n').length;

    segments.push({
      type: 'testCase',
      structure: {
        title: range.title,
        fullText: testFullText,
        header,
        blocks,
        footer,
        startLine,
        endLine,
        matchIndex: range.matchIndex
      }
    });

    lastIdx = range.blockEndIdx;
  }

  if (lastIdx < content.length) {
    segments.push({
      type: 'text',
      content: content.substring(lastIdx)
    });
  }

  return segments;
}

export function reconstructTestFile(segments: FileSegment[]): string {
  // Detect dominant indentation style from existing text segments
  let indentUnit = '  '; // default 2 spaces
  for (const seg of segments) {
    if (seg.type === 'text') {
      const indentMatch = (seg as { type: 'text'; content: string }).content.match(/^([ \t]{2,})\S/m);
      if (indentMatch) {
        const raw = indentMatch[1];
        indentUnit = raw.startsWith('\t') ? '\t' : ' '.repeat(Math.min(raw.length, 4));
        break;
      }
    }
  }

  let fileContent = '';
  for (const seg of segments) {
    if (seg.type === 'text') {
      fileContent += seg.content;
    } else if (seg.type === 'testCase') {
      const tc = seg.structure;
      let body = '';
      for (const b of tc.blocks) {
        const blockCode = b.code.trim();
        if (blockCode === '') continue;

        if (body !== '') {
          body = body.trimEnd() + '\n' + indentUnit + blockCode;
        } else {
          body = '\n' + indentUnit + blockCode;
        }
      }
      if (body !== '') {
        body = body.trimEnd() + '\n';
      }
      fileContent += tc.header + body + tc.footer;
    }
  }
  return fileContent;
}

/**
 * Parses spec file content to extract test cases with precise locations and requirement mapping.
 */
export function parseSpecFileContent(content: string, filePath: string): ParsedTest[] {
  const parsedTests: ParsedTest[] = [];
  const segments = parseTestFileStructure(content, filePath);

  for (const seg of segments) {
    if (seg.type === 'testCase') {
      const tc = seg.structure;
      const testStartIdx = tc.matchIndex !== undefined ? tc.matchIndex : content.indexOf(tc.header);
      const precedingComment = extractPrecedingComment(content, testStartIdx);

      let requirementId = extractRequirementId(tc.title) || extractRequirementId(precedingComment);
      const requirementIdsSet = new Set<string>();
      const reqBlocks: { id: string; code: string }[] = [];

      for (const block of tc.blocks) {
        if (block.type === 'requirement') {
          const ids = block.requirementIds || (block.requirementId ? [block.requirementId] : []);
          for (const rid of ids) {
            requirementIdsSet.add(rid);
            reqBlocks.push({ id: rid, code: block.code });
          }
        }
      }

      // Also parse multiple IDs from title and preceding comment!
      const titleAndPreCommentIds = extractAllRequirementIds(tc.title, precedingComment, '');
      for (const rid of titleAndPreCommentIds) {
        requirementIdsSet.add(rid);
      }

      // Extract requirement IDs from parent describe blocks!
      const parentDescribeIds = extractParentDescribeRequirements(content, testStartIdx);
      for (const rid of parentDescribeIds) {
        requirementIdsSet.add(rid);
      }

      const requirementIds = Array.from(requirementIdsSet);
      if (requirementIds.length > 0 && !requirementId) {
        requirementId = requirementIds[0];
      }

      // If there are no internal requirement comment blocks, but we have a requirementId,
      // map the entire body of the test case as the requirement block!
      if (reqBlocks.length === 0 && requirementId) {
        const genericCode = tc.blocks.map(b => b.code).join('');
        reqBlocks.push({ id: requirementId, code: genericCode });
        tc.blocks = [{
          type: 'requirement',
          requirementId,
          code: genericCode
        }];
      }

      const detectedUrls = detectHardcodedUrls(tc.fullText);
      const detectedLocators = detectHardcodedLocators(tc.fullText);
      const detectedSecrets = detectHardcodedSecrets(tc.fullText);

      parsedTests.push({
        title: tc.title,
        requirementId,
        requirementIds,
        fullText: tc.fullText,
        startLine: tc.startLine,
        endLine: tc.endLine,
        filePath,
        hasHardcodedUrl: detectedUrls.length > 0,
        hasHardcodedLocator: detectedLocators.length > 0,
        hasHardcodedSecret: detectedSecrets.length > 0,
        detectedUrls,
        detectedLocators,
        detectedSecrets,
        requirementBlocks: reqBlocks,
        fingerprint: generateTestFingerprint(tc.fullText)
      });
    }
  }

  return parsedTests;
}

/**
 * Normalizes requirement IDs by stripping leading zeros in the numeric suffix,
 * e.g., "FR-01" -> "FR-1", "R02" -> "R2", "TS-09" -> "TS-9".
 */
export function normalizeRequirementId(id: string): string {
  return id.toUpperCase().replace(/([A-Z]+-?)0*([1-9]\d*)/g, '$1$2');
}

/**
 * Helper to extract requirement ID (e.g., "R1", "R-01", "R12") from a string.
 * Fix Issue 7: R-?\d+ now covers both R01 and hyphenated R-01 forms.
 */
function extractRequirementId(text: string): string | null {
  const match = text.match(/\b(R-?\d+|FR-\d+|TS-\d+|TC\-[A-Z0-9\-]+)\b/i);
  return match ? match[1].toUpperCase() : null;
}

/**
 * Extracts comments directly above the test definition index.
 */
function extractPrecedingComment(content: string, testStartIdx: number): string {
  const beforeText = content.substring(0, testStartIdx);
  const lines = beforeText.split(/\r?\n/);
  let comments = '';

  // Read backwards from the last line
  for (let i = lines.length - 2; i >= 0; i--) {
    const line = lines[i].trim();
    if (line.startsWith('//')) {
      comments = line.replace(/^\/\/\s*/, '') + '\n' + comments;
    } else if (line.endsWith('*/')) {
      // Basic block comment search backwards
      let blockComment = '';
      for (let j = i; j >= 0; j--) {
        const bline = lines[j].trim();
        blockComment = bline + '\n' + blockComment;
        if (bline.startsWith('/*')) {
          comments = blockComment.replace(/\/\*|\*\//g, '') + '\n' + comments;
          break;
        }
      }
      break;
    } else if (line !== '') {
      // Stopped seeing comments
      break;
    }
  }
  return comments.trim();
}

function extractParentDescribeRequirements(content: string, testStartIdx: number): string[] {
  const parentIds = new Set<string>();
  const describeRegex = /\b(test\.describe|describe)\s*\(/g;
  let match;

  while ((match = describeRegex.exec(content)) !== null) {
    const describeIdx = match.index;
    
    // Find title of this describe block
    const { title, titleEndIdx } = parseTestTitle(content, describeIdx + match[0].length);
    if (title === null) continue;

    let bodyStartIdx = -1;
    const arrowIdx = content.indexOf('=>', titleEndIdx);
    if (arrowIdx !== -1) {
      bodyStartIdx = content.indexOf('{', arrowIdx);
    } else {
      bodyStartIdx = content.indexOf('{', titleEndIdx);
    }
    if (bodyStartIdx === -1) continue;

    const blockEndIdx = findMatchingClosingBrace(content, bodyStartIdx);
    if (blockEndIdx === -1) continue;

    // Check if the test case start index falls inside this describe block
    if (testStartIdx > bodyStartIdx && testStartIdx < blockEndIdx) {
      // This is a parent describe block! Extract requirement IDs from it.
      const precedingComment = extractPrecedingComment(content, describeIdx);
      const descIds = extractAllRequirementIds(title, precedingComment, '');
      for (const id of descIds) {
        parentIds.add(id);
      }
    }
  }

  return Array.from(parentIds);
}

/**
 * Parses the string argument inside test('title', ...)
 */
function parseTestTitle(content: string, startIdx: number): { title: string | null; titleEndIdx: number } {
  // Find first quote character
  let quoteChar = '';
  let titleStart = -1;

  for (let i = startIdx; i < content.length; i++) {
    const char = content[i];
    if (char === "'" || char === '"' || char === '`') {
      quoteChar = char;
      titleStart = i + 1;
      break;
    }
    // If we hit non-whitespace or opening parens before quote, invalid
    if (char !== ' ' && char !== '\t' && char !== '\r' && char !== '\n') {
      // Not a string literal title (maybe a variable, skip)
      return { title: null, titleEndIdx: startIdx };
    }
  }

  if (titleStart === -1) return { title: null, titleEndIdx: startIdx };

  // Find closing quote that is followed by the function signature.
  // Fix Issue 8: also matches Playwright's tagged test form:
  //   test('name', { tag: '@smoke' }, async ({ page }) => {
  // The optional non-capturing group (?:\{[\s\S]*?\}\s*,\s*)? allows for an
  // options object between the title and the async arrow function.
  let titleEnd = -1;
  const remaining = content.substring(titleStart);
  const closingRegex = new RegExp(
    `^([\\s\\S]*?)${quoteChar}` +
    `(\\s*,\\s*(?:\\{[\\s\\S]*?\\}\\s*,\\s*)?(?:async\\s*)?\\((?:[\\s\\S]*?)\\)\\s*=>\\s*\\{)`
  );
  const match = remaining.match(closingRegex);
  if (match) {
    titleEnd = titleStart + match[1].length;
  }

  if (titleEnd === -1) return { title: null, titleEndIdx: startIdx };

  const rawTitle = content.substring(titleStart, titleEnd);
  const title = rawTitle
    .replace(/\\'/g, "'")
    .replace(/\\"/g, '"')
    .replace(/\\`/g, '`');

  return {
    title,
    titleEndIdx: titleEnd + 1,
  };
}

/**
 * Counts braces starting at opening brace index to find the closing end of the test statement:
 * e.g., test('...', async () => { ... }); -> matches the last ');' or ')'
 */
function findMatchingClosingBrace(content: string, bodyStartIdx: number): number {
  let braceCount = 1;
  let inString: string | null = null;
  // Fix Issue 12: track template literal nesting depth so ${...} braces inside
  // backtick strings are not counted against the outer brace counter.
  let templateDepth = 0;
  let isEscaped = false;

  for (let i = bodyStartIdx + 1; i < content.length; i++) {
    const char = content[i];

    if (isEscaped) {
      isEscaped = false;
      continue;
    }

    if (char === '\\') {
      isEscaped = true;
      continue;
    }

    // Skip single-line comments (only when not inside a string)
    if (!inString && char === '/' && content[i + 1] === '/') {
      const eol = content.indexOf('\n', i + 2);
      if (eol !== -1) {
        i = eol;
      } else {
        break; // end of file
      }
      continue;
    }

    // Skip block comments (only when not inside a string)
    if (!inString && char === '/' && content[i + 1] === '*') {
      const endBlock = content.indexOf('*/', i + 2);
      if (endBlock !== -1) {
        i = endBlock + 1;
      } else {
        break; // end of file
      }
      continue;
    }

    // Inside a non-template string
    if (inString && inString !== '`') {
      if (char === inString) {
        inString = null;
      }
      continue;
    }

    // Inside a template literal: handle ${...} interpolation braces
    if (inString === '`') {
      if (char === '`') {
        // closing backtick — but only at depth 0
        if (templateDepth === 0) {
          inString = null;
        }
      } else if (char === '$' && content[i + 1] === '{') {
        // entering ${...} interpolation — track nesting depth
        templateDepth++;
        i++; // skip the '{'
      } else if (char === '{' && templateDepth > 0) {
        templateDepth++;
      } else if (char === '}' && templateDepth > 0) {
        templateDepth--;
      }
      continue;
    }

    // Start a new string
    if (char === "'" || char === '"' || char === '`') {
      inString = char;
      templateDepth = 0;
      continue;
    }

    if (char === '{') {
      braceCount++;
    } else if (char === '}') {
      braceCount--;
      if (braceCount === 0) {
        // Find the matching closing parenthesis ')' of the test call
        let afterBraceIdx = i + 1;
        while (afterBraceIdx < content.length) {
          const nextChar = content[afterBraceIdx];
          if (nextChar === ')') {
            // Include optional semicolon
            if (content[afterBraceIdx + 1] === ';') {
              return afterBraceIdx + 2;
            }
            return afterBraceIdx + 1;
          }
          if (nextChar !== ' ' && nextChar !== '\t' && nextChar !== '\r' && nextChar !== '\n') {
            break;
          }
          afterBraceIdx++;
        }
        return i + 1;
      }
    }
  }

  return -1;
}

/**
 * Detects hardcoded URLs in page.goto() calls.
 */
export function detectHardcodedUrls(text: string): string[] {
  const urls: string[] = [];
  const regex = /page\.goto\(\s*(['"`])(.*?)\1\s*\)/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    const url = match[2];
    if (match[1] === '`' && url.includes('${')) {
      continue;
    }
    urls.push(url);
  }
  return urls;
}

/**
 * Detects hardcoded selectors in Playwright locator/action methods.
 */
export function detectHardcodedLocators(text: string): string[] {
  const locators: string[] = [];
  const methods = ['locator', 'fill', 'click', 'check', 'uncheck', 'selectOption', 'hover', 'focus', 'dblclick', 'dragTo'];
  for (const method of methods) {
    const regex = new RegExp(`page\\.${method}\\(\\s*(['"\`])(.*?)\\1`, 'g');
    let match;
    while ((match = regex.exec(text)) !== null) {
      const locator = match[2];
      if (match[1] === '`' && locator.includes('${')) {
        continue;
      }
      // Simple check to see if it is not referencing a variable (e.g. not beginning with locators. or config.)
      if (!locator.startsWith('locators.') && !locator.startsWith('config.')) {
        locators.push(locator);
      }
    }
  }
  return locators;
}

/**
 * Detects hardcoded credentials/secrets passed into page.fill() calls.
 */
export function detectHardcodedSecrets(text: string): { selector: string; value: string }[] {
  const secrets: { selector: string; value: string }[] = [];
  const regex = /page\.fill\(\s*(['"`])(.*?)\1\s*,\s*(['"`])(.*?)\3\s*\)/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    const selector = match[2];
    const value = match[4];

    if (match[3] === '`' && value.includes('${')) {
      continue;
    }

    const lowerSel = selector.toLowerCase();
    const isSensitiveSelector =
      lowerSel.includes('password') ||
      lowerSel.includes('pass') ||
      lowerSel.includes('secret') ||
      lowerSel.includes('token') ||
      lowerSel.includes('key') ||
      lowerSel.includes('email') ||
      lowerSel.includes('user') ||
      lowerSel.includes('login') ||
      lowerSel.includes('cred');

    const isSensitiveValue = !value.startsWith('process.env.') && value.length > 0;

    if (isSensitiveSelector && isSensitiveValue) {
      secrets.push({ selector, value });
    }
  }
  return secrets;
}

/**
 * Extracts all requirement IDs from test title, preceding comments, and internal code comments.
 */
export function extractAllRequirementIds(title: string, precedingComment: string, fullText: string): string[] {
  const ids = new Set<string>();

  function findIds(text: string) {
    const regex = /\b(R-?\d+|FR-\d+|TS-\d+|TC\-[A-Z0-9\-]+)\b/gi;
    let match;
    while ((match = regex.exec(text)) !== null) {
      ids.add(match[1].toUpperCase());
    }
  }

  findIds(title);
  findIds(precedingComment);

  const commentRegex = /\/\*[\s\S]*?\*\/|\/\/.*/g;
  let commentMatch;
  while ((commentMatch = commentRegex.exec(fullText)) !== null) {
    findIds(commentMatch[0]);
  }

  return Array.from(ids);
}

export interface ResolvedLocators {
  path: string;
  varName: string;
  fileName: string;
}

export function resolveLocatorsPathForFile(filePath: string, projectRoot: string): ResolvedLocators {
  const defaultPath = path.join(projectRoot, 'locators.ts');
  const defaultVarName = 'locators';

  // 1. Try to read the file itself
  if (fs.existsSync(filePath)) {
    const content = fs.readFileSync(filePath, 'utf-8');
    const match = findLocatorsImport(content);
    if (match) {
      const absPath = path.resolve(path.dirname(filePath), match.importPath + '.ts');
      const absPathJs = absPath.replace(/\.ts$/, '.js');
      if (fs.existsSync(absPath) || fs.existsSync(absPathJs)) {
        return { path: absPath, varName: match.varName, fileName: path.basename(absPath) };
      }
    }
  }

  // 2. Try to read other existing spec files recursively to see their import
  const specFiles = getFilesRecursively(
    projectRoot,
    (name) => name.endsWith('.spec.ts'),
    ['node_modules', 'dist', 'build', '.git', '.github', 'test-results', 'playwright-report', 'qa-sync-tool']
  );
  for (const fullPath of specFiles) {
    const content = fs.readFileSync(fullPath, 'utf-8');
    const match = findLocatorsImport(content);
    if (match) {
      const absPath = path.resolve(path.dirname(fullPath), match.importPath + '.ts');
      const absPathJs = absPath.replace(/\.ts$/, '.js');
      if (fs.existsSync(absPath) || fs.existsSync(absPathJs)) {
        return { path: absPath, varName: match.varName, fileName: path.basename(absPath) };
      }
    }
  }

  // 3. Fallback to existing files matching *locators.ts recursively
  try {
    const locatorFiles = getFilesRecursively(
      projectRoot,
      (name) => name.toLowerCase().endsWith('locators.ts') || name.toLowerCase().endsWith('locators.js'),
      ['node_modules', 'dist', 'build', '.git', '.github', 'test-results', 'playwright-report', 'qa-sync-tool']
    );
    if (locatorFiles.length > 0) {
      const locatorPath = locatorFiles[0];
      const fileName = path.basename(locatorPath);
      // Fix Issue 10: read the actual exported const/let name from the file so we
      // don't incorrectly derive "Locators" from "Locators.ts" when the real
      // export is e.g. "theSouledStoreLocators".
      let varName = path.basename(fileName, path.extname(fileName));
      try {
        const fileText = fs.readFileSync(locatorPath, 'utf-8');
        const exportMatch = fileText.match(/export\s+(?:const|let)\s+(\w+)\s*[=:]/);
        if (exportMatch) varName = exportMatch[1];
      } catch (_) { /* keep file-name fallback */ }
      return { path: locatorPath, varName, fileName };
    }
  } catch (e) { }

  return { path: defaultPath, varName: defaultVarName, fileName: 'locators.ts' };
}

function findLocatorsImport(content: string): { varName: string; importPath: string } | null {
  // Matches e.g. import { appleLocators } from '../appleLocators'
  const regex = /import\s+\{\s*(\w*locators\w*)\s*\}\s+from\s+(['"])(.*?)\2/i;
  const match = content.match(regex);
  if (match) {
    return { varName: match[1], importPath: match[3] };
  }
  return null;
}

export function findConfigFilePath(projectRoot: string): string | null {
  const files = getFilesRecursively(
    projectRoot,
    (name) => (name.toLowerCase() === 'config.ts' || name.toLowerCase() === 'config.js') && name.toLowerCase() !== 'playwright.config.ts',
    ['node_modules', 'dist', 'build', '.git', '.github', 'test-results', 'playwright-report', 'qa-sync-tool']
  );
  return files.length > 0 ? files[0] : null;
}

export function findLocatorsFilePath(projectRoot: string): string | null {
  const files = getFilesRecursively(
    projectRoot,
    (name) => name.toLowerCase().endsWith('locators.ts') || name.toLowerCase().endsWith('locators.js'),
    ['node_modules', 'dist', 'build', '.git', '.github', 'test-results', 'playwright-report', 'qa-sync-tool']
  );
  return files.length > 0 ? files[0] : null;
}

export function getMissingLocators(projectRoot: string, specFiles: string[]): { key: string; file: string; locatorsPath: string }[] {
  const missing: { key: string; file: string; locatorsPath: string }[] = [];

  for (const sf of specFiles) {
    if (!fs.existsSync(sf)) continue;
    const rawContent = fs.readFileSync(sf, 'utf-8');

    // Strip single-line and block comments so we don't pick up commented-out locators
    const strippedContent = rawContent
      .replace(/\/\/[^\n]*/g, '')
      .replace(/\/\*[\s\S]*?\*\//g, '');

    const resolved = resolveLocatorsPathForFile(sf, projectRoot);
    if (!resolved || !fs.existsSync(resolved.path)) continue;

    const locContent = fs.readFileSync(resolved.path, 'utf-8');
    const varName = resolved.varName;

    // Regex matching: varName.parent.child — only inside real code, not strings
    const regex = new RegExp(`\\b${varName}\\.(\\w+)\\.(\\w+)\\b`, 'g');
    let match;
    while ((match = regex.exec(strippedContent)) !== null) {
      const [_, parent, child] = match;
      const key = `${parent}.${child}`;

      const parentRegex = new RegExp(`\\b${parent}\\s*:\\s*\\{[\\s\\S]*?\\b${child}\\s*:`, 'm');
      if (!parentRegex.test(locContent)) {
        if (!missing.some(m => m.key === key && m.locatorsPath === resolved.path)) {
          missing.push({ key, file: sf, locatorsPath: resolved.path });
        }
      }
    }
  }
  return missing;
}

/**
 * Computes a simple hash of the requirements document content for change detection.
 */
export function computeRdHash(content: string): string {
  let hash = 0;
  const str = content.replace(/\s+/g, ' ').trim(); // normalize whitespace
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // convert to 32bit int
  }
  return Math.abs(hash).toString(16);
}

/**
 * Scans all relevant project files (spec, locators, config) for the AI's full context.
 * Returns a map of filename -> content for cross-file impact analysis.
 */
export function scanAllProjectFiles(projectRoot: string): { filePath: string; content: string }[] {
  const results: { filePath: string; content: string }[] = [];
  const skipDirs = ['node_modules', 'dist', 'build', '.git', '.github', 'test-results', 'playwright-report', 'qa-sync-tool', 'scratch'];

  const relevantFiles = getFilesRecursively(
    projectRoot,
    (name) => {
      const ext = path.extname(name).toLowerCase();
      const lower = name.toLowerCase();
      return (ext === '.ts' || ext === '.js' || ext === '.json') &&
        !lower.includes('.spec.') &&
        lower !== 'playwright.config.ts' &&
        lower !== 'package.json' &&
        lower !== 'tsconfig.json';
    },
    skipDirs
  );

  for (const filePath of relevantFiles) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      results.push({ filePath, content });
    } catch (e) { /* ignore */ }
  }

  return results;
}

// Custom ANSI colors for terminal logs
const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const DIM = '\x1b[2m';

export interface Suggestion {
  requirementId: string;
  action: 'ADD' | 'MODIFY' | 'REMOVE' | 'NONE';
  title: string;
  description: string;
  filePath?: string;
  testTitle?: string;
  originalCode?: string;
  proposedCode?: string;
  startLine?: number;
  endLine?: number;
  configUpdates?: { key: string; value: string }[];
  locatorUpdates?: { key: string; value: string }[];
  envUpdates?: { key: string; value: string }[];
  requirementText?: string;
  implementationSummary?: string;
  differencesFound?: string;
  impactAnalysis?: string;
  codeChangesRequired?: 'Yes' | 'No';
  reason?: string;
  patchDiff?: string;
  whyNeeded?: string;
  proposedCodeOpt2?: string;
  patchDiffOpt2?: string;
  whyNeededOpt2?: string;
  classification?: string;
  detectedChanges?: string;
  deleteFile?: boolean;
}

export function findExternalDependencies(importsList: string[]): string[] {
  const standardLibs = new Set([
    '@playwright/test',
    '@google/generative-ai',
    'dotenv',
    'mammoth',
    'path',
    'fs',
    'readline',
    'child_process',
    'crypto',
    'events',
    'os',
    'util',
    'http',
    'https'
  ]);
  const externals: string[] = [];
  for (const imp of importsList) {
    const match = imp.match(/from\s+(['"`])(.*?)\1/) || imp.match(/import\s+(['"`])(.*?)\1/);
    if (match) {
      const lib = match[2];
      if (lib.startsWith('.')) continue;
      
      let pkgName = lib;
      if (lib.startsWith('@')) {
        const parts = lib.split('/');
        if (parts.length >= 2) pkgName = `${parts[0]}/${parts[1]}`;
      } else {
        pkgName = lib.split('/')[0];
      }
      
      if (!standardLibs.has(pkgName)) {
        externals.push(pkgName);
      }
    }
  }
  return Array.from(new Set(externals));
}

export function cleanProposedCode(code: string, stripWrapper = true): string {
  let currentCode = code.trim();
  currentCode = currentCode
    .replace(/^```(?:typescript|ts|javascript|js)?\s*\n?/gim, '')
    .replace(/\n?```\s*$/gim, '')
    .trim();

  if (currentCode.includes('LOW_CONFIDENCE')) {
    currentCode = currentCode
      .split('\n')
      .map(line =>
        line.includes('LOW_CONFIDENCE')
          ? `// SKIPPED: locator is LOW_CONFIDENCE — add a valid selector to Locators.ts before enabling this assertion`
          : line
      )
      .join('\n');
  }

  if (stripWrapper) {
    let prevCode = '';
    while (currentCode !== prevCode) {
      prevCode = currentCode;
      let leadingComments = '';
      const commentRegex = /^(\s*(\/\/.*|\/\*[\s\S]*?\*\/))+/;
      const commentMatch = currentCode.match(commentRegex);
      if (commentMatch) {
        leadingComments = commentMatch[0] + '\n';
        currentCode = currentCode.substring(commentMatch[0].length).trim();
      }

      const testWrapperRegex = /^test(?:\.(?:only|skip|fixme|fail))?\s*\(\s*(['"`])([\s\S]*?)\1\s*,\s*(?:async\s*)?\((?:[\s\S]*?)\)\s*=>\s*\{([\s\S]*)\}\s*\);?\s*$/;
      const match = currentCode.match(testWrapperRegex);
      if (match) {
        currentCode = (leadingComments + match[3].trim()).trim();
      } else {
        currentCode = (leadingComments + currentCode).trim();
        break;
      }
    }
  }

  const lines = currentCode.split('\n');
  const nonEmptyLines = lines.filter(l => l.trim() !== '');
  if (nonEmptyLines.length > 0) {
    const minIndent = Math.min(...nonEmptyLines.map(l => l.match(/^(\s*)/)?.[1].length ?? 0));
    currentCode = lines
      .map(l => {
        if (l.trim() === '') return '';
        const stripped = l.substring(minIndent);
        return stripped;
      })
      .join('\n')
      .trimEnd();
  }

  return currentCode;
}

export function ensureRequirementComment(code: string, requirementId: string, oldRequirementId?: string): string {
  let cleaned = code;
  if (oldRequirementId) {
    const commentRegex = new RegExp(`(//|/\\*)\\s*([^\\w\\s]*\\s*)${oldRequirementId}\\b`, 'gi');
    cleaned = cleaned.replace(commentRegex, `$1 $2${requirementId}`);
  }

  const idRegex = new RegExp(`\\b${requirementId}\\b`, 'i');
  const hasLeadingComment = /^\s*(\/\/|\/\*)/.test(cleaned);

  if (hasLeadingComment) {
    const firstLine = cleaned.split('\n')[0];
    if (!idRegex.test(firstLine)) {
      cleaned = `// ${requirementId}\n` + cleaned;
    }
  } else {
    cleaned = `// ${requirementId}\n` + cleaned;
  }

  return cleaned;
}

export function updateConfigFile(filePath: string, updates: { key: string; value: string }[]) {
  let content = '';
  if (fs.existsSync(filePath)) {
    content = fs.readFileSync(filePath, 'utf-8');
  } else {
    content = `export const config = {\n  baseURL: process.env.BASE_URL || 'http://localhost:3000',\n};\n`;
  }

  let updatedCount = 0;
  for (const item of updates) {
    const escapedKey = item.key.includes('.') || item.key.includes('-') ? `'${item.key}'` : item.key;
    const keyRegex = new RegExp(`(\\b${escapedKey.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')}\\s*:\\s*)(['"\`])[\\s\\S]*?\\2`, 'g');
    if (keyRegex.test(content)) {
      content = content.replace(keyRegex, `$1'${item.value}'`);
      console.log(`  ${CYAN}⚙️ Updated config: ${item.key} -> '${item.value}'${RESET}`);
      updatedCount++;
    } else {
      const lastBrace = content.lastIndexOf('};');
      if (lastBrace !== -1) {
        const beforeStr = content.substring(0, lastBrace).trimEnd();
        let prefix = '';
        if (beforeStr && !beforeStr.endsWith(',') && !beforeStr.endsWith('{')) {
          prefix = ',\n';
        }
        content = content.substring(0, lastBrace) + prefix + `  ${escapedKey}: '${item.value}',\n` + content.substring(lastBrace);
        console.log(`  ${GREEN}✨ Added new config: ${item.key} -> '${item.value}'${RESET}`);
        updatedCount++;
      }
    }
  }

  if (updatedCount > 0) {
    fs.writeFileSync(filePath, content, 'utf-8');
  }
}

export function findOutermostClosingBrace(content: string): number {
  for (let i = content.length - 1; i >= 0; i--) {
    if (content[i] === '}') {
      const suffix = content.substring(i).replace(/\s/g, '');
      if (suffix.startsWith('};') || suffix === '}') {
        return i;
      }
    }
  }
  return content.lastIndexOf('}');
}

export function updateLocatorsFile(filePath: string, updates: { key: string; value: string }[]) {
  let content = '';
  if (fs.existsSync(filePath)) {
    content = fs.readFileSync(filePath, 'utf-8');
  } else {
    content = `export const locators = {\n};\n`;
  }

  const originalContent = content;
  let updatedCount = 0;

  for (const item of updates) {
    const parts = item.key.split('.');
    let escapedVal = item.value.replace(/\\/g, '\\\\');
    // Clean outer quotes if any
    escapedVal = escapedVal.replace(/^(['"`])([\s\S]*)\1$/, '$2');

    if (parts.length === 2) {
      const [parent, child] = parts;
      const parentDeclRegex = new RegExp(`\\b${parent}\\s*:\\s*\\{`);
      const parentDeclMatch = content.match(parentDeclRegex);

      if (parentDeclMatch && parentDeclMatch.index !== undefined) {
        const blockOpenIdx = content.indexOf('{', parentDeclMatch.index + parentDeclMatch[0].length - 1);
        if (blockOpenIdx === -1) continue;

        let depth = 1;
        let i = blockOpenIdx + 1;
        while (i < content.length && depth > 0) {
          if (content[i] === '{') depth++;
          else if (content[i] === '}') depth--;
          i++;
        }
        const blockCloseIdx = i - 1;
        const blockBody = content.substring(blockOpenIdx + 1, blockCloseIdx);
        const childKeyRegex = new RegExp(`\\b${child}\\s*:`);

        if (childKeyRegex.test(blockBody)) {
          const childUpdateRegex = new RegExp(`(\\b${child}\\s*:\\s*)(['"\`])([^'"\`]*)\\2(\\s*,?)`);
          const newBlockBody = blockBody.replace(childUpdateRegex, `$1'${escapedVal}'$4`);
          content = content.substring(0, blockOpenIdx + 1) + newBlockBody + content.substring(blockCloseIdx);
          console.log(`  ${CYAN}⚙️  Updated locator: ${item.key} -> '${item.value}'${RESET}`);
        } else {
          const trimmedBody = blockBody.trimEnd();
          const needsComma = trimmedBody.length > 0 &&
            !trimmedBody.endsWith(',') &&
            !trimmedBody.endsWith('{');
          const newEntry = (needsComma ? ',' : '') + '\n    ' + child + `: '${escapedVal}',`;
          const newBlockBody = trimmedBody + newEntry + '\n  ';
          content = content.substring(0, blockOpenIdx + 1) + newBlockBody + content.substring(blockCloseIdx);
          console.log(`  ${GREEN}✨ Added new locator: ${item.key} -> '${item.value}'${RESET}`);
        }
        updatedCount++;
      } else {
        const lastBraceIdx = findOutermostClosingBrace(content);
        if (lastBraceIdx !== -1) {
          const before = content.substring(0, lastBraceIdx).trimEnd();
          const needsComma = before.length > 0 && !before.endsWith(',') && !before.endsWith('{');
          const newBlock = (needsComma ? ',' : '') + '\n  ' + parent + ': {\n    ' + child + `: '${escapedVal}',\n  }`;
          content = before + newBlock + '\n' + content.substring(lastBraceIdx);
          console.log(`  ${GREEN}✨ Created parent block and added locator: ${item.key} -> '${item.value}'${RESET}`);
          updatedCount++;
        }
      }
    } else {
      const escapedKey = item.key.includes('.') || item.key.includes('-') ? `'${item.key}'` : item.key;
      const topKeyRaw = escapedKey.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
      const topKeyRegex = new RegExp(`(\\b${topKeyRaw}\\s*:\\s*)(['"\`])[^'"\`]*\\2(\\s*,?)`);

      if (topKeyRegex.test(content)) {
        content = content.replace(topKeyRegex, `$1'${escapedVal}'$3`);
        console.log(`  ${CYAN}⚙️  Updated top-level locator: ${item.key} -> '${item.value}'${RESET}`);
        updatedCount++;
      } else {
        const lastBraceIdx = findOutermostClosingBrace(content);
        if (lastBraceIdx !== -1) {
          const before = content.substring(0, lastBraceIdx).trimEnd();
          const needsComma = before.length > 0 && !before.endsWith(',') && !before.endsWith('{');
          content = before + (needsComma ? ',' : '') + '\n  ' + escapedKey + `: '${escapedVal}',` + '\n' + content.substring(lastBraceIdx);
          console.log(`  ${GREEN}✨ Added top-level locator: ${item.key} -> '${item.value}'${RESET}`);
          updatedCount++;
        }
      }
    }
  }

  if (updatedCount > 0) {
    content = content.replace(/\n{3,}/g, '\n\n');
    fs.writeFileSync(filePath, content, 'utf-8');

    try {
      const { execSync } = require('child_process');
      let tscRoot = path.dirname(filePath);
      let maxUp = 6;
      while (maxUp-- > 0 && tscRoot !== path.dirname(tscRoot)) {
        if (fs.existsSync(path.join(tscRoot, 'tsconfig.json'))) break;
        tscRoot = path.dirname(tscRoot);
      }
      execSync(`npx tsc --noEmit --skipLibCheck`, { stdio: 'pipe', cwd: tscRoot });
    } catch (tscError: any) {
      const errOut = ((tscError.stdout || tscError.stderr || '').toString()).substring(0, 600);
      console.warn(`  ${YELLOW}⚠️ TypeScript validation failed for ${path.basename(filePath)} — rolling back locator changes.${RESET}`);
      if (errOut) console.warn(`  ${DIM}${errOut}${RESET}`);
      fs.writeFileSync(filePath, originalContent, 'utf-8');
    }
  }
}

export function updateEnvFile(filePath: string, updates: { key: string; value: string }[]) {
  let content = '';
  if (fs.existsSync(filePath)) {
    content = fs.readFileSync(filePath, 'utf-8');
  }

  let added = 0;
  for (const item of updates) {
    if (!content.includes(`${item.key}=`)) {
      content += `${item.key}="${item.value}"\n`;
      added++;
    }
  }
  if (added > 0) {
    fs.writeFileSync(filePath, content, 'utf-8');
    console.log(`  ${CYAN}⚙️ Updated environment variables: .env (${added} entries)${RESET}`);
  }
}

export function healSpecImports(projectRoot: string, specFiles: string[]) {
  const configPath = findConfigFilePath(projectRoot);
  if (!configPath) return;

  for (const filePath of specFiles) {
    if (!fs.existsSync(filePath)) continue;
    let content = fs.readFileSync(filePath, 'utf-8');
    let changed = false;

    const configImportRegex = /(import\s+\{\s*config\s*\}\s+from\s+)(['"])(.*?)\2(;?)/g;
    content = content.replace(configImportRegex, (match, prefix, quote, importPath, suffix) => {
      const correctRelative = path.relative(path.dirname(filePath), configPath);
      let importConfigPath = correctRelative.replace(/\\/g, '/');
      if (importConfigPath.endsWith('.ts')) {
        importConfigPath = importConfigPath.substring(0, importConfigPath.length - 3);
      }
      if (!importConfigPath.startsWith('.') && !importConfigPath.startsWith('/')) {
        importConfigPath = './' + importConfigPath;
      }
      if (importPath !== importConfigPath) {
        changed = true;
        console.log(`  🔧 Healing config import in ${path.basename(filePath)}: from "${importPath}" to "${importConfigPath}"`);
        return `${prefix}${quote}${importConfigPath}${quote}${suffix}`;
      }
      return match;
    });

    const resolvedLocators = resolveLocatorsPathForFile(filePath, projectRoot);
    const locatorImportRegex = /(import\s+\{\s*(\w*locators\w*)\s*\}\s+from\s+)(['"])(.*?)\3(;?)/gi;
    content = content.replace(locatorImportRegex, (match, prefix, varName, quote, importPath, suffix) => {
      const correctRelative = path.relative(path.dirname(filePath), resolvedLocators.path);
      let importLocatorsPath = correctRelative.replace(/\\/g, '/');
      if (importLocatorsPath.endsWith('.ts')) {
        importLocatorsPath = importLocatorsPath.substring(0, importLocatorsPath.length - 3);
      }
      if (!importLocatorsPath.startsWith('.') && !importLocatorsPath.startsWith('/')) {
        importLocatorsPath = './' + importLocatorsPath;
      }
      if (importPath !== importLocatorsPath) {
        changed = true;
        console.log(`  🔧 Healing locator import in ${path.basename(filePath)}: from "${importPath}" to "${importLocatorsPath}"`);
        return `${prefix}${quote}${importLocatorsPath}${quote}${suffix}`;
      }
      return match;
    });

    if (changed) {
      fs.writeFileSync(filePath, content, 'utf-8');
    }
  }
}

export function optimizeAndCleanImports(fileContent: string): string {
  const lines = fileContent.split(/\r?\n/);
  const importLines: string[] = [];
  const nonImportLines: string[] = [];

  let inMultilineImport = false;
  let currentImport = '';

  for (const line of lines) {
    const trimmed = line.trim();
    if (inMultilineImport) {
      currentImport += ' ' + trimmed;
      if (trimmed.includes('}')) {
        const fromMatch = line.match(/from\s+(['"`].*?['"`])/);
        if (fromMatch || trimmed.endsWith(';') || trimmed.endsWith("'") || trimmed.endsWith('"')) {
          inMultilineImport = false;
          importLines.push(currentImport);
          currentImport = '';
        }
      }
    } else if (trimmed.startsWith('import ') || trimmed.startsWith('import{')) {
      if (trimmed.includes('{') && !trimmed.includes('}')) {
        inMultilineImport = true;
        currentImport = trimmed;
      } else {
        importLines.push(trimmed);
      }
    } else {
      nonImportLines.push(line);
    }
  }

  const moduleImports = new Map<string, {
    named: Set<string>;
    default?: string;
    namespace?: string;
  }>();

  const reconstructedImports: string[] = [];

  for (const imp of importLines) {
    const mixedMatch = imp.match(/import\s+(\w+)\s*,\s*\{([^}]+)\}\s*from\s*(['"`])(.*?)\3;?/);
    if (mixedMatch) {
      const modulePath = mixedMatch[4];
      const defaultImport = mixedMatch[1];
      const members = mixedMatch[2].split(',').map(m => m.trim()).filter(Boolean);
      const existing = moduleImports.get(modulePath) || { named: new Set<string>() };
      existing.default = defaultImport;
      members.forEach(m => existing.named.add(m));
      moduleImports.set(modulePath, existing);
      continue;
    }

    const namedMatch = imp.match(/import\s+(?:type\s+)?\{([^}]+)\}\s*from\s*(['"`])(.*?)\2;?/);
    if (namedMatch) {
      const modulePath = namedMatch[3];
      const members = namedMatch[1].split(',').map(m => m.trim()).filter(Boolean);
      const existing = moduleImports.get(modulePath) || { named: new Set<string>() };
      members.forEach(m => existing.named.add(m));
      moduleImports.set(modulePath, existing);
      continue;
    }

    const nsMatch = imp.match(/import\s+\*\s+as\s+(\w+)\s*from\s*(['"`])(.*?)\2;?/);
    if (nsMatch) {
      const modulePath = nsMatch[3];
      const namespace = nsMatch[1];
      const existing = moduleImports.get(modulePath) || { named: new Set<string>() };
      existing.namespace = namespace;
      moduleImports.set(modulePath, existing);
      continue;
    }

    const defaultMatch = imp.match(/import\s+(\w+)\s*from\s*(['"`])(.*?)\2;?/);
    if (defaultMatch) {
      const modulePath = defaultMatch[3];
      const defaultImport = defaultMatch[1];
      const existing = moduleImports.get(modulePath) || { named: new Set<string>() };
      existing.default = defaultImport;
      moduleImports.set(modulePath, existing);
      continue;
    }

    reconstructedImports.push(imp);
  }

  const bodyText = nonImportLines.join('\n');

  for (const [modulePath, info] of moduleImports.entries()) {
    const activeNamed: string[] = [];
    for (const name of info.named) {
      const parts = name.split(/\s+as\s+/);
      const localName = parts[parts.length - 1].trim();
      const regex = new RegExp(`\\b${localName}\\b`);
      if (regex.test(bodyText)) {
        activeNamed.push(name);
      }
    }

    let useDefault = info.default;
    if (useDefault) {
      const regex = new RegExp(`\\b${useDefault}\\b`);
      if (!regex.test(bodyText)) {
        useDefault = undefined;
      }
    }

    let useNamespace = info.namespace;
    if (useNamespace) {
      const regex = new RegExp(`\\b${useNamespace}\\b`);
      if (!regex.test(bodyText)) {
        useNamespace = undefined;
      }
    }

    const isCoreModule = modulePath.includes('config') || modulePath.includes('Locators') || modulePath === '@playwright/test';
    if (isCoreModule) {
      if (modulePath === '@playwright/test') {
        info.named.forEach(n => {
          if (n === 'test' || n === 'expect' || n === 'Page') {
            if (!activeNamed.includes(n)) activeNamed.push(n);
          }
        });
      }
      if (info.default && !useDefault) useDefault = info.default;
      if (info.namespace && !useNamespace) useNamespace = info.namespace;
    }

    if (activeNamed.length > 0 || useDefault || useNamespace) {
      if (useNamespace) {
        reconstructedImports.push(`import * as ${useNamespace} from '${modulePath}';`);
      } else if (useDefault && activeNamed.length > 0) {
        reconstructedImports.push(`import ${useDefault}, { ${activeNamed.join(', ')} } from '${modulePath}';`);
      } else if (useDefault) {
        reconstructedImports.push(`import ${useDefault} from '${modulePath}';`);
      } else if (activeNamed.length > 0) {
        reconstructedImports.push(`import { ${activeNamed.join(', ')} } from '${modulePath}';`);
      }
    }
  }

  let combined = [...reconstructedImports, '', ...nonImportLines].join('\n');
  combined = combined.replace(/\n{3,}/g, '\n\n');
  return combined;
}

export function cleanUnusedCode(content: string): string {
  let code = content;
  let iteration = 0;
  const maxIterations = 5;

  while (iteration < maxIterations) {
    const originalCode = code;

    // 1. Remove empty describe blocks
    const emptyDescribeRegex = /test\.describe\(\s*(['"`])[\s\S]*?\1\s*,\s*(?:async\s*)?\(\s*\)\s*=>\s*\{\s*\}\s*\);?/g;
    code = code.replace(emptyDescribeRegex, '');

    // 2. Find all function declarations: function <name>(...) { ... }
    const funcDeclRegex = /(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(/g;
    let match;
    const functionsToRemove: { start: number; end: number; name: string }[] = [];

    while ((match = funcDeclRegex.exec(code)) !== null) {
      const funcName = match[1];
      const startIdx = match.index;
      
      // Find the opening brace '{'
      let braceIdx = code.indexOf('{', startIdx);
      if (braceIdx === -1) continue;

      // Find matching closing brace
      let openBraces = 1;
      let i = braceIdx + 1;
      while (i < code.length && openBraces > 0) {
        if (code[i] === '{') openBraces++;
        else if (code[i] === '}') openBraces--;
        i++;
      }

      if (openBraces === 0) {
        const endIdx = i;
        const blockText = code.substring(startIdx, endIdx);
        
        // Count occurrences of funcName outside of this block
        const escapedName = funcName.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
        const regex = new RegExp(`\\b${escapedName}\\b`, 'g');
        const totalMatches = (code.match(regex) || []).length;
        const blockMatches = (blockText.match(regex) || []).length;

        // If all occurrences are within the block itself, it is unused!
        if (totalMatches <= blockMatches) {
          functionsToRemove.push({ start: startIdx, end: endIdx, name: funcName });
        }
      }
    }

    // Remove functions from end to start to keep indices valid
    functionsToRemove.sort((a, b) => b.start - a.start);
    for (const fn of functionsToRemove) {
      code = code.substring(0, fn.start) + code.substring(fn.end);
      console.log(`  - Cleaned up unused helper function: "${fn.name}"`);
    }

    // 3. Find and remove unused const/let variable declarations
    const varDeclRegex = /\b(const|let)\s+(\w+)\s*=\s*([^;]+);/g;
    const varsToRemove: { start: number; end: number; name: string }[] = [];
    while ((match = varDeclRegex.exec(code)) !== null) {
      const varName = match[2];
      const startIdx = match.index;
      const endIdx = match.index + match[0].length;

      // Skip common page objects or configurations we want to keep
      if (varName === 'config' || varName === 'theSouledStoreLocators') continue;

      const escapedName = varName.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
      const regex = new RegExp(`\\b${escapedName}\\b`, 'g');
      const totalMatches = (code.match(regex) || []).length;

      // If it only appears once (its declaration), it is unused!
      if (totalMatches === 1) {
        varsToRemove.push({ start: startIdx, end: endIdx, name: varName });
      }
    }

    varsToRemove.sort((a, b) => b.start - a.start);
    for (const v of varsToRemove) {
      code = code.substring(0, v.start) + code.substring(v.end);
      console.log(`  - Cleaned up unused variable: "${v.name}"`);
    }

    if (code === originalCode) {
      break;
    }
    iteration++;
  }

  code = code.replace(/\n{3,}/g, '\n\n').trim() + '\n';
  return code;
}

export function applyChanges(
  projectRoot: string,
  suggestions: Suggestion[],
  specFiles: string[],
  validRequirementIds: string[]
) {
  const testsDir = specFiles.length > 0
    ? path.dirname(specFiles[0])
    : path.join(projectRoot, 'tests');
  if (!fs.existsSync(testsDir)) {
    fs.mkdirSync(testsDir, { recursive: true });
  }

  const configUpdates: { key: string; value: string }[] = [];
  const envUpdates: { key: string; value: string }[] = [];

  for (const sug of suggestions) {
    if (sug.configUpdates) configUpdates.push(...sug.configUpdates);
    if (sug.envUpdates) envUpdates.push(...sug.envUpdates);
  }

  if (configUpdates.length > 0) {
    const configPath = findConfigFilePath(projectRoot) || path.join(projectRoot, 'config.ts');
    updateConfigFile(configPath, configUpdates);
  }
  if (envUpdates.length > 0) {
    updateEnvFile(path.join(projectRoot, '.env'), envUpdates);
  }

  let defaultSpecPath = specFiles.length > 0 ? specFiles[0] : path.join(testsDir, 'playwright.spec.ts');

  const locatorUpdatesByFile = new Map<string, { key: string; value: string }[]>();
  for (const sug of suggestions) {
    if (sug.locatorUpdates && sug.locatorUpdates.length > 0) {
      const specFilePath = sug.filePath ? path.resolve(projectRoot, sug.filePath) : defaultSpecPath;
      const resolved = resolveLocatorsPathForFile(specFilePath, projectRoot);
      const updates = locatorUpdatesByFile.get(resolved.path) || [];
      updates.push(...sug.locatorUpdates);
      locatorUpdatesByFile.set(resolved.path, updates);
    }
  }

  for (const [locPath, updates] of locatorUpdatesByFile.entries()) {
    updateLocatorsFile(locPath, updates);
  }

  const fileSuggestionsMap = new Map<string, Suggestion[]>();
  for (const sug of suggestions) {
    if (sug.action === 'NONE') continue;

    let targetFile = sug.filePath ? path.resolve(projectRoot, sug.filePath) : defaultSpecPath;
    const list = fileSuggestionsMap.get(targetFile) || [];
    list.push(sug);
    fileSuggestionsMap.set(targetFile, list);
  }

  for (const [filePath, fileSugs] of fileSuggestionsMap.entries()) {
    const fileExists = fs.existsSync(filePath);
    let segments: FileSegment[] = [];

    if (fileExists) {
      const content = fs.readFileSync(filePath, 'utf-8');
      segments = parseTestFileStructure(content, filePath);
    } else {
      const configPath = findConfigFilePath(projectRoot);
      const resolved = resolveLocatorsPathForFile(filePath, projectRoot);
      const relConfig = configPath ? path.relative(path.dirname(filePath), configPath).replace(/\\/g, '/').replace(/\.ts$/, '') : '../config/config';
      const relLocators = resolved ? path.relative(path.dirname(filePath), resolved.path).replace(/\\/g, '/').replace(/\.ts$/, '') : '../config/Locators';
      const importConfigPath = relConfig.startsWith('.') ? relConfig : './' + relConfig;
      const importLocatorsPath = relLocators.startsWith('.') ? relLocators : './' + relLocators;
      const locatorVarName = resolved?.varName || 'theSouledStoreLocators';
      const describeTitle = path.basename(filePath, '.spec.ts').replace(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase()).replace(/'/g, "\\'");
      // Scan locator file for a dismiss/popup/close key in any group; fall back to CSS selector
      let scaffoldDismissExpr = `page.locator('[data-dismiss], .popup-close, .modal-close').first()`;
      try {
        if (resolved && fs.existsSync(resolved.path)) {
          const locContent = fs.readFileSync(resolved.path, 'utf-8');
          const dismissKeyMatch = locContent.match(/(\w+)\s*:\s*\{[^}]*?(\w*(?:dismiss|Dismiss|popup|Popup|close|Close)\w*)\s*:/);
          if (dismissKeyMatch) {
            scaffoldDismissExpr = `page.locator(${locatorVarName}.${dismissKeyMatch[1]}.${dismissKeyMatch[2]})`;
          }
        }
      } catch (e) { /* use CSS fallback */ }
      const headerText = `import { config } from '${importConfigPath}';
import { ${locatorVarName} } from '${importLocatorsPath}';
import { test, expect, Page } from '@playwright/test';

async function dismissMembershipPopup(page: Page) {
  const dismissBtn = ${scaffoldDismissExpr};
  try {
    await dismissBtn.waitFor({ state: 'visible', timeout: 2000 });
    await dismissBtn.click();
    await dismissBtn.waitFor({ state: 'hidden', timeout: 2000 });
  } catch (e) { /* Popup did not appear */ }
}

test.describe('${describeTitle}', () => {\n`;
      const footerText = `\n});\n`;
      segments = [
        { type: 'text', content: headerText },
        { type: 'text', content: footerText }
      ];
    }

    const normalizeCode = (c: string) => c.replace(/\s+/g, ' ').trim();

    for (const sug of fileSugs) {
      if (sug.action === 'REMOVE') {
        const normSugId = normalizeRequirementId(sug.requirementId);
        let matchedCaseIndex = segments.findIndex(seg =>
          seg.type === 'testCase' && (
            seg.structure.title === sug.testTitle ||
            seg.structure.blocks.some(b => {
              const bIds = (b.requirementIds || (b.requirementId ? [b.requirementId] : [])).map(normalizeRequirementId);
              return bIds.includes(normSugId) ||
                (sug.originalCode && normalizeCode(b.code) === normalizeCode(sug.originalCode)) ||
                (sug.originalCode && b.requirementId &&
                  normalizeCode(sug.originalCode).includes(normalizeRequirementId(b.requirementId)) &&
                  !validRequirementIds.includes(normalizeRequirementId(b.requirementId)));
            })
          )
        );

        if (matchedCaseIndex !== -1) {
          const tcSeg = segments[matchedCaseIndex] as { type: 'testCase'; structure: any };
          const tc = tcSeg.structure;

          const remainingBlocks = tc.blocks.filter((b: any) => {
            const bIds = (b.requirementIds || (b.requirementId ? [b.requirementId] : [])).map(normalizeRequirementId);
            return !(
              bIds.includes(normSugId) ||
              (sug.originalCode && normalizeCode(b.code) === normalizeCode(sug.originalCode)) ||
              (sug.originalCode && b.requirementId &&
                normalizeCode(sug.originalCode).includes(normalizeRequirementId(b.requirementId)) &&
                !validRequirementIds.includes(normalizeRequirementId(b.requirementId)))
            );
          });

          const hasRemainingRequirement = remainingBlocks.some((b: any) => b.type === 'requirement');

          if (!hasRemainingRequirement) {
            segments.splice(matchedCaseIndex, 1);
            console.log(`  ${RED}- Removed entire deprecated test: "${sug.testTitle || tc.title}" from ${path.basename(filePath)}${RESET}`);
          } else {
            tc.blocks = remainingBlocks;
            console.log(`  ${RED}- Removed deprecated block ${sug.requirementId} from test: "${tc.title}" in ${path.basename(filePath)}${RESET}`);
          }
        }
      }
    }

    for (const sug of fileSugs) {
      if (sug.action === 'MODIFY') {
        const cleanedProposed = sug.proposedCode ? cleanProposedCode(sug.proposedCode) : '';
        const normSugId = normalizeRequirementId(sug.requirementId);
        let matchedCase = segments.find(seg =>
          seg.type === 'testCase' && (
            seg.structure.title === sug.testTitle ||
            seg.structure.blocks.some(b => {
              const bIds = (b.requirementIds || (b.requirementId ? [b.requirementId] : [])).map(normalizeRequirementId);
              return bIds.includes(normSugId) ||
                (sug.originalCode && normalizeCode(b.code) === normalizeCode(sug.originalCode)) ||
                (sug.originalCode && b.requirementId &&
                  normalizeCode(sug.originalCode).includes(normalizeRequirementId(b.requirementId)) &&
                  !validRequirementIds.includes(normalizeRequirementId(b.requirementId)));
            })
          )
        );

        if (matchedCase && matchedCase.type === 'testCase') {
          const tc = matchedCase.structure;
          // Primary match: find block by requirement ID or matching original code
          let blockIndex = tc.blocks.findIndex((b: any) => {
            const bIds = (b.requirementIds || (b.requirementId ? [b.requirementId] : [])).map(normalizeRequirementId);
            return bIds.includes(normSugId) ||
              (sug.originalCode && normalizeCode(b.code) === normalizeCode(sug.originalCode)) ||
              (sug.originalCode && b.requirementId &&
                normalizeCode(sug.originalCode).includes(normalizeRequirementId(b.requirementId)) &&
                !validRequirementIds.includes(normalizeRequirementId(b.requirementId)));
          });

          if (blockIndex === -1 && tc.blocks.length === 1) {
            blockIndex = 0;
          }

          if (blockIndex !== -1) {
            // ── Standard path: replace the matched block in place ──────────────
            const matchedBlock = tc.blocks[blockIndex];
            const oldId = (matchedBlock.requirementId !== sug.requirementId) ? matchedBlock.requirementId : undefined;

            const hasDeletionPatch = sug.patchDiff && sug.patchDiff.split('\n').some(l => l.trim().startsWith('-'));
            let newCode = '';
            if (hasDeletionPatch && sug.patchDiff) {
              newCode = applyPatchDiff(matchedBlock.code, sug.patchDiff);
            } else {
              newCode = cleanedProposed;
            }

            tc.blocks[blockIndex].code = ensureRequirementComment(newCode, sug.requirementId, oldId);
            tc.blocks[blockIndex].requirementId = sug.requirementId;
            tc.blocks[blockIndex].requirementIds = [sug.requirementId];
            console.log(`  ${YELLOW}* Modified block ${sug.requirementId} in test: "${tc.title}" in ${path.basename(filePath)}${RESET}`);

          } else {
            // ── Fallback: SCOPE C patch — merge missing lines into existing block ──
            // A MODIFY must NEVER push a new block. Find the best target block to merge into:
            //   1. Any block whose requirementId is the same (case-insensitive substring match)
            //   2. The last 'requirement' block in the test
            //   3. The last block of any type in the test
            let targetIdx = tc.blocks.findIndex((b: any) =>
              b.requirementId && normalizeRequirementId(b.requirementId).includes(normSugId)
            );
            if (targetIdx === -1) {
              targetIdx = [...tc.blocks].reverse().findIndex((b: any) => b.type === 'requirement');
              if (targetIdx !== -1) targetIdx = tc.blocks.length - 1 - targetIdx;
            }
            if (targetIdx === -1 && tc.blocks.length > 0) {
              targetIdx = tc.blocks.length - 1;
            }

            if (targetIdx !== -1) {
              // Merge: append only the lines from cleanedProposed that are not already in the block
              const existingBlock = tc.blocks[targetIdx];
              const existingLines = new Set(
                existingBlock.code.split('\n').map((l: string) => l.trim()).filter(Boolean)
              );
              const newLines = cleanedProposed
                .split('\n')
                .filter((l: string) => {
                  const t = l.trim();
                  // Skip lines that are already present, empty, or are bare comment-only lines
                  return t.length > 0 && !existingLines.has(t);
                });

              if (newLines.length > 0) {
                // Insert new lines before the closing brace of the block
                const blockLines = existingBlock.code.split('\n');
                // Find last non-empty, non-closing-brace line to insert after
                let insertAt = blockLines.length;
                for (let bi = blockLines.length - 1; bi >= 0; bi--) {
                  const stripped = blockLines[bi].trimEnd();
                  if (stripped.endsWith('}') || stripped === '') continue;
                  insertAt = bi + 1;
                  break;
                }
                blockLines.splice(insertAt, 0, ...newLines);
                tc.blocks[targetIdx].code = ensureRequirementComment(blockLines.join('\n'), sug.requirementId);
                tc.blocks[targetIdx].requirementIds = [
                  ...(tc.blocks[targetIdx].requirementIds || []),
                  sug.requirementId
                ].filter((v, i, a) => a.indexOf(v) === i);
                console.log(`  ${YELLOW}* Patched ${sug.requirementId} into existing block in test: "${tc.title}" in ${path.basename(filePath)}${RESET}`);
              } else {
                console.log(`  ${DIM}⏭ MODIFY for ${sug.requirementId} skipped — all patch lines already present in "${tc.title}"${RESET}`);
              }
            } else {
              // No block at all found — cannot apply a MODIFY as an ADD. Warn and skip.
              console.log(`  ${YELLOW}⚠ MODIFY for ${sug.requirementId} skipped — no existing block found in "${tc.title}" to patch into. Use ADD instead.${RESET}`);
            }
          }
        }
      }
    }

    for (const sug of fileSugs) {
      if (sug.action === 'ADD' && sug.proposedCode) {
        // Preserve test() and describe() wrappers intact — scope decides how to insert
        const rawCode = cleanProposedCode(sug.proposedCode, false);

        // ── Scope detection ────────────────────────────────────────────────
        // SCOPE A: Full file or full describe block (has test.describe or root-level imports)
        // SCOPE B: Bare test() block — insert into existing describe
        const isScopeA = rawCode.includes('test.describe(') || /^\s*import\s+/m.test(rawCode);
        const isScopeB = !isScopeA && /\btest\s*\(/.test(rawCode);

        if (isScopeA && !fileExists) {
          // ── SCOPE A, NEW FILE: write AI output directly; discard scaffold ──
          segments = [{ type: 'text', content: rawCode }];

        } else if (isScopeB) {
          // ── SCOPE B: Insert bare test() block into existing describe ───────
          const titleMatch = rawCode.match(/\btest(?:\.(?:only|skip|fixme|fail))?\s*\(\s*(['"`])([\s\S]*?)\1/);
          const testTitle = titleMatch ? titleMatch[2] : (sug.testTitle || `${sug.requirementId}: ${sug.title}`);

          // Duplicate title guard
          if (segments.some(seg => seg.type === 'testCase' && seg.structure.title === testTitle)) {
            console.log(`  ${YELLOW}⚠ Skipping ADD for ${sug.requirementId}: test "${testTitle}" already exists in ${path.basename(filePath)}${RESET}`);
            continue;
          }

          // Annotate with requirement ID comment and add 2-space indent for inside-describe placement
          const annotated = ensureRequirementComment(rawCode.trim(), sug.requirementId);
          const indented = '\n\n  ' + annotated.split('\n').join('\n  ') + '\n';
          const textSeg: FileSegment = { type: 'text', content: indented };

          // Find insertion point: scan backwards for the describe closing bracket
          let inserted = false;
          for (let i = segments.length - 1; i >= 0; i--) {
            const seg = segments[i];
            if (seg.type === 'text') {
              const txt = (seg as { type: 'text'; content: string }).content;
              const closeMatch = txt.match(/^([\s\S]*?)(\n?\s*\}\s*\)\s*;?\s*)$/);
              if (closeMatch && closeMatch[2].trim().length > 0) {
                segments.splice(i, 1,
                  { type: 'text', content: closeMatch[1] },
                  textSeg,
                  { type: 'text', content: closeMatch[2] }
                );
                inserted = true;
                break;
              }
            } else if (seg.type === 'testCase') {
              segments.splice(i + 1, 0, textSeg);
              inserted = true;
              break;
            }
          }
          if (!inserted) segments.push(textSeg);

        } else if (isScopeA) {
          // ── SCOPE A, EXISTING FILE: extract imports/helpers; insert describe body ──
          const importRegex = /^import\s+[\s\S]*?from\s+(['"`]).*?\1\s*;?\r?\n?/gm;
          const importsList: string[] = [];
          let importMatch;
          while ((importMatch = importRegex.exec(rawCode)) !== null) {
            importsList.push(importMatch[0].trimEnd());
          }
          let bodyCode = rawCode.replace(/^import\s+[\s\S]*?from\s+(['"`]).*?\1\s*;?\r?\n?/gm, '').trim();

          // Extract helpers: function declarations AND const arrow functions
          const helperFuncRegex = /(?:(?:async\s+)?function\s+\w+\s*\([^)]*\)\s*(?::\s*\w[\w<>, \s[\]|]*\s*)?\{[\s\S]*?\n\}|const\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[\w<>[\]|,\s]+\s*)?=>\s*\{[\s\S]*?\n\})/g;
          const helpersList: string[] = [];
          let helperMatch;
          while ((helperMatch = helperFuncRegex.exec(bodyCode)) !== null) {
            helpersList.push(helperMatch[0]);
          }
          bodyCode = bodyCode.replace(helperFuncRegex, '').trim();

          // Merge non-duplicate imports and helpers into the first segment header
          if (segments.length > 0 && segments[0].type === 'text') {
            const existingContent = segments[0].content;
            const newImports = importsList.filter(imp => {
              const modMatch = imp.match(/from\s+(['"`])(.*?)\1/);
              return modMatch && !existingContent.includes(modMatch[2]);
            });
            const newHelpers = helpersList.filter(h => {
              const nm = h.match(/(?:function\s+(\w+)|const\s+(\w+)\s*=)/);
              const name = nm?.[1] || nm?.[2];
              return name && !existingContent.includes(name);
            });
            const importsText = newImports.length > 0 ? newImports.join('\n') + '\n' : '';
            const helpersText = newHelpers.length > 0 ? '\n' + newHelpers.join('\n\n') + '\n' : '';
            segments[0].content = importsText + existingContent + helpersText;
          }

          // Wrap bare code in describe if helper extraction removed the outer describe
          if (!bodyCode.includes('test.describe(')) {
            const escapedTitle = sug.title.replace(/'/g, "\\'");
            bodyCode = `test.describe('${sug.requirementId}: ${escapedTitle}', () => {\n\n${bodyCode}\n\n});`;
          }
          const annotatedBody = ensureRequirementComment(bodyCode, sug.requirementId);
          const textSeg: FileSegment = { type: 'text', content: '\n\n' + annotatedBody + '\n' };

          // Append after the last test segment or at end of file
          let lastTestSegIdx = -1;
          for (let i = segments.length - 1; i >= 0; i--) {
            if (segments[i].type === 'testCase') { lastTestSegIdx = i; break; }
          }
          if (lastTestSegIdx !== -1) {
            segments.splice(lastTestSegIdx + 1, 0, textSeg);
          } else {
            const lastIdx = segments.length - 1;
            const lastSeg = lastIdx >= 0 ? segments[lastIdx] : undefined;
            let done = false;
            if (lastSeg && lastSeg.type === 'text') {
              const txt = (lastSeg as { type: 'text'; content: string }).content;
              const closeMatch = txt.match(/^([\s\S]*?)(\n?\s*\}\s*\)\s*;?\s*)$/);
              if (closeMatch && closeMatch[2].trim().length > 0) {
                segments.splice(lastIdx, 1,
                  { type: 'text', content: closeMatch[1] },
                  textSeg,
                  { type: 'text', content: closeMatch[2] }
                );
                done = true;
              }
            }
            if (!done) segments.push(textSeg);
          }

        } else {
          // ── Fallback: bare assertions without any wrapper ──────────────────
          const cleanedFallback = cleanProposedCode(sug.proposedCode, true);
          const escapedFbTitle = sug.title.replace(/'/g, "\\'");
          const wrappedFallback = `test.describe('${sug.requirementId}: ${escapedFbTitle}', () => {\n\n${cleanedFallback}\n\n});`;
          const annotatedFallback = ensureRequirementComment(wrappedFallback, sug.requirementId);
          const textSegFb: FileSegment = { type: 'text', content: '\n\n' + annotatedFallback + '\n' };

          let lastFbIdx = -1;
          for (let i = segments.length - 1; i >= 0; i--) {
            if (segments[i].type === 'testCase') { lastFbIdx = i; break; }
          }
          if (lastFbIdx !== -1) {
            segments.splice(lastFbIdx + 1, 0, textSegFb);
          } else {
            const fbLastIdx = segments.length - 1;
            const fbLastSeg = fbLastIdx >= 0 ? segments[fbLastIdx] : undefined;
            let fbDone = false;
            if (fbLastSeg && fbLastSeg.type === 'text') {
              const fbTxt = (fbLastSeg as { type: 'text'; content: string }).content;
              const fbClose = fbTxt.match(/^([\s\S]*?)(\n?\s*\}\s*\)\s*;?\s*)$/);
              if (fbClose && fbClose[2].trim().length > 0) {
                segments.splice(fbLastIdx, 1,
                  { type: 'text', content: fbClose[1] },
                  textSegFb,
                  { type: 'text', content: fbClose[2] }
                );
                fbDone = true;
              }
            }
            if (!fbDone) segments.push(textSegFb);
          }
        }

        console.log(`  ${GREEN}+ Created new test case block for ${sug.requirementId} in ${path.basename(filePath)}${RESET}`);
      }
    }

    let fileContent = reconstructTestFile(segments);

    // If deleteFile is suggested or removing the last test leaves no test cases, delete the entire file
    const shouldDeleteFile = fileSugs.some(s => s.deleteFile) || !/\btest\s*\(/.test(fileContent);
    if (shouldDeleteFile) {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
        console.log(`  ${RED}- Deleted empty/deprecated test file: ${path.basename(filePath)}${RESET}`);
      }
      continue;
    }

    let imports = '';
    if (fileContent.includes('config.') && !fileContent.includes('import { config }')) {
      const configPath = findConfigFilePath(projectRoot) || path.join(projectRoot, 'config.ts');
      const relativeConfigPath = path.relative(path.dirname(filePath), configPath);
      let importConfigPath = relativeConfigPath.replace(/\\/g, '/');
      if (importConfigPath.endsWith('.ts')) {
        importConfigPath = importConfigPath.substring(0, importConfigPath.length - 3);
      }
      if (!importConfigPath.startsWith('.') && !importConfigPath.startsWith('/')) {
        importConfigPath = './' + importConfigPath;
      }
      imports += `import { config } from '${importConfigPath}';\n`;
    }

    const resolvedLocators = resolveLocatorsPathForFile(filePath, projectRoot);
    const locatorVarName = resolvedLocators.varName;

    const locatorVarNameRegex = new RegExp(`\\b${locatorVarName}\\b`);
    if (locatorVarNameRegex.test(fileContent) && !fileContent.includes(`import { ${locatorVarName} }`)) {
      const relativeLocatorsPath = path.relative(path.dirname(filePath), resolvedLocators.path);
      let importLocatorsPath = relativeLocatorsPath.replace(/\\/g, '/');
      if (importLocatorsPath.endsWith('.ts')) {
        importLocatorsPath = importLocatorsPath.substring(0, importLocatorsPath.length - 3);
      }
      if (!importLocatorsPath.startsWith('.') && !importLocatorsPath.startsWith('/')) {
        importLocatorsPath = './' + importLocatorsPath;
      }
      imports += `import { ${locatorVarName} } from '${importLocatorsPath}';\n`;
    }

    if (imports) {
      fileContent = imports + fileContent;
    }

    fileContent = cleanUnusedCode(fileContent);
    fileContent = optimizeAndCleanImports(fileContent);

    fs.writeFileSync(filePath, fileContent, 'utf-8');
  }
}

export function applyPatchDiff(originalCode: string, patchDiff: string): string {
  if (!patchDiff || patchDiff.trim().toLowerCase() === 'none') {
    return originalCode;
  }

  const originalLines = originalCode.split('\n');
  const patchLines = patchDiff.split('\n').map(l => l.trimEnd());
  const linesToRemove = new Set<string>();

  for (const line of patchLines) {
    if (line.startsWith('-')) {
      const content = line.substring(1).trim();
      if (content) {
        linesToRemove.add(content);
      }
    }
  }

  if (linesToRemove.size === 0) {
    return originalCode;
  }

  const resultLines: string[] = [];
  for (const line of originalLines) {
    const trimmed = line.trim();
    if (linesToRemove.has(trimmed)) {
      continue;
    }
    resultLines.push(line);
  }

  return resultLines.join('\n');
}