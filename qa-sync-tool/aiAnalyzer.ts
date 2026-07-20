import { GoogleGenerativeAI } from '@google/generative-ai';
import * as dotenv from 'dotenv';
import * as fs from 'fs';
import * as path from 'path';
import { Requirement, ParsedTest, resolveLocatorsPathForFile, parseTestFileStructure, normalizeRequirementId, extractAllRequirementIds, findConfigFilePath, findLocatorsFilePath, scanAllProjectFiles, extractExecutableAtomics, findTestScripts } from './testScanner';
import { CodeGenerationEngine } from './codeGenerator';

dotenv.config();

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
  // Structured analysis fields from per-requirement block output
  requirementText?: string;
  implementationSummary?: string;
  differencesFound?: string;
  impactAnalysis?: string;
  codeChangesRequired?: 'Yes' | 'No';
  reason?: string;
  // Minimal-patch fields extracted from PROPOSED_CODE by the parser
  patchDiff?: string;   // unified diff block (PATCH_DIFF section)
  whyNeeded?: string;   // one-sentence reason (WHY_NEEDED section)
  // Option 2 fields
  proposedCodeOpt2?: string;
  patchDiffOpt2?: string;
  whyNeededOpt2?: string;
  classification?: string;
  detectedChanges?: string;
  deleteFile?: boolean;
}

// Stable mapping between previous requirement IDs and current requirement IDs for the active run
export const prevIdToCurrIdMap = new Map<string, string>();
export const currIdToPrevIdMap = new Map<string, string>();
// Stable mapping between previous requirement fingerprints and current requirement fingerprints for the active run
export const prevFingerprintToCurrFingerprintMap = new Map<string, string>();
export const currFingerprintToPrevFingerprintMap = new Map<string, string>();
export interface RequirementDiff {
  id: string;
  classification: 'ADD' | 'MODIFY' | 'REVIEW' | 'IN_SYNC';
  changedFields: {
    field: 'Title' | 'Goal' | 'Requirement' | 'Expected Result';
    oldVal: string;
    newVal: string;
  }[];
}

export function isWordForWordIdentical(s1?: string, s2?: string): boolean {
  const getWords = (s?: string) =>
    (s || '')
      .replace(/\\'/g, "'")
      .replace(/\\"/g, '"')
      .replace(/\\`/g, '`')
      .split(/\s+/)
      .map(w => w.trim())
      .filter(Boolean);

  const words1 = getWords(s1);
  const words2 = getWords(s2);

  if (words1.length !== words2.length) return false;
  for (let i = 0; i < words1.length; i++) {
    if (words1[i] !== words2[i]) return false;
  }
  return true;
}

export function compareRequirementDocuments(
  prevReqs: Requirement[],
  currReqs: Requirement[]
): Map<string, RequirementDiff> {
  prevIdToCurrIdMap.clear();
  currIdToPrevIdMap.clear();
  prevFingerprintToCurrFingerprintMap.clear();
  currFingerprintToPrevFingerprintMap.clear();
  const diffs = new Map<string, RequirementDiff>();

  // ── Normalize whitespace/line-breaks/tabs so formatting-only changes are ignored ──
  const norm = (s?: string): string =>
    (s || '')
      .replace(/\\'/g, "'")
      .replace(/\\"/g, '"')
      .replace(/\\`/g, '`')
      .replace(/\s+/g, ' ')
      .trim();

  const normLower = (s?: string): string => norm(s).toLowerCase();

  const cleanTitle = (title: string): string => {
    return normLower(title).replace(/^(fr|ts|tc|r)-?\d+\s*[:\-]?\s*/, '');
  };

  const combinedText = (r: Requirement): string =>
    [r.title, r.goal || '', r.description, r.expectedResult || ''].join(' \n ');

  const prevUnmatched = new Set<number>(prevReqs.map((_, i) => i));
  const currUnmatched = new Set<number>(currReqs.map((_, i) => i));
  const matches: { prevIdx: number; currIdx: number }[] = [];

  // ── PASS 1 — Exact content fingerprint match ───────────────────────────────────
  for (const pIdx of Array.from(prevUnmatched)) {
    const prev = prevReqs[pIdx];
    if (!prev.fingerprint) continue;

    for (const cIdx of Array.from(currUnmatched)) {
      const curr = currReqs[cIdx];
      if (curr.fingerprint === prev.fingerprint) {
        matches.push({ prevIdx: pIdx, currIdx: cIdx });
        prevUnmatched.delete(pIdx);
        currUnmatched.delete(cIdx);
        break;
      }
    }
  }

  // ── PASS 2 — Exact clean title match (ignoring ID prefix) ────────────────────
  for (const pIdx of Array.from(prevUnmatched)) {
    const prev = prevReqs[pIdx];
    const prevClean = cleanTitle(prev.title);
    if (!prevClean) continue;

    for (const cIdx of Array.from(currUnmatched)) {
      const curr = currReqs[cIdx];
      if (cleanTitle(curr.title) === prevClean) {
        matches.push({ prevIdx: pIdx, currIdx: cIdx });
        prevUnmatched.delete(pIdx);
        currUnmatched.delete(cIdx);
        break;
      }
    }
  }

  // ── PASS 3 — Weighted fuzzy similarity, greedy one-to-one assignment ────────
  interface Candidate { pIdx: number; cIdx: number; score: number; }
  const candidates: Candidate[] = [];

  for (const pIdx of Array.from(prevUnmatched)) {
    const prev = prevReqs[pIdx];
    for (const cIdx of Array.from(currUnmatched)) {
      const curr = currReqs[cIdx];

      const prevClean = cleanTitle(prev.title);
      const currClean = cleanTitle(curr.title);

      const titleSim = getJaccardSimilarity(prevClean, currClean);
      const reqSim = getJaccardSimilarity(prev.description, curr.description);
      const goalSim = getJaccardSimilarity(prev.goal || '', curr.goal || '');
      const expSim = getJaccardSimilarity(prev.expectedResult || '', curr.expectedResult || '');
      const overallSim = getJaccardSimilarity(combinedText(prev), combinedText(curr));

      const score =
        titleSim * 0.40 +
        reqSim * 0.35 +
        goalSim * 0.10 +
        expSim * 0.10 +
        overallSim * 0.05;

      if (score >= 0.35) {
        candidates.push({ pIdx, cIdx, score });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  for (const cand of candidates) {
    if (!prevUnmatched.has(cand.pIdx) || !currUnmatched.has(cand.cIdx)) continue;
    matches.push({ prevIdx: cand.pIdx, currIdx: cand.cIdx });
    prevUnmatched.delete(cand.pIdx);
    currUnmatched.delete(cand.cIdx);
  }

  // ── STORE MATCHED REQUIREMENT FINGERPRINT & ID MAPPING ────────────────────
  for (const m of matches) {
    const prev = prevReqs[m.prevIdx];
    const curr = currReqs[m.currIdx];
    prevFingerprintToCurrFingerprintMap.set(prev.fingerprint, curr.fingerprint);
    currFingerprintToPrevFingerprintMap.set(curr.fingerprint, prev.fingerprint);
    // Legacy ID support
    prevIdToCurrIdMap.set(normalizeRequirementId(prev.id), normalizeRequirementId(curr.id));
    currIdToPrevIdMap.set(normalizeRequirementId(curr.id), normalizeRequirementId(prev.id));
  }

  // ── Matched pairs → field-level diff (whitespace-normalized) ───────────────
  for (const m of matches) {
    const prev = prevReqs[m.prevIdx];
    const curr = currReqs[m.currIdx];

    const changedFields: {
      field: 'Title' | 'Goal' | 'Requirement' | 'Expected Result';
      oldVal: string;
      newVal: string;
    }[] = [];

    if (!isWordForWordIdentical(prev.title, curr.title)) {
      changedFields.push({ field: 'Title', oldVal: prev.title, newVal: curr.title });
    }
    if (!isWordForWordIdentical(prev.goal, curr.goal)) {
      changedFields.push({ field: 'Goal', oldVal: prev.goal || '', newVal: curr.goal || '' });
    }
    if (!isWordForWordIdentical(prev.description, curr.description)) {
      changedFields.push({ field: 'Requirement', oldVal: prev.description, newVal: curr.description });
    }
    if (!isWordForWordIdentical(prev.expectedResult, curr.expectedResult)) {
      changedFields.push({ field: 'Expected Result', oldVal: prev.expectedResult || '', newVal: curr.expectedResult || '' });
    }

    const key = curr.fingerprint;

    if (changedFields.length > 0) {
      const onlyDocChanged = changedFields.every(
        f => f.field === 'Title' || f.field === 'Goal' || f.field === 'Expected Result'
      );
      const diffVal: RequirementDiff = {
        id: curr.id,
        classification: onlyDocChanged ? 'REVIEW' : 'MODIFY',
        changedFields
      };
      diffs.set(key, diffVal);
      diffs.set(normalizeRequirementId(curr.id), diffVal);
    } else {
      const diffVal: RequirementDiff = {
        id: curr.id,
        classification: 'IN_SYNC',
        changedFields: []
      };
      diffs.set(key, diffVal);
      diffs.set(normalizeRequirementId(curr.id), diffVal);
    }
  }

  // ── Unmatched current requirements → ADD ────────────────────────────────────
  for (const cIdx of currUnmatched) {
    const curr = currReqs[cIdx];
    const diffVal: RequirementDiff = {
      id: curr.id,
      classification: 'ADD',
      changedFields: [
        { field: 'Requirement', oldVal: '', newVal: curr.description }
      ]
    };
    diffs.set(curr.fingerprint, diffVal);
    diffs.set(normalizeRequirementId(curr.id), diffVal);
  }



  return diffs;
}

/**
 * ORPHAN REQUIREMENT CLEANUP (deterministic, cache-independent).
 *
 * Scans the CURRENT test files' embedded requirement IDs (requirementBlocks) against
 * the CURRENT active Requirement Document. Any ID still referenced in test code that
 * is no longer an active requirement = orphan.
 *
 * Cache-independent by design: does NOT rely on .qa-sync-cache.json (prev vs curr diff),
 * because that cache is overwritten every run regardless of whether the user applied a
 * suggestion. Using live test content instead means a deleted requirement's orphan code
 * stays flagged on every subsequent run until it is ACTUALLY removed from disk.
 *
 * - Test dedicated solely to deleted requirement(s) → suggest removing the entire test.
 * - Test also covers active requirements → suggest removing only the orphaned block(s),
 *   preserving the rest (handled naturally by applyChanges' block-level REMOVE logic).
 */
export function detectDeletedRequirementOrphans(
  requirements: Requirement[],
  prevRequirements: Requirement[],
  tests: ParsedTest[]
): Suggestion[] {
  const suggestions: Suggestion[] = [];

  // Helper to identify unique tests
  const testKey = (t: ParsedTest) => `${t.filePath}::${t.title}`;

  // Find all active tests from the final requirement-to-test mapping
  const activeTests = new Set<string>();
  for (const [reqId, test] of finalRequirementTestMapping.entries()) {
    if (test) {
      activeTests.add(testKey(test));
    }
  }

  const isBlockActive = (bid: string, test: ParsedTest): boolean => {
    const normId = normalizeRequirementId(bid);

    // 1. Resolve old ID → new ID via the cache diff map
    const currId = prevIdToCurrIdMap.get(normId) || normId;
    const req = requirements.find(r => normalizeRequirementId(r.id) === currId);
    if (req) {
      const matchedTest = finalRequirementTestMapping.get(req.fingerprint);
      if (matchedTest && matchedTest.filePath === test.filePath && matchedTest.title === test.title) {
        return true;
      }
    }

    // 2. Fallback: find any active requirement whose mapped test has the same title as this test.
    // This handles the case where the document switched numbering schemes (e.g. FR-1 → I)
    // and the cache ID map doesn't exist yet (first run with a new document).
    for (const [fingerprint, mappedTest] of finalRequirementTestMapping.entries()) {
      if (mappedTest && mappedTest.title === test.title && mappedTest.filePath === test.filePath) {
        return true;
      }
    }

    return false;
  };

  const isFileImportedByOthers = (filePath: string): boolean => {
    const baseName = path.basename(filePath, '.spec.ts');
    const allFiles = Array.from(new Set(tests.map(t => t.filePath)));
    for (const f of allFiles) {
      if (f === filePath) continue;
      try {
        if (fs.existsSync(f)) {
          const content = fs.readFileSync(f, 'utf-8');
          if (content.includes(baseName)) {
            return true;
          }
        }
      } catch {
        // ignore
      }
    }
    return false;
  };

  for (const test of tests) {
    const isTestActive = activeTests.has(testKey(test));

    if (!isTestActive) {
      // Unmapped Test -> Orphan Test
      const allTestsInFile = tests.filter(t => t.filePath === test.filePath);
      const activeTestsInFile = allTestsInFile.filter(t => activeTests.has(testKey(t)));
      const isShared = isFileImportedByOthers(test.filePath);

      if (activeTestsInFile.length === 0 && !isShared) {
        let originalContent = '';
        try {
          if (fs.existsSync(test.filePath)) {
            originalContent = fs.readFileSync(test.filePath, 'utf-8');
          }
        } catch {}

        suggestions.push({
          requirementId: test.requirementId || 'UNMAPPED',
          action: 'REMOVE',
          title: `Delete empty spec file: ${path.basename(test.filePath)}`,
          testTitle: test.title,
          implementationSummary: `"${test.title}" in ${path.basename(test.filePath)}`,
          description:
            `• Affected file: ${path.basename(test.filePath)}\n` +
            `• Reason: This file contains no active tests mapping to requirements. The entire spec file will be deleted.`,
          filePath: test.filePath,
          originalCode: originalContent,
          deleteFile: true,
          classification: 'ORPHAN',
          codeChangesRequired: 'Yes'
        });
      } else {
        suggestions.push({
          requirementId: test.requirementId || 'UNMAPPED',
          action: 'REMOVE',
          title: `Orphan test: ${test.title}`,
          testTitle: test.title,
          implementationSummary: `"${test.title}" in ${path.basename(test.filePath)}`,
          description:
            `• Affected file: ${path.basename(test.filePath)}\n` +
            `• Test: "${test.title}"\n` +
            `• Reason: This test does not map to any active requirement in the Requirement Document. The entire test case will be removed.`,
          filePath: test.filePath,
          startLine: test.startLine,
          endLine: test.endLine,
          originalCode: test.fullText,
          deleteFile: false,
          classification: 'ORPHAN',
          codeChangesRequired: 'Yes'
        });
      }
    } else if (test.requirementBlocks && test.requirementBlocks.length > 0) {
      // Mapped Test -> Check for individual orphan blocks
      const orphanIds = Array.from(new Set(
        test.requirementBlocks
          .map(b => normalizeRequirementId(b.id))
          .filter(id => id && !isBlockActive(id, test))
      ));

      if (orphanIds.length > 0) {
        for (const orphanId of orphanIds) {
          const deletedBlocks = test.requirementBlocks.filter(b => normalizeRequirementId(b.id) === orphanId);
          const originalReqId = deletedBlocks[0]?.id || orphanId;
          const codeToRemove = deletedBlocks.map(b => b.code).join('\n');

          suggestions.push({
            requirementId: originalReqId,
            action: 'REMOVE',
            title: `Orphan requirement block in: ${test.title}`,
            testTitle: test.title,
            implementationSummary: `"${test.title}" in ${path.basename(test.filePath)}`,
            description:
              `• Orphan requirement: ${originalReqId}\n` +
              `• Affected file: ${path.basename(test.filePath)}\n` +
              `• Test: "${test.title}" (still covers other active requirements — only this block will be removed)\n` +
              `• Reason: Requirement ${originalReqId} does not map to any active requirement in the Requirement Document.\n` +
              `• Code to remove (partial block only):\n${codeToRemove}`,
            filePath: test.filePath,
            startLine: test.startLine,
            endLine: test.endLine,
            originalCode: codeToRemove,
            classification: 'ORPHAN',
            codeChangesRequired: 'Yes'
          });
        }
      }
    }
  }

  return suggestions;
}

export function formatDetectedChanges(diff: RequirementDiff): string {
  if (diff.classification === 'IN_SYNC') return 'None';

  const lines: string[] = [];

  for (const field of diff.changedFields) {
    lines.push(`Section: ${field.field}`);
    lines.push(`Old:\n${field.oldVal || 'None'}`);
    lines.push(`New:\n${field.newVal || 'None'}\n`);
  }

  return lines.join('\n').trim();
}



function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"']/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function getWordSet(text: string): Set<string> {
  return new Set(text.split(' ').filter(w => w.length > 1));
}

function getJaccardSimilarity(textA: string, textB: string): number {
  const setA = getWordSet(textA);
  const setB = getWordSet(textB);
  if (setA.size === 0 || setB.size === 0) return 0;
  const intersection = new Set([...setA].filter(x => setB.has(x)));
  const union = new Set([...setA, ...setB]);
  return intersection.size / union.size;
}

function extractComments(testCode: string): string[] {
  const comments: string[] = [];
  const lines = testCode.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('//')) {
      const clean = trimmed.replace(/^\/\/+\s*/, '').trim();
      if (clean && clean.length > 5) comments.push(clean);
    }
  }
  const blockCommentRegex = /\/\*([\s\S]*?)\*\//g;
  let match;
  while ((match = blockCommentRegex.exec(testCode)) !== null) {
    const inner = match[1];
    const innerLines = inner.split(/\r?\n/).map(l => l.replace(/^\s*\*\s*/, '').trim()).filter(l => l.length > 5);
    comments.push(...innerLines);
  }
  return comments;
}

function checkForLocalChanges(
  requirements: Requirement[],
  tests: ParsedTest[],
  missingLocators: { key: string; file: string }[]
): boolean {
  if (missingLocators && missingLocators.length > 0) {
    return true;
  }

  const activeReqIds = new Set(requirements.map(r => normalizeRequirementId(r.id)));
  for (const t of tests) {
    for (const rid of t.requirementIds) {
      if (rid && !activeReqIds.has(normalizeRequirementId(rid))) {
        return true;
      }
    }
    if (t.requirementBlocks) {
      for (const b of t.requirementBlocks) {
        if (b.id && !activeReqIds.has(normalizeRequirementId(b.id))) {
          return true;
        }
      }
    }
  }

  const matchedTests = new Set<ParsedTest>();
  for (const req of requirements) {
    const matched = findMatchedTest(req, tests);
    if (matched) {
      matchedTests.add(matched);
    }
  }
  for (const t of tests) {
    if (!matchedTests.has(t)) {
      return true;
    }
  }

  for (const req of requirements) {
    const matched = findMatchedTest(req, tests);
    if (!matched) {
      return true;
    }

    const atomics = extractExecutableAtomics(req);
    if (atomics.length === 0) {
      continue;
    }

    const executableLines = matched.fullText.split('\n').filter(l => {
      const trimmed = l.trim();
      if (trimmed.startsWith('//') || trimmed.startsWith('*') || trimmed.startsWith('/*')) return false;
      return PLAYWRIGHT_EXEC_PATTERN.test(trimmed);
    });

    for (const atomicText of atomics) {
      const terms = extractAtomicKeyTerms(atomicText);
      if (terms.length < 2) {
        continue;
      }

      let covered = false;
      const INTERACTION_KEYWORDS = ['select', 'click', 'tap', 'choose', 'press', 'submit', 'enter', 'fill', 'type', 'navigate', 'open', 'locate'];
      const PLAYWRIGHT_ACTIONS = ['.click', '.fill', '.press', '.goto', '.selectOption', '.hover', '.check', '.uncheck'];

      const OUTCOME_KEYWORDS = ['display', 'show', 'visible', 'contain', 'verify', 'assert', 'should', 'shall', 'must', 'reflect', 'confirm', 'exist', 'be'];
      const PLAYWRIGHT_ASSERTIONS = ['expect(', 'tobevisible', 'tohaveurl', 'tohavetext', 'tocontaintext', 'tobeenabled', 'tohavevalue', 'tobechecked', 'tohavecount'];

      for (const line of executableLines) {
        const lineLower = line.toLowerCase();
        let hits = terms.filter(t => lineLower.includes(t)).length;

        const hasInteractionVerb = terms.some(t => INTERACTION_KEYWORDS.includes(t));
        const hasPlaywrightAction = PLAYWRIGHT_ACTIONS.some(a => lineLower.includes(a));
        if (hasInteractionVerb && hasPlaywrightAction) {
          hits += 1;
        }

        const hasOutcomeVerb = terms.some(t => OUTCOME_KEYWORDS.includes(t));
        const hasPlaywrightAssertion = PLAYWRIGHT_ASSERTIONS.some(a => lineLower.includes(a));
        if (hasOutcomeVerb && hasPlaywrightAssertion) {
          hits += 1;
        }

        if (hits >= 2) {
          covered = true;
          break;
        }
      }

      if (!covered) {
        return true;
      }
    }

    const comments = extractComments(matched.fullText);
    const normComments = comments.map(c => ({
      original: c,
      normalized: normalizeText(c)
    }));

    for (const atomicText of atomics) {
      const normAtomic = normalizeText(atomicText);
      let bestScore = 0;
      let matchedCommentNormalized = '';

      for (let i = 0; i < normComments.length; i++) {
        const score = getJaccardSimilarity(normAtomic, normComments[i].normalized);
        if (score > bestScore) {
          bestScore = score;
          matchedCommentNormalized = normComments[i].normalized;
        }
      }

      if (bestScore >= 0.35 && normAtomic !== matchedCommentNormalized) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Compares requirements with tests and generates suggestions.
 * Requires GEMINI_API_KEY to be set — throws if it is missing. There is no
 * static-matching fallback; AI is mandatory for the reasoning/code-generation step.
 */
export async function analyzeCoverage(
  requirements: Requirement[],
  tests: ParsedTest[],
  testsFolderExists: boolean,
  rawRdContent?: string,
  projectRoot: string = process.cwd(),
  missingLocators: { key: string; file: string }[] = [],
  rdChanged: boolean = false
): Promise<Suggestion[]> {
  const geminiKey = process.env.GEMINI_API_KEY;

  if (!geminiKey || geminiKey.trim() === '') {
    throw new Error("No active AI provider API key (GEMINI_API_KEY) was found in your environment/.env file. Please configure GEMINI_API_KEY to perform coverage analysis.");
  }

  finalRequirementTestMapping = buildFinalRequirementTestMapping(requirements, tests);

  // Load previous snapshot from cache to perform local comparison
  const cachePath = path.join(projectRoot, '.qa-sync-cache.json');
  let prevRequirements: Requirement[] = [];
  if (fs.existsSync(cachePath)) {
    try {
      prevRequirements = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    } catch (e) {
      // ignore
    }
  }

  const reqDiffs = compareRequirementDocuments(prevRequirements, requirements);
  let rdActuallyChanged = false;
  for (const diff of reqDiffs.values()) {
    if (diff.classification !== 'IN_SYNC') {
      rdActuallyChanged = true;
      break;
    }
  }

  // Deterministic, cache-independent orphan cleanup — must run every time.
  const orphanRequirementCleanup = detectDeletedRequirementOrphans(requirements, prevRequirements, tests);

  // Pre-check if any changes are present. Skip Gemini completely if everything is in sync.
  if (!rdActuallyChanged && !checkForLocalChanges(requirements, tests, missingLocators) && orphanRequirementCleanup.length === 0) {
    console.log(`  ✓ No changes detected. Skipping Gemini AI analysis...`);
    const suggestions: Suggestion[] = [];
    for (const req of requirements) {
      const matched = findMatchedTest(req, tests)!;
      suggestions.push({
        requirementId: req.id,
        action: 'NONE',
        title: req.title,
        description: [
          `• Requirement:\n  ${req.id}: ${req.title}`,
          `• Implementation:\n  All atomics covered.`,
          `• Differences: No differences detected.`,
          `• Impact Analysis:\n  None`,
          `• Recommendation: IN_SYNC  |  Code Changes Required: No`,
          `• Reason:\n  Requirement is fully covered and in sync.`,
        ].join('\n'),
        filePath: matched.filePath,
        testTitle: matched.title,
        originalCode: matched.fullText,
        startLine: matched.startLine,
        endLine: matched.endLine,
        classification: 'IN_SYNC',
        codeChangesRequired: 'No'
      });
    }
    validateFinalSuggestions(requirements, suggestions);
    return suggestions;
  }

  const isVerbose = process.env.QA_SYNC_VERBOSE === 'true';

  if (isVerbose) {
    // 1. Log Parsed Requirements
    console.log(`\n======================================================`);
    console.log(`⚙️ DEBUG: REQUIREMENT ANALYSIS ENGINE VALIDATION LAYER`);
    console.log(`======================================================`);
    console.log(`\n📋 Parsed Requirements (${requirements.length}):`);
    requirements.forEach(r => console.log(`   - [${r.id}] ${r.title}`));

    // 2. Log Parsed Test Coverage
    console.log(`\n🧪 Parsed Test Coverage (${tests.length} tests found):`);
    tests.forEach(t => {
      console.log(`   - Test: "${t.title}" in ${path.basename(t.filePath)}`);
      console.log(`     Explicit mapped requirement IDs: ${t.requirementIds.join(', ') || 'none'}`);
    });

    console.log(`\n  → Using Gemini AI for analysis...`);
  }
  const rawSuggestions = await analyzeWithGemini(geminiKey, requirements, tests, rawRdContent, projectRoot, missingLocators, rdChanged);
  const atomicCoverageReport = buildAtomicCoverageReport(requirements, tests);

  const validatedSuggestions: Suggestion[] = [...orphanRequirementCleanup];

  for (const sug of rawSuggestions) {
    const req = requirements.find(r => normalizeRequirementId(r.id) === normalizeRequirementId(sug.requirementId));
    if (!req) {
      // Skip deleted requirement suggestions from Gemini. Scanned orphan checks handle code removal.
      continue;
    }

    // ── Find matched test using the single source of truth mapping ────
    let matchedTest = findMatchedTest(req, tests);

    const clone = { ...sug };

    // Force active requirements to NOT be ORPHAN
    if (clone.classification === 'ORPHAN') {
      if (matchedTest) {
        const diff = reqDiffs.get(req.fingerprint);
        clone.classification = diff ? diff.classification : 'IN_SYNC';
        clone.action = (clone.classification === 'MODIFY') ? 'MODIFY' : 'NONE';
      } else {
        clone.classification = 'ADD';
        clone.action = 'ADD';
      }
    }

    // Validate / Override classification based on local atomic coverage
    const localCoverage = atomicCoverageReport.results.find((r: any) => normalizeRequirementId(r.reqId) === normalizeRequirementId(req.id));
    const diff = reqDiffs.get(req.fingerprint);
    const hasDocChanges = diff && diff.classification !== 'IN_SYNC';

    if (localCoverage && localCoverage.missingCount === 0 && matchedTest && !hasDocChanges) {
      if (clone.action === 'MODIFY' || clone.action === 'ADD') {
        if (isVerbose) {
          console.log(`   ⚙️ Local override: [${req.id}] has complete executable coverage. Overriding action to NONE.`);
        }
        clone.action = 'NONE';
        clone.classification = 'IN_SYNC';
        clone.codeChangesRequired = 'No';
        clone.proposedCode = undefined;
        clone.patchDiff = undefined;
      }
    }

    if (clone.action === 'NONE') {
      if (!matchedTest) {
        console.log(`   ⚠️ Validation FAILED for [${req.id}]: classified as IN_SYNC but no test covers this requirement. Overriding to ADD.`);
        clone.action = 'ADD';
        clone.codeChangesRequired = 'Yes';
        clone.differencesFound = `Validation override: No matching Playwright test maps to requirement ${req.id}. A new test must be created.`;
        clone.filePath = guessFilePath(req.id, req.title, tests.map(t => t.filePath));
      } else {
        clone.filePath = matchedTest.filePath;
        clone.testTitle = matchedTest.title;
        clone.originalCode = matchedTest.fullText;
        clone.startLine = matchedTest.startLine;
        clone.endLine = matchedTest.endLine;
      }
    } else {
      if (matchedTest) {
        clone.filePath = matchedTest.filePath;
        clone.testTitle = matchedTest.title;
        clone.originalCode = matchedTest.fullText;
        clone.startLine = matchedTest.startLine;
        clone.endLine = matchedTest.endLine;
      } else if (clone.action === 'MODIFY') {
        clone.action = 'ADD';
        clone.filePath = guessFilePath(req.id, req.title, tests.map(t => t.filePath));
      } else if (clone.action === 'ADD' && !clone.filePath) {
        clone.filePath = guessFilePath(req.id, req.title, tests.map(t => t.filePath));
      }
    }

    validatedSuggestions.push(clone);
  }

  // ── Post-processing: strictly filter out REMOVE suggestions for any test that is matched/referenced ──
  const matchedTestKeys = new Set<string>();
  const testKey = (filePath: string, title: string) => `${path.resolve(filePath).toLowerCase()}::${title.trim().toLowerCase()}`;

  // 1. Check finalRequirementTestMapping for matched tests
  for (const [fingerprint, test] of finalRequirementTestMapping.entries()) {
    if (test) {
      matchedTestKeys.add(testKey(test.filePath, test.title));
    }
  }

  // 2. Check final suggestions (excluding REMOVE suggestions themselves)
  for (const sug of validatedSuggestions) {
    if (sug.action !== 'REMOVE' && sug.filePath && sug.testTitle) {
      matchedTestKeys.add(testKey(sug.filePath, sug.testTitle));
    }
  }

  // Deduplicate final suggestions and apply the orphan filter
  const uniqueSuggestions: Suggestion[] = [];
  const seenKeys = new Set<string>();
  for (const sug of validatedSuggestions) {
    const normId = normalizeRequirementId(sug.requirementId);

    // If it's a REMOVE suggestion, verify the target test is not in matchedTestKeys
    if (sug.action === 'REMOVE' && sug.filePath && sug.testTitle) {
      const key = testKey(sug.filePath, sug.testTitle);
      if (matchedTestKeys.has(key)) {
        if (isVerbose) {
          console.log(`   ⚙️ Filtering out REMOVE suggestion for test "${sug.testTitle}" in ${path.basename(sug.filePath)}: it is matched to an active requirement.`);
        }
        continue;
      }
    }

    const key = sug.classification === 'ORPHAN'
      ? `${normId}::orphan::${sug.filePath}::${sug.testTitle}`
      : normId;
    if (!seenKeys.has(key)) {
      seenKeys.add(key);
      uniqueSuggestions.push(sug);
    }
  }

  if (isVerbose) {
    console.log(`\n🔒 Final Validated Result:`);
    uniqueSuggestions.forEach(sug => {
      console.log(`   - [${sug.requirementId}] Title: "${sug.title}" → Action: ${sug.action}`);
    });
    console.log(`======================================================\n`);
  }

  validateFinalSuggestions(requirements, uniqueSuggestions);

  return uniqueSuggestions;
}

// ─────────────────────────────────────────────────────────────────────────────
// ATOMIC-LEVEL COVERAGE ENGINE
// Completely stateless — recomputed fresh on every invocation.
// Uses RD document structure (not verb heuristics) to identify executable atomics.
// Coverage is scoped to the matched test only — not the entire project.
// ─────────────────────────────────────────────────────────────────────────────

interface AtomicCoverageEntry {
  label: string;       // [A1], [A2] …
  text: string;        // The atomic statement text
  covered: boolean;
  evidence?: string;   // Exact Playwright line providing coverage
}

interface RequirementAtomicCoverage {
  reqId: string;
  reqTitle: string;
  matchedTestTitle?: string;
  matchedTestFile?: string;
  atomics: AtomicCoverageEntry[];
  missingCount: number;
}

export interface RequirementChangeReport {
  reqId: string;
  reqTitle: string;
  changeType: 'ADDED_REQUIREMENT' | 'DELETED_REQUIREMENT' | 'MODIFIED' | 'UNCHANGED';
  changedLines: { type: 'added' | 'removed' | 'changed'; text: string; before?: string }[];
}

// Playwright executable line patterns
const PLAYWRIGHT_EXEC_PATTERN = /expect\s*\(|toBeVisible|toHaveURL|toHaveText|toHaveValue|toBeEnabled|toBeChecked|toHaveCount|toContainText|toHaveAttribute|toHaveTitle|\.click\s*\(|\.fill\s*\(|\.hover\s*\(|\.select|\.check\s*\(|\.uncheck\s*\(|\.dragTo\s*\(|locator\s*\(/i;

// Stop-words for key-term extraction (used for behavioral matching)
const ATOMIC_STOP_WORDS = new Set([
  'shall', 'must', 'will', 'that', 'this', 'with', 'from', 'when', 'then',
  'have', 'been', 'into', 'upon', 'each', 'over', 'also', 'both', 'more',
  'user', 'users', 'page', 'should', 'would', 'could', 'which', 'where',
  'able', 'make', 'need', 'only', 'such', 'some', 'does', 'were',
  'their', 'there', 'these', 'those', 'about', 'after', 'before',
  'during', 'while', 'until', 'every'
]);

function extractAtomicKeyTerms(sentence: string): string[] {
  const words = sentence
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length >= 4 && !ATOMIC_STOP_WORDS.has(w));
  return [...new Set(words)].slice(0, 8);
}

/**
 * Finds the matched test for a requirement.
 * Priority 1: Explicit requirement tag match (FR-XX in test requirementIds)
 * Priority 2: Title fuzzy match (≥2 significant words overlap)
 * Returns the matched test or null.
 */
function getTitleWords(title: string): string[] {
  const STOP_WORDS = new Set(['to', 'the', 'and', 'of', 'for', 'in', 'on', 'at', 'a', 'an', 'is', 'are', 'with', 'this', 'that', 'it', 'from', 'by', 'as', 'or']);
  return title.toLowerCase()
    .replace(/[^a-z0-9\s]/g, ' ')
    .split(/\s+/)
    .map(w => w.trim())
    .filter(w => w.length > 0 && !STOP_WORDS.has(w));
}

export class MappingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MappingError';
  }
}

export let finalRequirementTestMapping = new Map<string, ParsedTest | null>();

export function buildFinalRequirementTestMapping(
  requirements: Requirement[],
  tests: ParsedTest[]
): Map<string, ParsedTest | null> {
  const mapping = new Map<string, ParsedTest | null>();

  const norm = (s?: string): string =>
    (s || '')
      .replace(/\\'/g, "'")
      .replace(/\\"/g, '"')
      .replace(/\\`/g, '`')
      .replace(/\s+/g, ' ')
      .trim();

  const normLower = (s?: string): string => norm(s).toLowerCase();

  const cleanTitle = (title: string): string => {
    return normLower(title).replace(/^(fr|ts|tc|r)-?\d+\s*:\s*/, '');
  };

  const combinedReqText = (r: Requirement): string =>
    [r.title, r.goal || '', r.description, r.expectedResult || ''].join(' \n ');

  const reqUnmatched = new Set<number>(requirements.map((_, i) => i));
  const testUnmatched = new Set<number>(tests.map((_, i) => i));
  const matches: { reqIdx: number; testIdx: number }[] = [];

  // ── PASS 1 — Exact normalized title match (ignoring ID prefix) ──────────────
  for (const rIdx of Array.from(reqUnmatched)) {
    const req = requirements[rIdx];
    const reqCleanTitle = cleanTitle(req.title);
    if (!reqCleanTitle) continue;

    for (const tIdx of Array.from(testUnmatched)) {
      const test = tests[tIdx];
      if (cleanTitle(test.title) === reqCleanTitle) {
        matches.push({ reqIdx: rIdx, testIdx: tIdx });
        reqUnmatched.delete(rIdx);
        testUnmatched.delete(tIdx);
        break;
      }
    }
  }

  // ── PASS 2 — Weighted fuzzy similarity match ────────────────────────────────
  interface Candidate { reqIdx: number; testIdx: number; score: number; }
  const candidates: Candidate[] = [];

  for (const rIdx of Array.from(reqUnmatched)) {
    const req = requirements[rIdx];
    const reqWords = getTitleWords(req.title);
    const reqClean = cleanTitle(req.title);

    for (const tIdx of Array.from(testUnmatched)) {
      const test = tests[tIdx];
      const testClean = cleanTitle(test.title);

      const titleSim = getJaccardSimilarity(reqClean, testClean);

      const testWords = getTitleWords(test.title);
      const overlapWords = reqWords.filter(rw => testWords.includes(rw)).length;
      const wordSim = testWords.length > 0 ? overlapWords / Math.max(reqWords.length, testWords.length) : 0;

      const contentSim = getJaccardSimilarity(combinedReqText(req), test.fullText);

      const score = titleSim * 0.40 + wordSim * 0.40 + contentSim * 0.20;

      if (score >= 0.25) {
        candidates.push({ reqIdx: rIdx, testIdx: tIdx, score });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);
  for (const cand of candidates) {
    if (!reqUnmatched.has(cand.reqIdx) || !testUnmatched.has(cand.testIdx)) continue;
    matches.push({ reqIdx: cand.reqIdx, testIdx: cand.testIdx });
    reqUnmatched.delete(cand.reqIdx);
    testUnmatched.delete(cand.testIdx);
  }

  // ── PASS 3 — Content-based and phrase-based mapping for remaining unmatched items ──
  const contentCandidates: Candidate[] = [];
  for (const rIdx of Array.from(reqUnmatched)) {
    const req = requirements[rIdx];
    const reqText = combinedReqText(req);

    for (const tIdx of Array.from(testUnmatched)) {
      const test = tests[tIdx];
      const contentSim = getJaccardSimilarity(reqText, test.fullText);

      let phraseMatchCount = 0;
      const cleanDesc = req.description.toLowerCase();
      const cleanTest = test.fullText.toLowerCase();

      const sentences = cleanDesc.split(/[.\n]+/).map(s => s.trim()).filter(s => s.length > 10);
      for (const sentence of sentences) {
        if (cleanTest.includes(sentence)) {
          phraseMatchCount++;
        }
      }

      const phraseScore = sentences.length > 0 ? (phraseMatchCount / sentences.length) : 0;
      const finalContentScore = contentSim * 0.60 + phraseScore * 0.40;

      if (finalContentScore >= 0.15) {
        contentCandidates.push({ reqIdx: rIdx, testIdx: tIdx, score: finalContentScore });
      }
    }
  }

  contentCandidates.sort((a, b) => b.score - a.score);
  for (const cand of contentCandidates) {
    if (!reqUnmatched.has(cand.reqIdx) || !testUnmatched.has(cand.testIdx)) continue;
    matches.push({ reqIdx: cand.reqIdx, testIdx: cand.testIdx });
    reqUnmatched.delete(cand.reqIdx);
    testUnmatched.delete(cand.testIdx);
  }

  // Populate the final mapping
  for (const req of requirements) {
    mapping.set(req.fingerprint, null);
  }
  for (const m of matches) {
    const req = requirements[m.reqIdx];
    const test = tests[m.testIdx];
    mapping.set(req.fingerprint, test);
  }

  // Validate every mapping to ensure logical relation
  for (const [reqFingerprint, test] of mapping.entries()) {
    if (!test) continue;
    const req = requirements.find(r => r.fingerprint === reqFingerprint)!;

    const reqClean = cleanTitle(req.title);
    const testClean = cleanTitle(test.title);
    const titleSim = getJaccardSimilarity(reqClean, testClean);

    const reqWords = getTitleWords(req.title);
    const testWords = getTitleWords(test.title);
    const overlapWords = reqWords.filter(rw => testWords.includes(rw)).length;
    const wordSim = testWords.length > 0 ? overlapWords / Math.max(reqWords.length, testWords.length) : 0;

    const contentSim = getJaccardSimilarity(combinedReqText(req), test.fullText);
    const score = titleSim * 0.40 + wordSim * 0.40 + contentSim * 0.20;

    // Check if the test contains matched sentences or keywords
    const cleanDesc = req.description.toLowerCase();
    const cleanTest = test.fullText.toLowerCase();
    const sentences = cleanDesc.split(/[.\n]+/).map(s => s.trim()).filter(s => s.length > 10);
    let phraseMatch = false;
    for (const sentence of sentences) {
      if (cleanTest.includes(sentence)) {
        phraseMatch = true;
        break;
      }
    }

    const isLogical = (score >= 0.18) || (contentSim >= 0.18) || phraseMatch || (reqClean === testClean);

    // Reject mapping if not logically related
    if (!isLogical) {
      throw new MappingError(
        `Mapping validation failed: Requirement "${req.id}: ${req.title}" was matched to Test "${test.title}" in ${path.basename(test.filePath)}, but they are not logically related (similarity score: ${score.toFixed(2)}, content score: ${contentSim.toFixed(2)}).`
      );
    }
  }

  return mapping;
}

function findMatchedTest(req: Requirement, tests: ParsedTest[]): ParsedTest | null {
  return finalRequirementTestMapping.get(req.fingerprint) || null;
}

export function validateFinalSuggestions(
  requirements: Requirement[],
  suggestions: Suggestion[]
): void {
  const activeIds = new Set(requirements.map(r => normalizeRequirementId(r.id)));
  const deletedIds = new Set<string>();



  // Rule: Every active requirement appears only once (excluding ORPHAN suggestions).
  const activeSeen = new Set<string>();
  for (const sug of suggestions) {
    if (sug.classification && sug.classification !== 'ORPHAN') {
      const normId = normalizeRequirementId(sug.requirementId);
      if (activeIds.has(normId)) {
        if (activeSeen.has(normId)) {
          throw new MappingError(
            `Validation failed: Active requirement "${sug.requirementId}" appears multiple times in the suggestions.`
          );
        }
        activeSeen.add(normId);
      }
    }
  }
}

/**
 * Phase 0 — Requirement Change Detection.
 * Compares every requirement's current text against what was previously mapped
 * in the test suite (via requirement tags and test titles).
 * Detects: Added requirements, Deleted requirements (orphan tests), Modified lines.
 * Completely stateless — only uses current RD and current test files.
 */

/**
 * Phase 1 — Atomic Coverage Pre-Analysis.
 * For each requirement:
 *   1. Calls extractExecutableAtomics() (structure-based, not verb-based)
 *   2. Finds the matched test (scoped to that test only — not entire project)
 *   3. For each atomic, checks if the matched test has an EXECUTABLE Playwright line
 *      (actual actions/assertions — not comments or test titles)
 *      that corresponds to it
 * Returns SECTION A — injected into AI prompt.
 */
function buildAtomicCoverageReport(
  requirements: Requirement[],
  tests: ParsedTest[]
): { sectionText: string; results: RequirementAtomicCoverage[] } {
  const results: RequirementAtomicCoverage[] = [];

  for (const req of requirements) {
    const matched = findMatchedTest(req, tests);
    const atomics = extractExecutableAtomics(req);

    // Get only the EXECUTABLE lines from the matched test (not comments, not titles)
    const executableLines = matched
      ? matched.fullText.split('\n').filter(l => {
        const trimmed = l.trim();
        // Skip comments
        if (trimmed.startsWith('//') || trimmed.startsWith('*') || trimmed.startsWith('/*')) return false;
        // Must match a Playwright executable pattern
        return PLAYWRIGHT_EXEC_PATTERN.test(trimmed);
      })
      : [];

    const atomicEntries: AtomicCoverageEntry[] = atomics.map((text, idx) => {
      const terms = extractAtomicKeyTerms(text);

      if (terms.length < 2) {
        return {
          label: `[A${idx + 1}]`,
          text,
          covered: true,
          evidence: '(statement too short for independent keyword match — assumed covered)'
        };
      }

      let evidence: string | undefined;
      const INTERACTION_KEYWORDS = ['select', 'click', 'tap', 'choose', 'press', 'submit', 'enter', 'fill', 'type', 'navigate', 'open', 'locate'];
      const PLAYWRIGHT_ACTIONS = ['.click', '.fill', '.press', '.goto', '.selectOption', '.hover', '.check', '.uncheck'];

      const OUTCOME_KEYWORDS = ['display', 'show', 'visible', 'contain', 'verify', 'assert', 'should', 'shall', 'must', 'reflect', 'confirm', 'exist', 'be'];
      const PLAYWRIGHT_ASSERTIONS = ['expect(', 'tobevisible', 'tohaveurl', 'tohavetext', 'tocontaintext', 'tobeenabled', 'tohavevalue', 'tobechecked', 'tohavecount'];

      for (const line of executableLines) {
        const lineLower = line.toLowerCase();
        let hits = terms.filter(t => lineLower.includes(t)).length;

        // Check for interaction requirement match
        const hasInteractionVerb = terms.some(t => INTERACTION_KEYWORDS.includes(t));
        const hasPlaywrightAction = PLAYWRIGHT_ACTIONS.some(a => lineLower.includes(a));
        if (hasInteractionVerb && hasPlaywrightAction) {
          hits += 1;
        }

        // Check for outcome requirement match
        const hasOutcomeVerb = terms.some(t => OUTCOME_KEYWORDS.includes(t));
        const hasPlaywrightAssertion = PLAYWRIGHT_ASSERTIONS.some(a => lineLower.includes(a));
        if (hasOutcomeVerb && hasPlaywrightAssertion) {
          hits += 1;
        }

        // Require at least 2 matching key terms
        if (hits >= 2) {
          evidence = line.trim();
          break;
        }
      }

      return {
        label: `[A${idx + 1}]`,
        text,
        covered: !!evidence,
        evidence
      };
    });

    const missingCount = atomicEntries.filter(a => !a.covered).length;
    results.push({
      reqId: req.id,
      reqTitle: req.title,
      matchedTestTitle: matched?.title,
      matchedTestFile: matched ? path.basename(matched.filePath) : undefined,
      atomics: atomicEntries,
      missingCount
    });
  }

  // Build SECTION A text for prompt injection
  const sectionLines: string[] = [
    '════════════════════════════════════════════════════════════════',
    'SECTION A — ATOMIC COVERAGE PRE-ANALYSIS (stateless, current run only)',
    '════════════════════════════════════════════════════════════════',
    'Source: executable atomics extracted from RD structure only (body content).',
    'Titles, Goals, Purpose, Notes, Expected Result headers, ALL-CAPS headings are EXCLUDED.',
    'Coverage: checked against matched test\'s EXECUTABLE lines only (not entire project).',
    'Executable lines = actual Playwright actions/assertions (NOT comments or test titles).',
    '',
    '━━━ MANDATORY RULES ━━━',
    '1. For each requirement, SECTION A shows the matched test and its atomic coverage.',
    '2. Every [Ax] marked ✗ MISSING MUST be listed in your ATOMIC CHECKS output.',
    '3. NEVER classify as IN_SYNC if any [Ax] is ✗ MISSING.',
    '4. Wording change only (behavior unchanged) → REVIEW. Behavior changed → MODIFY.',
    '5. No matched test at all → ADD.',
    '',
  ];

  for (const r of results) {
    sectionLines.push(`[${r.reqId}] ${r.reqTitle}`);
    if (r.matchedTestTitle) {
      sectionLines.push(`  Matched test: "${r.matchedTestTitle}" in ${r.matchedTestFile}`);
    } else {
      sectionLines.push(`  Matched test: NONE — no test covers this requirement → classify as ADD`);
    }

    if (r.atomics.length === 0) {
      sectionLines.push(`  Executable atomics: NONE extracted (requirement body may be structural only)`);
    } else {
      for (const a of r.atomics) {
        if (a.covered) {
          sectionLines.push(`  ${a.label} ✓ COVERED   → "${a.text}"`);
          sectionLines.push(`           Evidence: ${a.evidence}`);
        } else {
          sectionLines.push(`  ${a.label} ✗ MISSING   → "${a.text}"`);
          sectionLines.push(`           No executable Playwright line found in matched test.`);
        }
      }
    }

    if (r.missingCount === 0 && r.matchedTestTitle) {
      sectionLines.push(`  ↳ All atomics covered. Verify exact wording alignment still holds.`);
    } else if (r.missingCount > 0) {
      sectionLines.push(`  ↳ ${r.missingCount} atomic(s) MISSING → classify as MODIFY (if test exists) or ADD (if no test).`);
    }
    sectionLines.push('');
  }

  return { sectionText: sectionLines.join('\n') + '\n', results };
}

/**
 * Shared prompt builder using the user-specified prompt design.
 */
export function buildUserAnalysisPrompt(
  requirements: Requirement[],
  tests: ParsedTest[],
  rawRdContent?: string,
  projectRoot: string = process.cwd(),
  missingLocators: { key: string; file: string }[] = [],
  rdChanged: boolean = false,
  runNonce: string = Date.now().toString(36) + '-' + process.pid.toString(36)
): string {
  let configContent = '';
  try {
    const configPath = findConfigFilePath(projectRoot);
    if (configPath && fs.existsSync(configPath)) {
      configContent = fs.readFileSync(configPath, 'utf-8');
    }
  } catch (e) { }

  let tsConfigContent = '';
  try {
    const tsConfigPath = path.join(projectRoot, 'tsconfig.json');
    if (fs.existsSync(tsConfigPath)) {
      tsConfigContent = fs.readFileSync(tsConfigPath, 'utf-8');
    }
  } catch (e) { }

  let packageJsonContent = '';
  try {
    const packageJsonPath = path.join(projectRoot, 'package.json');
    if (fs.existsSync(packageJsonPath)) {
      packageJsonContent = fs.readFileSync(packageJsonPath, 'utf-8');
    }
  } catch (e) { }

  let projectHelpersSection = '';
  try {
    const helpersDir = path.join(projectRoot, 'tests', 'helpers');
    if (fs.existsSync(helpersDir)) {
      const getFilesRecursively = (dir: string): string[] => {
        let results: string[] = [];
        const list = fs.readdirSync(dir);
        for (const file of list) {
          const fullPath = path.join(dir, file);
          const stat = fs.statSync(fullPath);
          if (stat && stat.isDirectory()) {
            results = results.concat(getFilesRecursively(fullPath));
          } else if (file.endsWith('.ts') || file.endsWith('.js')) {
            results.push(fullPath);
          }
        }
        return results;
      };
      const helperFiles = getFilesRecursively(helpersDir);
      for (const hf of helperFiles) {
        const content = fs.readFileSync(hf, 'utf-8');
        projectHelpersSection += `// File: tests/helpers/${path.relative(helpersDir, hf).replace(/\\/g, '/')}\n${content}\n\n`;
      }
    }
  } catch (e) { }

  let appUrl = 'http://localhost:3000';
  const urlMatch = configContent.match(/baseURL:\s*(['"`])(.*?)\1/);
  if (urlMatch) {
    appUrl = urlMatch[2];
  }

  const specFilesList = Array.from(new Set(tests.map(t => t.filePath)));
  const requirementsText = requirements.map(r => {
    const matchedTest = findMatchedTest(r, tests);
    let scopeStr = '';
    let targetFileStr = '';
    
    if (matchedTest) {
      scopeStr = 'SCOPE C (inline patch for existing test)';
      targetFileStr = path.basename(matchedTest.filePath);
    } else {
      const guessedPath = guessFilePath(r.id, r.title, specFilesList);
      const fileExists = fs.existsSync(guessedPath);
      if (fileExists) {
        scopeStr = 'SCOPE B (new test() block in existing file)';
        targetFileStr = path.basename(guessedPath);
      } else {
        scopeStr = 'SCOPE A (new .spec.ts file)';
        targetFileStr = path.basename(guessedPath);
      }
    }

    return `ID: ${r.id}\n` +
      `Title: ${r.title}\n` +
      `Goal: ${r.goal || 'None'}\n` +
      `Requirement: ${r.description}\n` +
      `Expected Result: ${r.expectedResult || 'None'}\n` +
      `MANDATORY TARGET FILE: ${targetFileStr}\n` +
      `MANDATORY GENERATION SCOPE: ${scopeStr}`;
  }).join('\n\n');

  // Identify deleted requirements and orphan test cases
  const activeReqIds = new Set(requirements.map(r => normalizeRequirementId(r.id)));
  const testReqIds = new Set<string>();
  for (const t of tests) {
    if (t.requirementIds) {
      for (const id of t.requirementIds) {
        if (id) testReqIds.add(normalizeRequirementId(id));
      }
    }
    if (t.requirementBlocks) {
      for (const b of t.requirementBlocks) {
        if (b.id) testReqIds.add(normalizeRequirementId(b.id));
      }
    }
  }
  const deletedReqIds: string[] = [];
  for (const id of testReqIds) {
    const norm = normalizeRequirementId(id);
    if (!activeReqIds.has(norm)) {
      deletedReqIds.push(id);
    }
  }

  const matchedTests = new Set<ParsedTest>();
  for (const req of requirements) {
    const matched = findMatchedTest(req, tests);
    if (matched) {
      matchedTests.add(matched);
    }
  }

  const orphanTests: ParsedTest[] = [];
  for (const t of tests) {
    if (!matchedTests.has(t)) {
      orphanTests.push(t);
    }
  }

  let deletedSection = '';
  const deletedLines: string[] = [];

  if (deletedReqIds.length > 0) {
    deletedLines.push(`DELETED REQUIREMENT IDS (These requirement IDs are present in the test code/comments but are completely missing from the Requirement Document):`);
    deletedReqIds.forEach(id => deletedLines.push(`- ${id}`));
  }

  if (orphanTests.length > 0) {
    deletedLines.push(`\nDELETED/ORPHAN TEST CASES (These test cases exist in the codebase but do not correspond to any active requirement in the document):`);
    orphanTests.forEach(t => deletedLines.push(`- Test Case: "${t.title}" in ${path.basename(t.filePath)}`));
  }

  if (deletedLines.length > 0) {
    deletedSection = deletedLines.join('\n') + `\n\n` +
      `CRITICAL: You MUST output a report block for each deleted requirement ID and deleted/orphan test case listed above. Use the requirement ID or the test case title as the block header (e.g. --- Social Media Navigation ---). Classify them as DELETED. Set ACTION to REMOVE. Do NOT generate proposed code. Suggest removing the test block.\n\n`;
  }

  const testNames = tests.map(t => t.title);
  const testsSummary = testNames.length > 0
    ? testNames.map(name => `- ${name}`).join('\n')
    : "No tests exist yet.";

  // Compile specCode from existing files (always fresh from disk)
  let specCode = '';
  const specFiles = Array.from(new Set(tests.map(t => t.filePath)));
  for (const sf of specFiles) {
    if (fs.existsSync(sf)) {
      specCode += `// File: ${path.basename(sf)}\n` + fs.readFileSync(sf, 'utf-8') + '\n\n';
    }
  }
  if (!specCode.trim()) {
    specCode = 'No tests exist yet.';
  }

  // ── Phase 1: Atomic Coverage Pre-Analysis ─────────────────────────────────────────
  // Uses structure-based extractExecutableAtomics (not verb heuristics).
  // Coverage scoped to matched test only — not entire project.
  // Completely stateless — no cache, fresh on every run.
  const atomicCoverageReport = tests.length > 0
    ? buildAtomicCoverageReport(requirements, tests)
    : { sectionText: '', results: [] };
  const sectionA = atomicCoverageReport.sectionText;



  // ── Build a test→requirement mapping table ────────────────────────────────
  // When tests have no embedded requirement IDs (e.g. after IDs were removed from
  // test titles/comments), we compute the match by keyword overlap between each
  // test title and each requirement's ID+title. This mapping is injected into the
  // prompt so the AI has an explicit link table to work from.
  const testsHaveNoIds = tests.every(t => t.requirementIds.length === 0);
  let testReqMappingSection = '';
  if (testsHaveNoIds && tests.length > 0 && requirements.length > 0) {
    const mappingLines: string[] = [];
    const usedReqIds = new Set<string>();

    for (const t of tests) {
      const testWords = getTitleWords(t.title);

      let bestReqId = '';
      let bestScore = 0;
      let minScore = 2;

      for (const req of requirements) {
        const reqWords = getTitleWords(req.title);
        const score = reqWords.filter(rw => testWords.includes(rw)).length;
        const potentialMin = Math.min(reqWords.length, testWords.length, 2);
        if (score > bestScore && score >= potentialMin) {
          bestScore = score;
          bestReqId = req.id;
          minScore = potentialMin;
        }
      }

      if (bestReqId && bestScore >= minScore) {
        mappingLines.push(`- Test: "${t.title}"  →  Candidate Requirement: ${bestReqId}`);
        // Attach the mapped ID to the parsed test so the rest of the pipeline can use it.
        if (!t.requirementIds.includes(bestReqId)) {
          t.requirementIds.push(bestReqId);
          if (!t.requirementId) t.requirementId = bestReqId;
        }
        usedReqIds.add(bestReqId);
      } else {
        mappingLines.push(`- Test: "${t.title}"  →  Candidate Requirement: UNKNOWN (no confident keyword match)`);
      }
    }

    testReqMappingSection =
      `\nTEST → REQUIREMENT MAPPING (pre-computed by keyword similarity — candidate links only):\n` +
      `CRITICAL WARNING: This mapping shows the CLOSEST test candidate for each requirement.\n` +
      `A candidate mapping does NOT mean the requirement is covered.\n` +
      `You MUST verify executable coverage at the individual assertion level for EVERY requirement.\n` +
      `A requirement is covered ONLY if the test contains a specific expect(), page.locator(), .click(), .fill(), toHaveURL(), toBeVisible(), toHaveText(), or equivalent Playwright action/assertion that directly validates it.\n` +
      `NOTE: Requirement IDs were removed from test titles/comments. Use this mapping only to identify the candidate test — not to infer coverage.\n` +
      mappingLines.join('\n') + '\n';
  }

  // Compile locators as ragText context
  let locatorsSection = '';
  const uniqueLocators = new Map<string, { varName: string; fileName: string; content: string }>();
  for (const t of tests) {
    if (t.filePath) {
      try {
        const resolved = resolveLocatorsPathForFile(t.filePath, projectRoot);
        if (resolved && fs.existsSync(resolved.path)) {
          if (!uniqueLocators.has(resolved.path)) {
            const content = fs.readFileSync(resolved.path, 'utf-8');
            uniqueLocators.set(resolved.path, {
              varName: resolved.varName,
              fileName: resolved.fileName,
              content
            });
          }
        }
      } catch (e) { }
    }
  }
  if (uniqueLocators.size === 0) {
    try {
      const locatorPath = findLocatorsFilePath(projectRoot);
      if (locatorPath && fs.existsSync(locatorPath)) {
        locatorsSection = fs.readFileSync(locatorPath, 'utf-8');
      }
    } catch (e) { }
  } else {
    for (const locInfo of uniqueLocators.values()) {
      locatorsSection += `// File: ${locInfo.fileName} (var: ${locInfo.varName})\n${locInfo.content}\n\n`;
    }
  }
  const ragText = locatorsSection || 'No additional codebase context.';

  // Build list of ALL existing locator keys for the AI to validate against
  let allLocatorKeys = '';
  for (const locInfo of uniqueLocators.values()) {
    const flatKeys: string[] = [];
    const parentRegex = /(\w+)\s*:\s*\{/g;
    let parentMatch;
    while ((parentMatch = parentRegex.exec(locInfo.content)) !== null) {
      const parent = parentMatch[1];
      const blockStart = parentMatch.index + parentMatch[0].length;
      let braceDepth = 1;
      let i = blockStart;
      while (i < locInfo.content.length && braceDepth > 0) {
        if (locInfo.content[i] === '{') braceDepth++;
        else if (locInfo.content[i] === '}') braceDepth--;
        i++;
      }
      const blockContent = locInfo.content.substring(blockStart, i - 1);
      let childMatch;
      const childRe = /(\w+)\s*:/g;
      while ((childMatch = childRe.exec(blockContent)) !== null) {
        flatKeys.push(`${parent}.${childMatch[1]}`);
      }
    }
    if (flatKeys.length > 0) {
      allLocatorKeys += `// ${locInfo.fileName}\n${flatKeys.join('\n')}\n\n`;
    }
  }

  let emptyTodosList = '';
  for (const t of tests) {
    if (isTestBodyEmptyOrTodo(t.fullText)) {
      emptyTodosList += `- ${t.title} (covers requirement: ${t.requirementIds.join(', ')})\n`;
    }
  }

  // Build existing spec file names list for AI to use as file routing hints
  const existingSpecFileNames = specFiles.map(f => path.basename(f)).join(', ');

  // Project-wide files context for cross-file impact analysis (used when RD changes)
  let projectFilesContext = '';
  if (rdChanged) {
    const projectFiles = scanAllProjectFiles(projectRoot);
    projectFilesContext = "\nREQUIREMENT DOCUMENT CHANGE DETECTED — CROSS-FILE IMPACT CONTEXT:\n" +
      "The Requirements Document has changed since the last run. Analyze ALL the following project files for impact:\n" +
      projectFiles.map(f => `// File: ${path.relative(projectRoot, f.filePath)}\n${f.content.substring(0, 800)}...`).join('\n\n') + '\n';
  }

  let missingLocatorsSection = '';
  if (missingLocators && missingLocators.length > 0) {
    missingLocatorsSection = "\nCRITICAL: The following locators are referenced in the Playwright test code but are missing from the locators file. You MUST generate unique, accurate, and robust CSS/XPath selectors for them. ONLY use CSS class names that actually appear in the CODEBASE CONTEXT above — NEVER invent class names. List them in the LOCATOR_UPDATES section:\n" +
      missingLocators.map(m => `- ${m.key} (used in ${path.basename(m.file)})`).join('\n') + '\n';
  }

  return (
    // ── PER-RUN NONCE ────────────────────────────────────────────────────────────────
    // This nonce changes on every invocation of npm start.
    // It is injected as the very first token in the prompt so that the Gemini API
    // server-side response cache (which caches by prompt hash) is always invalidated.
    // Without this, temperature=0 causes Gemini to return the same cached response
    // even when the spec file content has changed.
    `RUN_ID: ${runNonce} | TIMESTAMP: ${new Date().toISOString()}\n` +
    `INSTRUCTION: Analyze the current project state as of this run only. ` +
    `Do not reuse any previous analysis result. The run ID above changes on every invocation.\n\n` +
    // ───────────────────────────────────────────────────────────────────
    "You are a senior QA engineer syncing Playwright test scripts with a requirements document.\n\n" +

    // ─────────────────────────────────────────────────────────────────
    // SECTION A: CONSISTENCY & ANTI-HALLUCINATION
    // ─────────────────────────────────────────────────────────────────
    "════════════════════════════════════════════════════════════════\n" +
    "SECTION A — STATELESS ANALYSIS RULES (CRITICAL)\n" +
    "════════════════════════════════════════════════════════════════\n" +
    "EVERY RUN IS A COMPLETELY FRESH ANALYSIS. There is NO state between runs.\n" +
    "\n" +
    "MANDATORY: You MUST read and analyze the EXACT CURRENT CONTENT of every file provided\n" +
    "in this prompt. The source of truth is strictly defined as follows:\n" +
    "  - Source of Truth for Requirements: ONLY the REQUIREMENTS DOCUMENT text in this prompt.\n" +
    "  - Source of Truth for current coverage: ONLY the Playwright EXISTING TEST CODE text in this prompt.\n" +
    "  - The ADDITIONAL CONTEXT (locators, config) are strictly RAG/supporting context to help you generate correct code, not to define requirements or prove coverage.\n" +
    "\n" +
    "FORBIDDEN — The following are explicitly banned in every run:\n" +
    "  ✗ Reusing results, mappings, or classifications from any previous run\n" +
    "  ✗ Assuming a requirement is covered because it was covered in a previous run\n" +
    "  ✗ Treating a previously approved suggestion as permanently applied\n" +
    "  ✗ Reporting IN_SYNC for a requirement without finding the specific assertion in the CURRENT test code\n" +
    "  ✗ Using test titles or file names as evidence of coverage — only executable code counts\n" +
    "  ✗ Inferring that an assertion exists because a test exists for the same feature\n" +
    "\n" +
    "IF AN ASSERTION WAS REMOVED: If a Playwright assertion, action, or locator was previously\n" +
    "present but has since been removed from the test file, you MUST detect it as missing and\n" +
    "classify the requirement as MODIFY — exactly as if the test had never been written.\n" +
    "The approval of a suggestion in a previous run does NOT mean the requirement is permanently covered.\n" +
    "Coverage is determined solely by what is in the CURRENT test code in this prompt.\n" +
    "\n" +
    "ANTI-HALLUCINATION: Never invent suggestions. Never restore deleted requirements.\n" +
    "Never generate code for requirements not in the document.\n\n" +

    // ─────────────────────────────────────────────────────────────────
    // SECTION B: DUPLICATE PREVENTION
    // ─────────────────────────────────────────────────────────────────
    "════════════════════════════════════════════════════════════════\n" +
    "SECTION B — DUPLICATE REPORT PREVENTION\n" +
    "════════════════════════════════════════════════════════════════\n" +
    "- Each Requirement ID (FR-XX / TS-XX / TC-XX) must appear ONLY ONCE in your final response.\n" +
    "- Never report the same requirement more than once.\n" +
    "- Never print the same reason or code diff multiple times.\n" +
    "- Show exactly one Requirement ID, one Action, one Reason, one Code Diff, and one Suggested Update per requirement.\n" +
    "- Merge any duplicate observations for the same requirement into a single clear statement.\n\n" +

    // ─────────────────────────────────────────────────────────────────
    // SECTION C: PROVIDED CONTEXT (project data injected at runtime)
    // A live snapshot fingerprint is injected first so that the model
    // always reads the current file state and cannot reuse a cached
    // response from a previous run in which the spec content differed.
    // ─────────────────────────────────────────────────────────────────
    "════════════════════════════════════════════════════════════════\n" +
    "SECTION C — PROVIDED PROJECT CONTEXT\n" +
    "════════════════════════════════════════════════════════════════\n" +
    "MANDATORY FIRST STEP — CATALOGUE ALL PROJECT ASSETS:\n" +
    "Before analyzing any single requirement, you MUST read and internally catalogue:\n" +
    "  1. IMPORTS: Every import statement found in EXISTING TEST CODE and locator files.\n" +
    "  2. HELPERS: Every helper function or utility defined at module scope (e.g. dismissMembershipPopup, loginUser).\n" +
    "  3. FIXTURES: Every test.beforeEach, test.afterEach, test.beforeAll, test.afterAll, test.use(...), and test.extend(...) block. Also catalogue module-scope const fixture variables.\n" +
    "  4. LOCATOR GROUPS: Every parent object and child key from ALL EXISTING LOCATOR KEYS section.\n" +
    "  5. TEST TITLES: Every test('...') and test.describe('...') title already present in the spec code.\n" +
    "  6. CONFIG KEYS: Every property found in CONFIG FILE CONTENT.\n" +
    "  7. TSCONFIG CONFIGURATION: compilerOptions (target, lib, strict, module type).\n" +
    "  8. PACKAGE DEPENDENCIES: available dependencies and devDependencies in package.json.\n" +
    "  9. PROJECT HELPERS & UTILITIES: custom helper/utility files found in the tests/helpers/ directory.\n" +
    "DO NOT generate any code that references an item not in this catalogue unless it is standard in the project configuration or generated in updates.\n" +
    "Never assume browser globals (e.g. window, document, alert) or DOM APIs exist. Never assume non-existent external libraries exist.\n" +
    "If an equivalent helper function or utility is available in the catalogue (e.g., dismissMembershipPopup in tests/helpers/dismissPopup.ts), reuse it instead of generating a new approach.\n" +
    "Adapt generated code to the project configuration (tsconfig, package.json dependencies, target language options, and project coding patterns) rather than expecting the project to adapt to the generated code.\n" +
    "If an item is missing, generate it in LOCATOR_UPDATES / ENV_UPDATES / CONFIG_UPDATES FIRST, then reference it in PROPOSED_CODE.\n\n" +
    `CURRENT_RUN_SNAPSHOT (read this — it changes every time a spec file is modified):\n${(() => {
      const specFiles2 = Array.from(new Set(tests.map(t => t.filePath)));
      const lines: string[] = [];
      for (const sf of specFiles2) {
        try {
          const raw = fs.readFileSync(sf, 'utf-8');
          const lineCount = raw.split('\n').length;
          // djb2 hash of file content
          let h = 5381;
          for (let i = 0; i < raw.length; i++) { h = ((h << 5) + h) + raw.charCodeAt(i); h = h & h; }
          lines.push(`  ${path.basename(sf)}: ${lineCount} lines, hash=${Math.abs(h).toString(16)}`);
        } catch (_) { lines.push(`  ${path.basename(sf)}: unreadable`); }
      }
      if (lines.length === 0) lines.push('  (no spec files found)');
      return lines.join('\n');
    })()}\n\n` +
    `APPLICATION URL: ${appUrl}\n\n` +
    (tsConfigContent ? `PROJECT TSCONFIG:\n${tsConfigContent}\n\n` : '') +
    (packageJsonContent ? `PROJECT PACKAGE.JSON:\n${packageJsonContent}\n\n` : '') +
    (projectHelpersSection ? `PROJECT HELPERS & UTILITIES:\n${projectHelpersSection}\n\n` : '') +
    `CONFIG FILE CONTENT (use only properties/variables defined here):\n${configContent}\n\n` +
    `REQUIREMENTS DOCUMENT:\n${requirementsText}\n\n` +
    `EXISTING TEST NAMES:\n${testsSummary}\n\n` +
    testReqMappingSection +
    deletedSection +
    `EXISTING SPEC FILES IN THIS PROJECT: ${existingSpecFileNames}\n\n` +
    `EXISTING TEST CODE (analyze ONLY what is literally written here — this is the current state of the codebase):\n${specCode}\n\n` +
    `ADDITIONAL CONTEXT FROM CODEBASE (locators, helpers, config):\n${ragText}\n\n` +
    (allLocatorKeys ? `ALL EXISTING LOCATOR KEYS (reference only — NEVER use keys or class names not found here):\n${allLocatorKeys}\n\n` : '') +
    projectFilesContext +
    missingLocatorsSection +
    sectionA +



    // ─────────────────────────────────────────────────────────────────
    // SECTION D: REQUIREMENT ANALYSIS RULES
    // ─────────────────────────────────────────────────────────────────
    "════════════════════════════════════════════════════════════════\n" +
    "SECTION D — CRITICAL REQUIREMENT DOCUMENT ANALYSIS RULES\n" +
    "════════════════════════════════════════════════════════════════\n" +
    "D1. Stage 1 – Requirement Mapping:\n" +
    "    - Map every individual requirement to its closest matching test (use the TEST→REQUIREMENT MAPPING table if available).\n" +
    "    - This mapping is a CANDIDATE link only. It does NOT mean the requirement is covered.\n" +
    "    - Every requirement is an independent functional requirement and must be analyzed on its own.\n" +
    "    - If no test exists for this specific requirement, classify as ADD.\n\n" +
    "D2. Stage 2 – Requirement Atomicity Analysis (MANDATORY — perform before any executable check):\n" +
    "    - Read the full requirement text and decompose it into individual atomic checks.\n" +
    "    - An atomic check is any single, independently verifiable statement, including:\n" +
    "      * A specific UI element that must be visible, enabled, or in a given state\n" +
    "      * A specific user action (click, fill, select, check, hover, drag)\n" +
    "      * A specific expected result (URL, page title, text content, element count, error message)\n" +
    "      * A specific validation (field value, element state, visibility, selection, badge count)\n" +
    "      * A specific interaction (button triggers modal, toggle changes state, form submits)\n" +
    "    - Number each atomic check: [A1], [A2], [A3], ... for this requirement.\n" +
    "    - Each atomic check is an independent verification point. Treat it separately from all others.\n" +
    "    - Do NOT group multiple atomic checks under a single verification unless one Playwright line\n" +
    "      explicitly validates all of them simultaneously.\n" +
    "    - Coverage is complete ONLY if every [A1], [A2], ... [AN] has its own Playwright action or assertion.\n" +
    "    - If any atomic check has no corresponding Playwright line, report it by its label (e.g. '[A3] missing')\n" +
    "      and classify as MODIFY.\n" +
    "    - FORBIDDEN: Using one assertion to satisfy multiple independent atomic checks unless it\n" +
    "      explicitly and demonstrably validates all of them.\n\n" +
    "D3. Stage 3 – Difference Detection (Mandatory, line-by-line):\n" +
    "    - Using the atomic checks from Stage 2, scan the matched test code line-by-line.\n" +
    "    - For each atomic check [A1]...[AN], find the specific Playwright line that covers it.\n" +
    "    - Report a difference for every atomic check that has no corresponding Playwright line.\n" +
    "    - A difference MUST be reported if:\n" +
    "      1. Any atomic check from the requirement is missing or not implemented in the test.\n" +
    "      2. A requirement line was explicitly removed from the RD but is still executed in the test.\n" +
    "    - FORBIDDEN: Reporting 'No differences detected' when:\n" +
    "      * The test has no explicit assertion for the expected result stated in the requirement.\n" +
    "      * The test covers a different variant or edge-case of the same feature but not this specific requirement.\n" +
    "      * The test file covers the same feature but does not contain the specific action/assertion for this requirement.\n" +
    "    - Do NOT report the following as differences (setup/prerequisites — not business requirements):\n" +
    "      * Opening the website / navigating to the base URL\n" +
    "      * Searching for a product\n" +
    "      * Navigating to a page (PDP, Cart, Wishlist, Category)\n" +
    "      * Selecting a product to reach the required page\n" +
    "      * Visual setup, dismissing popups/safeguards, or handling hydration delays\n" +
    "      * Minor non-functional wording variations or casing (e.g., 'Gift card' vs 'Gift Voucher' for the same element)\n" +
    "    - Never ignore real functional differences or infer/assume missing requirements.\n\n" +
    "D4. Stage 4 – Impact Analysis:\n" +
    "    - Analyze whether each detected difference changes the executable Playwright behavior.\n" +
    "    - Clearly state: does the existing test behavior satisfy this requirement or not?\n" +
    "    - If any atomic check [Ax] is absent from the test, the behavior does NOT satisfy the requirement.\n\n" +
    "D5. Stage 5 – Classification:\n" +
    "    - Use only these classification rules:\n" +
    "      * IN_SYNC: EVERY atomic check [A1]…[AN] has a corresponding explicit Playwright line in the test. Minor wording variation, prerequisite steps, or document typos are IN_SYNC.\n" +
    "      * REVIEW: Differences were detected (e.g. extra test steps from previously deleted requirement lines, or wording changes pointing to different elements/flows), but the executable Playwright behavior still satisfies every atomic check. Report every difference. State no code changes are required.\n" +
    "      * MODIFY: (a) One or more atomic checks [Ax] have no corresponding Playwright assertion/action, OR (b) a test exists for the feature area but does NOT contain Playwright logic covering every atomic check of this individual requirement. MODIFY is correct whenever any atomic check is uncovered — even if the overall scenario is related.\n" +
    "      * ADD: This specific requirement has no matching test block at all.\n" +
    "      * DELETED: Test exists but the requirement no longer exists in the document.\n\n" +
    "D6. Stage 6 – Code Generation (MINIMAL PATCH ONLY):\n" +
    "    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
    "    MANDATORY PRE-GENERATION CHECKLIST (run in order before writing any code):\n" +
    "    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
    "    STEP 0 — LOCATOR EXISTENCE CHECK (use CATALOGUE 4 from Phase 0):\n" +
    "      - Consult CATALOGUE 4 built in Phase 0. It lists every parent group and child key in the locator file.\n" +
    "      - If the locator exists in CATALOGUE 4, reference it directly. Do NOT add a LOCATOR_UPDATES entry for it.\n" +
    "      - If it is absent from CATALOGUE 4, it must be generated: add it to LOCATOR_UPDATES first, then reference it in PROPOSED_CODE.\n" +
    "      - CRITICAL: NEVER use inline fallback expressions (e.g. `locator || '.selector'`) or invented properties. Any fallback expression containing `||` will fail validation. All locators must come from CATALOGUE 4 or a new LOCATOR_UPDATES entry.\n" +
    "      - Never reference properties or sub-properties that do not exist in CATALOGUE 4 or LOCATOR_UPDATES.\n" +
    "    STEP 1 — PATCH SCOPE CHECK & VALID SCOPING:\n" +
    "      - Insert code only inside an existing async Playwright test or helper function. Never generate code outside a valid scope.\n" +
    "      - Identify only the SPECIFIC lines of the existing test body that need to change.\n" +
    "      - Do NOT regenerate the entire test function, describe block, imports, or helpers.\n" +
    "      - If only one assertion is missing, output only that one assertion.\n" +
    "    STEP 1a — EXECUTION FLOW ANALYSIS (MANDATORY for all MODIFY / SCOPE C patches):\n" +
    "      Before writing a single line of patch code, you MUST trace the complete execution flow of the\n" +
    "      existing test from top to bottom and build an ordered map of every step:\n" +
    "      FLOW MAP FORMAT (internal — do not output this; use it to determine the insertion point):\n" +
    "        Line N: <action/assertion> → <page/state produced>\n" +
    "        Line N+1: <action/assertion> → <page/state produced>\n" +
    "        ...\n" +
    "      INSERTION POINT RULE (CRITICAL — violation will cause wrong test behavior):\n" +
    "        - For each missing atomic check [Ax], identify the EXACT action already in the test that\n" +
    "          produces the page/state that [Ax] needs to verify.\n" +
    "        - The patch line(s) for [Ax] MUST be inserted IMMEDIATELY AFTER that action — not at the\n" +
    "          end of the test, not before it, not in an arbitrary position.\n" +
    "        - Example: If [Ax] asserts that a product title is visible after clicking the product card,\n" +
    "          find the .click() on the product card in the flow map and insert the assertion on the\n" +
    "          very next line after it.\n" +
    "        - Example: If [Ax] asserts the cart page URL after navigation, find the page.goto('/cart')\n" +
    "          or the cart button click and insert the toHaveURL assertion directly after it.\n" +
    "      FORBIDDEN PLACEMENTS:\n" +
    "        ✗ Appending assertions at the end of the test when the relevant action appears earlier.\n" +
    "        ✗ Inserting before the action that produces the state being asserted.\n" +
    "        ✗ Grouping all missing assertions at a single arbitrary location in the test body.\n" +
    "        ✗ Inserting inside a different test block or outside the matched test scope.\n" +
    "      CONTEXT LINE RULE: The PATCH_DIFF must include 2–3 unchanged lines immediately before and\n" +
    "      after the insertion point so the diff anchors precisely to the correct location in the file.\n" +
    "    STEP 2 — ASSERTION SELECTION:\n" +
    "      - Choose the most precise official Playwright assertion for each atomic check:\n" +
    "          toBeVisible() / toHaveText() / toHaveURL() / toBeEnabled() / toHaveValue()\n" +
    "          toBeChecked() / toHaveCount() / toHaveAttribute() / toContainText()\n" +
    "      - Do NOT invent non-existent Playwright APIs.\n" +
    "      - Ensure every line compiles without TypeScript errors.\n" +
    "    STEP 2a — SCOPE INVENTORY (MANDATORY — build this before writing any patch line):\n" +
    "      Read the ENTIRE matched test file from top to bottom and catalogue:\n" +
    "      INVENTORY A — IMPORTS: every symbol already imported at the top of the file.\n" +
    "      INVENTORY B — DECLARATIONS: every const/let/var name declared in the test body,\n" +
    "        describe block, module scope, loop variables, and catch parameters.\n" +
    "      INVENTORY C — LOCATOR REFERENCES: every locator key expression already used in\n" +
    "        the test body (e.g. theSouledStoreLocators.nav.wishlistIcon).\n" +
    "      INVENTORY D — HELPERS: every helper function already called or defined in the file.\n" +
    "      REUSE RULE (applies to every line of patch code generated after this step):\n" +
    "        - Symbol in INVENTORY A  → import already exists. Do NOT add another import for it.\n" +
    "        - Name in INVENTORY B    → variable already declared. Do NOT use const/let for it.\n" +
    "          Reference the existing variable directly. If you need a different value, use a new name.\n" +
    "        - Key in INVENTORY C     → locator already referenced. Do NOT add it to LOCATOR_UPDATES.\n" +
    "          Reuse the existing reference directly.\n" +
    "        - Helper in INVENTORY D  → already available. Do NOT redefine or reimport it.\n" +
    "      NAME CONFLICT RULE: If the patch needs a new variable whose name matches any entry in\n" +
    "        INVENTORY B, choose a distinct name before writing the patch line.\n" +
    "        FORBIDDEN: Reusing a name from INVENTORY B as a new const/let/var identifier.\n" +
    "    STEP 3 — DUPLICATION CHECK (cross-reference STEP 2a inventories before output):\n" +
    "      - Every import in the patch must NOT be in INVENTORY A. If it is, remove it from the patch.\n" +
    "      - Every const/let/var name in the patch must NOT be in INVENTORY B. If it is, reuse or rename.\n" +
    "      - Every locator key in the patch must NOT already be in INVENTORY C as an existing reference.\n" +
    "      - Every helper call in the patch must NOT be a duplicate of INVENTORY D.\n" +
    "      - No existing assertion from the current test may be re-emitted unchanged in the patch.\n" +
    "    STEP 4 — STYLE CHECK:\n" +
    "      - Match the indentation, spacing, quote style, and naming conventions of the existing file.\n" +
    "    STEP 5 — COMPILE VALIDATION:\n" +
    "      - Before outputting, check every identifier in the patch against INVENTORY B.\n" +
    "        If it is already declared → the patch must USE (not re-declare) it. A second const/let\n" +
    "        for the same name is a TypeScript TS2300 / TS2451 compile error.\n" +
    "      - Check every import in the patch against INVENTORY A. If it is already imported → omit\n" +
    "        the import line from the patch entirely.\n" +
    "      - Check that all other variables referenced are either already in INVENTORY B or declared\n" +
    "        with a fresh name (not in INVENTORY B) within the patch itself.\n" +
    "    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
    "    OUTPUT RULES:\n" +
    "      - Generate code only for MODIFY and ADD.\n" +
    "      - For MODIFY (SCOPE C — inline patch): Output ONLY the minimal missing lines as PATCH_DIFF + raw PROPOSED_CODE. Do NOT include unchanged surrounding code. If two consecutive atomic checks share the same parent locator variable, declare it once and reference it in both.\n" +
    "      - PLACEMENT RULE (MANDATORY for SCOPE C): Each missing assertion MUST be placed at the\n" +
    "        logically correct position identified in STEP 1a — immediately after the action that\n" +
    "        produces the state being verified. NEVER append missing assertions at the end of the\n" +
    "        test by default. If multiple assertions are missing, each one must be placed after its\n" +
    "        own triggering action, preserving the correct semantic execution order.\n" +
    "      - For ADD (SCOPE A — new file):\n" +
    "          COMPLETENESS MANDATE: Before writing a single line, decompose EVERY atomic statement\n" +
    "          of the requirement [A1]..[AN]. The generated file must satisfy ALL of them in one pass.\n" +
    "          Output: imports + test.describe block + module-scope helpers + test() block + every\n" +
    "          assertion and action required by every atomic check.\n" +
    "          Do NOT copy imports or helpers not used by this specific test.\n" +
    "          FORBIDDEN: Generating a stub or skeleton that covers only the first atomic.\n" +
    "          FORBIDDEN: Writing a minimum viable test that will require a MODIFY on the next run.\n" +
    "      - For ADD (SCOPE B — new test block in existing file):\n" +
    "          COMPLETENESS MANDATE: Before writing a single line, decompose EVERY atomic statement\n" +
    "          of the requirement [A1]..[AN]. The test() block must satisfy ALL of them in one pass.\n" +
    "          Output: ONLY the new test('...', async ({ page }) => { ... }) block.\n" +
    "          Do NOT re-emit existing imports, describe wrappers, or helpers already in the file.\n" +
    "          The new block must rely only on already-existing file-level helpers.\n" +
    "          FORBIDDEN: Generating a stub or skeleton that covers only the first atomic.\n" +
    "          FORBIDDEN: Writing a minimum viable test that will require a MODIFY on the next run.\n" +
    "      - ADD SELF-REVIEW (MANDATORY — run after generating ADD code, before returning output):\n" +
    "          For each atomic check [Ax] of the requirement, locate the EXACT Playwright line in the\n" +
    "          generated code that satisfies it. If any [Ax] has no satisfying line — add it now.\n" +
    "          Only return the code after every [Ax] has a named satisfying line.\n" +
    "          A requirement implemented through ADD must not require another MODIFY on the next run\n" +
    "          unless the Requirement Document itself changes.\n" +
    "      - SCOPE DECISION RULE: Use SCOPE A only when no spec file exists for this feature area.\n" +
    "        Use SCOPE B when a spec file exists but has no test block for this requirement.\n" +
    "        Use SCOPE C for all MODIFY cases.\n" +
    "      - If you cannot generate a complete implementation with high confidence, set CLASSIFICATION\n" +
    "        to MODIFY or ADD, explain what is missing in the REASON field, and set PROPOSED_CODE to:\n" +
    "        'LOW_CONFIDENCE — <explain missing info>'.\n" +
    "      - Never generate code for REVIEW or IN_SYNC.\n" +
    "    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" +
    "    UNIFIED DIFF FORMAT (mandatory for all MODIFY patches):\n" +
    "      - Show the change as a unified diff block prefixed with PATCH_DIFF: before the code.\n" +
    "      - Use the standard unified diff format:\n" +
    "          @@ -<old_start>,<old_count> +<new_start>,<new_count> @@\n" +
    "           (context lines with a leading space)\n" +
    "          -<removed line>\n" +
    "          +<added line>\n" +
    "      - Include 2–3 lines of unchanged context above and below each change block. These\n" +
    "        context lines MUST be the lines immediately surrounding the insertion point\n" +
    "        determined in STEP 1a — not arbitrary lines from elsewhere in the test.\n" +
    "      - Then include WHY_NEEDED: <one sentence stating which atomic check [Ax] this satisfies\n" +
    "        AND why this specific line in the test is the correct insertion point based on the\n" +
    "        execution flow (e.g. 'inserted after the .click() that navigates to the PDP because\n" +
    "        [A3] requires verifying the product title is visible on that page').>\n" +
    "      - The full content (compilable patch) goes under PROPOSED_CODE as usual.\n\n" +
    "D7. Critical Guidelines:\n" +
    "    - The Requirement Document is the only source of truth.\n" +
    "    - Never invent, restore, or assume missing requirements.\n" +
    "    - Never justify differences by saying they are implied, necessary, or prerequisites.\n" +
    "    - Always compare first, then reason.\n" +
    "    - Every detected difference must appear in the report before the classification is decided.\n" +
    "    - The final report must clearly separate:\n" +
    "      1. Requirement\n" +
    "      2. Atomic Checks [A1]...[AN]\n" +
    "      3. Current Test Implementation\n" +
    "      4. Differences Detected (per atomic check)\n" +
    "      5. Impact Analysis\n" +
    "      6. Classification\n" +
    "      7. Code Changes Required\n\n" +
    "D8. Stability Safeguards:\n" +
    "    - Do NOT suggest removing or modifying stability safeguards, try-catch blocks, custom loop logic, or helper functions/calls used for dismissing transient elements, popups, or overlays (such as `dismissMembershipPopup`). These are runtime stability measures and do not represent requirement mismatches.\n\n" +
    "D9. Accuracy Rules:\n" +
    "    - Never rely only on semantic similarity.\n" +
    "    - Never assume two requirements are equivalent because they look similar or belong to the same feature.\n" +
    "    - Every suggestion must be traceable: Requirement Change → Existing Code → Suggested Update.\n" +
    "    - If you cannot justify a code change from the REQUIREMENTS DOCUMENT, do not suggest one.\n" +
    "    - Accuracy is more important than the number of suggestions. Missing a real requirement change is an error; suggesting unnecessary code changes is also an error.\n\n" +
    "D10. Individual Executable Coverage Verification (MANDATORY for every requirement):\n" +
    "    - For each atomic check [Ax], you must find the SPECIFIC Playwright line that:\n" +
    "      * Asserts the expected result: expect(...).toBeVisible() / toHaveText() / toHaveURL() / toBeEnabled() / toHaveValue() / etc.\n" +
    "      * Performs the required action: .click() / .fill() / .selectOption() / .check() / .hover() / etc.\n" +
    "      * Validates the state described in the atomic check\n" +
    "    - If no such specific line exists in the test code, that atomic check is NOT covered.\n" +
    "    - In that case, classify as MODIFY and specify EXACTLY which atomic check [Ax] is missing and what action/assertion/locator is required.\n" +
    "    FORBIDDEN justifications for IN_SYNC:\n" +
    "      ✗ 'The test navigates to the same page' → NOT executable coverage\n" +
    "      ✗ 'The test file covers this feature' → NOT executable coverage\n" +
    "      ✗ 'The requirement is implied by another test step' → NOT executable coverage\n" +
    "      ✗ 'A semantically similar requirement is tested' → NOT executable coverage\n" +
    "      ✗ 'The test title or comment mentions this requirement' → NOT executable coverage\n" +
    "      ✗ 'The same feature area is tested' → NOT executable coverage\n" +
    "    REQUIRED for IN_SYNC: Every atomic check [A1]…[AN] must have a named expect() assertion or Playwright action directly validating it in the test body.\n\n" +
    "D11. Newly Added and Modified Requirement Detection:\n" +
    "    - Treat every requirement as a completely independent functional requirement.\n" +
    "    - A requirement is 'newly added' or 'modified' if any of its atomic checks has no matching executable assertion/action — even if a test file exists for the same feature area.\n" +
    "    - Similar meaning does NOT imply coverage. Each atomic check must be verified individually.\n" +
    "    - If a new requirement is related to an existing feature but any atomic check has no specific assertion, classify as:\n" +
    "      * ADD: if no test block at all covers this specific requirement\n" +
    "      * MODIFY: if a test block exists for the feature but lacks atomic check coverage\n" +
    "    - Never merge a newly added requirement with an adjacent requirement under the same feature.\n" +
    "    - Never mark a requirement as IN_SYNC because another similar requirement is covered.\n" +
    "    - Always report each requirement separately with its own atomic checks, classification, and code suggestion.\n\n" +

    "D12. Atomic Coverage Protocol (uses SECTION A — MANDATORY):\n" +
    "    SECTION A (above) contains a pre-computed, stateless atomic coverage report.\n" +
    "    Every executable atomic in every requirement was individually checked against\n" +
    "    ONLY the EXECUTABLE PLAYWRIGHT LINES in the MATCHED TEST (not entire project).\n" +
    "    Non-executable content (Titles, Goals, Purpose, Notes, Expected Result headers,\n" +
    "    ALL-CAPS headings) was EXCLUDED — only requirement body content was analyzed.\n" +
    "\n" +
    "    STEP 1 — Read SECTION A for each requirement before doing your own analysis.\n" +
    "    STEP 2 — Evaluate executable coverage as a whole for each requirement:\n" +
    "      - Distinguish interaction requirements (e.g. click a button, fill a field) from outcome requirements (e.g. page shows title, URL contains text).\n" +
    "      - A Playwright action (click(), fill(), press(), selectOption(), goto(), hover(), check(), uncheck()) satisfies an interaction requirement by itself and does NOT require an additional assertion unless the requirement explicitly asks to verify the outcome.\n" +
    "      - A Playwright assertion (expect(), toBeVisible(), toHaveURL(), toHaveText(), etc.) satisfies an outcome requirement.\n" +
    "      - Do NOT require artificial assertions such as \"toBeClicked()\" or \"toHaveValue()\" when the existing Playwright action already proves the interaction.\n" +
    "      - Multiple Playwright actions and assertions may collectively satisfy a single requirement.\n" +
    "      - Only classify MODIFY when required executable behavior is genuinely missing. Otherwise prefer IN_SYNC.\n" +
    "    STEP 3 — Wording changes (compare RD text vs what the matched test actually does):\n" +
    "      - If wording changed but EXECUTABLE BEHAVIOR is identical (or satisfied by existing test steps) → REVIEW (no code change suggested)\n" +
    "      - If wording change means the test targets incorrect behavior or fails → MODIFY\n" +
    "    STEP 4 — Removed Atomics:\n" +
    "      - If an atomic statement was REMOVED from the Requirement Document, immediately stop analyzing it.\n" +
    "      - FORBIDDEN: Reporting the removed atomic as 'missing'. FORBIDDEN: Generating a patch for it.\n" +
    "      - FORBIDDEN: Classifying as MODIFY because of a removed atomic — evaluate only the REMAINING atomics.\n" +
    "      - If the test still asserts the removed atomic, classify as REVIEW (test has extra coverage, not missing).\n" +
    "    STEP 5 — Cumulative Navigation:\n" +
    "      - Treat executable behavior as cumulative. A click() followed by toHaveURL() satisfies BOTH the navigation AND the display requirement.\n" +
    "      - ALLOWED: Multiple Playwright lines collectively satisfying one atomic check.\n" +
    "      - FORBIDDEN: Requiring a page-specific assertion when toHaveURL() already uniquely identifies the target page.\n" +
    "\n" +
    "    CLASSIFICATION DECISION TABLE (apply after reporting all atomics):\n" +
    "    ┌──────────────────────────────────────────────────────────────────────┬───────────────┐\n" +
    "    │ Situation                                                            │ Classification│\n" +
    "    ├──────────────────────────────────────────────────────────────────────┼───────────────┤\n" +
    "    │ All required executable behavior covered, wording behavior identical  │ IN_SYNC       │\n" +
    "    │ Wording changed but executable behavior remains identical            │ REVIEW        │\n" +
    "    │ Atomic removed from RD, test still asserts it                        │ REVIEW        │\n" +
    "    │ Required functional action/assertion is genuinely missing            │ MODIFY        │\n" +
    "    │ Word changed in RD, test now targets incorrect behavior              │ MODIFY        │\n" +
    "    │ No matched test exists for this requirement at all                   │ ADD           │\n" +
    "    └──────────────────────────────────────────────────────────────────────┴───────────────┘\n\n" +

    "D13. Non-Executable Exclusion (MANDATORY):\n" +
    "    The following are NEVER subject to coverage analysis:\n" +
    "      - The requirement Title itself\n" +
    "      - Lines starting with: Goal:, Goals:, Purpose:, Overview:, Note:, Notes:, Context:\n" +
    "      - Expected Result: / Expected Results: header lines\n" +
    "      - ALL-CAPS section headings\n" +
    "      - Lines shorter than 10 characters\n" +
    "    SECTION A already excludes these. Do NOT flag them as missing coverage.\n" +
    "    Do NOT use the Title or Goal as evidence of executable coverage.\n\n" +




    // ─────────────────────────────────────────────────────────────────
    // SECTION E: EMPTY/STUB TEST WARNING (injected only when applicable)
    // ─────────────────────────────────────────────────────────────────
    (emptyTodosList
      ? "════════════════════════════════════════════════════════════════\n" +
      "SECTION E — EMPTY/TODO STUB TESTS (CRITICAL)\n" +
      "════════════════════════════════════════════════════════════════\n" +
      "The following tests are currently empty TODO stubs with no actual implementation.\n" +
      "You MUST classify them as MODIFY (NOT IN_SYNC) and provide a complete implementation under PROPOSED_CODE\n" +
      "with a complete, fully functional Playwright implementation:\n" +
      emptyTodosList + "\n"
      : "") +

    // ─────────────────────────────────────────────────────────────────
    // SECTION F: PLAYWRIGHT GUARD RAIL RULES
    // ─────────────────────────────────────────────────────────────────
    "════════════════════════════════════════════════════════════════\n" +
    "SECTION F — PLAYWRIGHT GUARD RAIL RULES FOR GENERATED/MODIFIED TESTS\n" +
    "════════════════════════════════════════════════════════════════\n" +
    "F1.  Auto-Waiting:\n" +
    "     - Always rely on Playwright's built-in auto-waiting.\n" +
    "     - Do NOT use page.waitForTimeout() or any hardcoded delays.\n" +
    "     - Use locator-based actions (.click(), .fill()) which automatically wait for actionability.\n\n" +
    "F2.  Locator Strictness:\n" +
    "     - Always use precise, robust locators defined in the codebase context (CSS selectors preferred).\n" +
    "     - Prefer standard CSS selectors over Playwright-specific helpers (e.g. getByRole, getByText).\n" +
    "     - Use .first() or .nth() when multiple elements match to avoid strict mode violations.\n" +
    "     - Avoid ambiguous selectors.\n\n" +
    "F3.  Actionability Checks:\n" +
    "     - Before performing actions, verify elements are visible and enabled when appropriate.\n" +
    "     - Prefer assertions like `await expect(locator).toBeVisible();` and `await expect(locator).toBeEnabled();` to block/wait naturally.\n\n" +
    "F4.  Assertion Guard Rails:\n" +
    "     - Always use Playwright's `expect(...)` assertions for validating state.\n" +
    "     - Prefer auto-retrying, web-first assertions: `await expect(locator).toBeVisible();`, `await expect(page).toHaveURL(/pattern/);`, `await expect(page).toHaveTitle(/pattern/i)`.\n" +
    "     - Do NOT use inline timeouts (e.g. `{ timeout: 15000 }`) inside expect assertions. The global timeout is set in playwright.config.ts.\n\n" +
    "F5.  Navigation Handling:\n" +
    "     - Always navigate using `await page.goto(url, { waitUntil: 'domcontentloaded' });`.\n" +
    "     - Do NOT assume navigation timing manually or add hard waits after navigation.\n\n" +
    "F6.  No Hard Waits:\n" +
    "     - `page.waitForTimeout()` is strictly forbidden in generated or modified test code.\n" +
    "     - Use retryable web-first assertions or `locator.waitFor()` if a specific element transition is needed.\n\n" +
    "F7.  Test Stability:\n" +
    "     - Avoid flaky patterns. Do not depend on timing-based logic. Use deterministic checks only.\n\n" +
    "F8.  Minimal & Safe Actions:\n" +
    "     - Do not perform unnecessary clicks or redundant steps. Only perform steps required by the requirement.\n\n" +
    "F9.  Isolation Awareness:\n" +
    "     - Each test must run independently and not assume or rely on shared state from other tests.\n\n" +
    "F10. Clean Error Handling:\n" +
    "     - For optional elements or transient popups (like dismissing membership popups), use safe try/catch handling or conditional visibility checks.\n" +
    "     - When generating a NEW test file or modifying a test, if you call local helper functions (such as `dismissMembershipPopup`), you MUST include their full implementation at the top of the generated code block (after imports) so the file is self-contained and does not throw ReferenceErrors.\n\n" +
    "F11. Specific Matchers:\n" +
    "     - To assert that a size option or interactive item is selected, check if either the element itself OR its parent element (such as an <li>) has a class matching 'active', 'selected', or 'current' (e.g. `await locator.evaluate(el => { const hasClass = (item: any) => /selected|active|current/i.test(item.className || ''); return hasClass(el) || (el.parentElement ? hasClass(el.parentElement) : false); })`). Do NOT use exact `classList.contains('selected')` as frameworks often use classes like `selectedSize`.\n" +
    "     - To verify page headings or view identifiers robustly, search for headings (h1, h2, [role='heading']) containing the keyword (e.g. `page.locator('h1, h2, [role=\"heading\"]').filter({ hasText: /Deals/i }).first()`) and check visibility, or check if the keyword is present in the page title.\n\n" +
    "F12. Test Structure Rules (MANDATORY):\n" +
    "     - NEVER place `test.setTimeout()` inside a `test(...)` body. Set it at describe level: `test.describe.configure({ timeout: 60000 })` or remove it entirely.\n" +
    "     - ALL new test files MUST wrap test cases inside a `test.describe('Description', () => { ... })` block.\n" +
    "     - Helper functions (like `dismissMembershipPopup`) MUST be defined at module scope (top level), NOT inside describe or test blocks.\n" +
    "     - Generated test imports MUST follow the exact import style of existing spec files (same import order: config → locators → playwright).\n\n" +
    "F13. Auto-Generate Missing Locators (MANDATORY — with deep pre-check):\n" +
    "     STEP 0 — DEEP DEPENDENCY PRE-CHECK (do this BEFORE anything else):\n" +
    "       Before generating any new locator, you MUST exhaustively audit every existing locator to confirm the required element is truly absent. Do the following checks in order:\n" +
    "         CHECK A — Scan ALL existing locator keys in the ALL EXISTING LOCATOR KEYS section. Search every parent group (nav, product, cart, search, wishlist, category, footer, etc.) and every child key for any locator that targets the same element or serves the same purpose — even if the key name is different.\n" +
    "         CHECK B — Scan the ADDITIONAL CONTEXT FROM CODEBASE (locators file content). Read every selector value, not just the key names. A locator may already exist with a different name (e.g. 'mainImage' vs 'productMainImage', 'thumbnailImages' vs 'galleryThumbnails'). If the selector targets the same element, treat it as ALREADY EXISTING.\n" +
    "         CHECK C — Scan the EXISTING TEST CODE. Look for any `page.locator(...)` or `theSouledStoreLocators.<group>.<key>` references related to the requirement's UI element. If the element is already located in existing tests under a different key, reuse that key.\n" +
    "         CHECK D — Check related or adjacent parent groups. A locator for a product page element might already be defined under `product`, `search`, `pdp`, or `category`. Do not generate a new one if it exists in any group.\n" +
    "       DECISION RULE: Only proceed to generate a new locator if ALL four checks confirm the element is truly absent. If any check finds an existing locator that covers the same element, REUSE it by referencing the existing key — do NOT create a duplicate.\n" +
    "       REUSE FORMAT: Use the existing key directly in the generated test code. Do NOT add it to LOCATOR_UPDATES (it already exists). If you reuse an existing key, note it as: 'Reusing existing: <parentObj>.<childKey>'.\n\n" +
    "     Only if the element is genuinely missing after all four checks, follow this generation pipeline:\n" +
    "       STEP 1 — ANALYZE THE REQUIREMENT: Read the requirement description carefully. Identify exactly what UI element is needed (e.g. 'Add to Cart button on the product page', 'wishlist icon in the nav bar').\n" +
    "       STEP 2 — INSPECT EXISTING LOCATOR STYLE: Look at the existing locators in the codebase context. Understand the naming convention (e.g. camelCase keys, parentObj.childKey structure) and selector style (CSS vs XPath, attribute vs class-based).\n" +
    "       STEP 3 — IDENTIFY THE PARENT OBJECT: Determine which parent group the new locator belongs to (e.g. `nav`, `product`, `cart`, `wishlist`) by matching the requirement's feature area to the existing locator groups.\n" +
    "       STEP 4 — GENERATE A ROBUST SELECTOR: Build the best possible CSS selector using this priority order:\n" +
    "         Priority 1: Stable HTML attributes — `[data-testid='...']`, `[aria-label='...']`, `[name='...']`, `[id='...']`, `[href='...']`, `[alt='...']`, `[title='...']`, `[role='...']`, `[placeholder='...']`\n" +
    "         Priority 2: Semantic HTML elements — `button`, `a`, `input`, `select`, `form`, `nav`, `header`\n" +
    "         Priority 3: Stable, non-dynamic CSS class names visible in the codebase context\n" +
    "         Priority 4: Structural CSS combinators — e.g. `.product-card button`, `.wishlist-section .remove-btn`\n" +
    "         NEVER USE: dynamically hashed classes, autogenerated IDs, positional XPath like `//div[3]`, or any attribute value not confirmed to exist\n" +
    "       STEP 5 — VALIDATE UNIQUENESS: The generated selector must uniquely identify exactly one element type on the target page. If it may match multiple elements, add a qualifier (e.g. `.nav-bar a[href=\"/wishlist\"]`, not just `a`).\n" +
    "       STEP 6 — NAME THE KEY: Name the locator key using camelCase, following the project's naming convention visible in the existing locators file. Example: `addToWishlistBtn`, `galleryThumbnails`, `removeFromWishlist`.\n" +
    "       STEP 7 — OUTPUT IT: Add the new locator to the LOCATOR_UPDATES section as `<parentObj>.<childKey>: <selector>`. Then use `theSouledStoreLocators.<parentObj>.<childKey>` (or the project's locator variable name) in the generated test code.\n" +
    "     - If NO reliable selector can be inferred from the requirement and codebase context, mark it as LOW_CONFIDENCE and explain why: `<parentObj>.<childKey>: LOW_CONFIDENCE — <reason>`.\n" +
    "     - FORBIDDEN: Placeholder selectors, TODO values, empty strings, or any selector that is made up and not inferable from the requirement or codebase.\n" +
    "     - FORBIDDEN: Creating a duplicate locator when an equivalent one already exists under a different key name.\n\n" +
    "F14. Playwright API Strictness Rules:\n" +
    "     - Before suggesting any code, verify it is valid Playwright TypeScript that will compile without errors.\n" +
    "     - Use only official Playwright APIs. Do not use APIs that do not exist.\n" +
    "     - If an assertion or method is uncertain, fall back to documented Playwright API.\n" +
    "     - NEVER use `expect(locator).toHaveCount({ gte: X })` — Playwright's `toHaveCount()` accepts only a number. Instead use `expect(await locator.count()).toBeGreaterThanOrEqual(X)` or `expect(locator.nth(X-1)).toBeVisible()`.\n\n" +

    "F15. No Markdown Inside Executable Code (MANDATORY):\n" +
    "     - NEVER wrap generated TypeScript inside Markdown code fences (```typescript, ```ts, ```, etc.).\n" +
    "     - The PROPOSED_CODE field must contain raw TypeScript only — no Markdown syntax of any kind.\n" +
    "     - Backtick-fenced code blocks written into a .ts file produce a syntax error. Do not use them.\n\n" +

    "F16. No Variable or Import Redeclarations (MANDATORY):\n" +
    "     - Before declaring any variable (const, let), scan the EXISTING TEST CODE block currently in scope.\n" +
    "     - Before adding any import statement, scan ALL existing import lines at the top of the target file.\n" +
    "     - FORBIDDEN: Adding `import { test, expect } from '@playwright/test'` if it already exists in the file.\n" +
    "     - FORBIDDEN: Adding any locator or utility import that is already present in the file's import section.\n" +
    "     - Reuse the exact existing import path and symbol name. Do NOT add a new import for a symbol already imported.\n" +
    "     - If a variable with the same name is already declared in the same test body, DO NOT redeclare it.\n" +
    "     - REUSE the existing variable directly. If the variable holds the wrong value, use a new name.\n" +
    "     - FORBIDDEN: `const element = ...` when `element` is already declared in the same test.\n" +
    "     - FORBIDDEN: `const selector = ...`, `const targetContainer = ...` when\n" +
    "       these are already declared earlier in the same test body.\n\n" +

    "F17. LOW_CONFIDENCE Locators — Never Use in Test Code (MANDATORY):\n" +
    "     - If no reliable selector can be determined, mark the locator in LOCATOR_UPDATES as LOW_CONFIDENCE.\n" +
    "     - In the PROPOSED_CODE, DO NOT write any assertion or action that uses a LOW_CONFIDENCE locator.\n" +
    "     - Instead, leave a TODO comment: `// TODO: add selector to Locators.ts for <elementName> and enable this assertion`.\n" +
    "     - A LOW_CONFIDENCE value written as a Playwright selector causes a runtime error. Do not generate it.\n\n" +

    "F18. No Inference of Unsupported UI Behavior (MANDATORY):\n" +
    "     - If a requirement describes a UI feature (e.g. zoom preview, animation, drag-and-drop, tooltip) that:\n" +
    "         (a) has no matching locator in the locator file, AND\n" +
    "         (b) cannot be verified from the EXISTING TEST CODE or CODEBASE CONTEXT,\n" +
    "       then DO NOT generate a test assertion for it.\n" +
    "     - Instead, in the REASON field, explain: 'This feature cannot be verified from the available codebase context.\n" +
    "       Manual implementation is required once the selector is known.'\n" +
    "     - Classify as REVIEW (not MODIFY) if the only missing coverage is for an unverifiable UI feature.\n" +
    "     - FORBIDDEN: Using `// Assuming a zoom preview...` or similar speculative comments to justify invented assertions.\n\n" +

    // ─────────────────────────────────────────────────────────────────
    // SECTION G: LOCATOR & CONFIGURATION CHANGE HANDLING
    // ─────────────────────────────────────────────────────────────────
    "════════════════════════════════════════════════════════════════\n" +
    "SECTION G — LOCATOR & CONFIGURATION CHANGE HANDLING\n" +
    "════════════════════════════════════════════════════════════════\n" +
    "G1. Automatic Locator Generation (MANDATORY — deep pre-check required first):\n" +
    "    PRE-CHECK (MANDATORY before writing any test code or generating any locator):\n" +
    "    Before generating or suggesting any new locator, you MUST perform a complete dependency audit:\n" +
    "      CHECK A — Read ALL keys in ALL EXISTING LOCATOR KEYS. Search every parent group and every child key for any locator that targets the same element or serves the same purpose, even under a different name.\n" +
    "      CHECK B — Read every selector VALUE in the ADDITIONAL CONTEXT FROM CODEBASE (locators file). A locator may already exist with a semantically equivalent selector under a different key name (e.g. 'product.mainImage' may already target the same element as what the requirement calls 'main product image'). If the element is already targeted, REUSE the existing key.\n" +
    "      CHECK C — Read every locator reference in the EXISTING TEST CODE. If an existing test already uses `theSouledStoreLocators.<group>.<key>` to access the same element the new requirement needs, reuse that exact reference.\n" +
    "      CHECK D — Check adjacent parent groups. The needed locator may be defined under a parent group you did not check first (e.g. a product page image locator might be under `product`, `pdp`, or `search`).\n" +
    "    DECISION RULE: Only generate a new locator if ALL four checks confirm the element is truly absent from the entire locator file. Do NOT add a new locator key if an equivalent one already exists — that creates unmaintainable duplicates.\n" +
    "    REUSE RULE: If an existing locator covers the needed element, reference it directly in the generated test code. Do NOT include it in LOCATOR_UPDATES. Note in your description: 'Reusing existing locator: <parentObj>.<childKey>'.\n\n" +
    "    If a required locator is genuinely MISSING after all checks, generate it automatically following these steps:\n" +
    "      STEP 1 — Read the requirement to identify the exact UI element (e.g. 'wishlist heart icon on product card').\n" +
    "      STEP 2 — Inspect the existing locator file (ADDITIONAL CONTEXT FROM CODEBASE) to learn the naming convention and selector style used in this project.\n" +
    "      STEP 3 — Determine the correct parent group (e.g. `nav`, `product`, `cart`, `wishlist`, `search`) by matching the element's feature area to existing parent objects.\n" +
    "      STEP 4 — Generate the most robust, unique CSS selector possible using this priority:\n" +
    "               #1 Stable HTML attributes: [data-testid], [aria-label], [name], [id], [href], [alt], [title], [role], [placeholder]\n" +
    "               #2 Semantic HTML elements: button, a, input, select, nav, header, form\n" +
    "               #3 Non-dynamic, stable CSS class names confirmed to exist in the codebase context\n" +
    "               #4 Structural CSS combinators: e.g. `.product-card .wishlist-btn`, `.nav-bar a[href='/wishlist']`\n" +
    "               NEVER USE: hashed/dynamic classes, autogenerated IDs, positional XPath (//div[3]), or any value not inferable from the project\n" +
    "      STEP 5 — Ensure the selector is unique on its target page. Add qualifiers if needed.\n" +
    "      STEP 6 — Name the key in camelCase following the project's existing naming convention.\n" +
    "      STEP 7 — Output the locator in LOCATOR_UPDATES as `<parentObj>.<childKey>: <selector>` and use it in the generated test code as `theSouledStoreLocators.<parentObj>.<childKey>` (or the project's locator variable name).\n" +
    "    - If NO reliable selector can be inferred at all, output: `<parentObj>.<childKey>: LOW_CONFIDENCE — <brief reason>` and still use it in the test code so the test is complete.\n" +
    "    - FORBIDDEN: placeholder selectors, TODO values, empty strings, or made-up selectors not inferable from the project.\n" +
    "    - FORBIDDEN: adding a new locator key when an equivalent one already exists anywhere in the locator file.\n\n" +
    "G2. Locator File Placement:\n" +
    "    - Automatically identify the correct locator file in the project (see ADDITIONAL CONTEXT FROM CODEBASE).\n" +
    "    - Add new locators to the same parent object group that logically owns them (e.g. a 'wishlist heart icon on product card' belongs under the `product` group, a 'My Wishlist nav link' belongs under `nav`).\n" +
    "    - Follow the exact TypeScript object literal format used in the existing locator file.\n" +
    "    - All test files that need the new locator must reference it via the locator variable — never hardcode selectors inline in test code.\n\n" +
    "G3. Configuration & Environment Variable Changes:\n" +
    "    - If a requirement change requires a new config property, list it under CONFIG_UPDATES.\n" +
    "    - If a new environment variable is needed, list it under ENV_UPDATES.\n\n" +
    "G4. Requirement Traceability for All Project-Level Changes (MANDATORY):\n" +
    "    - Every project-level change you suggest — including locator additions, locator updates, config changes, environment variable changes, helper function additions, shared utility changes, or constant definitions — MUST be tagged with the exact Requirement ID that directly triggered it.\n" +
    "    - Format every entry in LOCATOR_UPDATES, CONFIG_UPDATES, and ENV_UPDATES as follows:\n" +
    "        [FR-XX] <parentObj>.<childKey>: <value>\n" +
    "      where [FR-XX] is the Requirement ID that directly caused this change to be needed.\n" +
    "    - NEVER associate a project-level update with an unrelated requirement.\n" +
    "    - NEVER attach all locator/config/env updates to the first requirement analyzed simply because it happens to be processed first.\n" +
    "    - If a single locator or config update is needed by multiple requirements, list it once and tag it with ALL relevant Requirement IDs, e.g. [FR-05, FR-12].\n" +
    "    - If you cannot trace a project-level change back to a specific requirement, do NOT suggest it.\n\n" +

    // ─────────────────────────────────────────────────────────────────
    // SECTION H: REQUIRED RESPONSE FORMAT
    // ─────────────────────────────────────────────────────────────────
    "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n" +
    "SECTION H \u2014 REQUIRED RESPONSE FORMAT\n" +
    "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\n" +
    "For EVERY requirement in the document, AND for every requirement ID listed under DELETED REQUIREMENT IDS, output one block in the exact format below.\n" +
    "If the requirement is deleted, write 'DELETED REQUIREMENT' under the REQUIREMENT: label.\n" +
    "Do NOT skip any active or deleted requirement. Do NOT merge multiple requirements into one block.\n" +
    "MANDATORY: Every field must be present. If no data, write 'None'.\n\n" +
    "--- <Requirement ID> ---\n" +
    "REQUIREMENT:\n" +
    "<exact verbatim text from the Requirement Document>\n\n" +
    "MATCHED TEST: <\"test title\" in filename.spec.ts>  OR  'No test found.'\n\n" +
    "IMPLEMENTATION:\n" +
    "<one sentence describing what the matched test currently does — or 'No implementation found.' if no test exists>\n\n" +
    "ATOMIC CHECKS:\n" +
    "List EVERY [Ax] from SECTION A for this requirement. Use this format per atomic:\n" +
    "  [A1] \u2713 COVERED   \u2014 <atomic text>\n" +
    "       Evidence: <exact Playwright line from the test>\n" +
    "  [A2] \u2717 MISSING   \u2014 <atomic text>\n" +
    "       Missing: <conceptual description of what is absent — keep it generic, do NOT write code snippets or specific API examples here>\n" +
    "  [A3] \u26a0 WORDING   \u2014 <atomic text>\n" +
    "       RD says: '<new wording>' | Test asserts: '<old wording>'\n" +
    "       Behavioral impact: <Same behavior = REVIEW / Different behavior = MODIFY>\n" +
    "(Write 'None extracted \u2014 requirement body is structural only.' if no atomics were found.)\n\n" +
    "DETECTED_CHANGES:\n" +
    "If the Requirement Document changed compared to the current test scripts/comments:\n" +
    "For EVERY changed section (Title, Goal, Requirement, Expected Result), output EXACTLY in this format:\n" +
    "  Section: <Title | Goal | Requirement | Expected Result>\n" +
    "  Old:\n" +
    "  <previous text>\n" +
    "  New:\n" +
    "  <current text>\n" +
    "Do NOT print any generic message. Output only the sections that actually changed.\n" +
    "If absolutely nothing changed, write 'None'.\n\n" +
    "CLASSIFICATION: <one of: IN_SYNC | REVIEW | MODIFY | ADD | DELETED>\n" +
    "CODE_CHANGES_REQUIRED: <Yes | No>\n" +
    "(Yes for MODIFY and ADD. No for IN_SYNC, REVIEW, and DELETED.)\n\n" +
    "REASON: <one sentence citing the specific changes or [Ax] that drives this classification>\n\n" +
    "[Include only if CLASSIFICATION is MODIFY or ADD:]\n" +
    "TARGET_FILE: <e.g. navigation.spec.ts>\n\n" +
    "LOCATOR_UPDATES:\n" +
    "[FR-XX] <parentObj>.<childKey>: <CSS selector>\n" +
    "(Write 'None' if not needed.)\n\n" +
    "CONFIG_UPDATES:\n" +
    "[FR-XX] <Config Key>: <value>\n" +
    "(Write 'None' if not needed.)\n\n" +
    "ENV_UPDATES:\n" +
    "[FR-XX] <Env Key>: <value>\n" +
    "(Write 'None' if not needed.)\n\n" +
    "PROPOSED_CODE:\n" +
    "Scope-aware output (MANDATORY — based on SCOPE DECISION TREE in D6):\n" +
    "  SCOPE A (new file): Full compilable .spec.ts — imports + test.describe block + module-scope helpers + test() block.\n" +
    "  SCOPE B (new test block in existing file): ONLY the new test('...', async ({ page }) => { ... }) block. NO imports, NO describe wrapper, NO helpers that already exist in the file.\n" +
    "  SCOPE C (inline patch): ONLY the missing lines. NO surrounding unchanged code.\n" +
    "<No markdown fences. Raw TypeScript only.>\n\n" +
    "PATCH_DIFF:\n" +
    "<@@ -<old_start>,<old_count> +<new_start>,<new_count> @@\n" +
    " <2-3 unchanged context lines above change>\n" +
    "-<removed line>\n" +
    "+<added line>\n" +
    " <2-3 unchanged context lines below change>>\n" +
    "Write 'None' for SCOPE A (new file) or SCOPE B (new test block — put full block in PROPOSED_CODE instead).\n\n" +
    "WHY_NEEDED:\n" +
    "<why Option 1 is robust>\n\n" +
    "PROPOSED_CODE_OPT2:\n" +
    "<compilable Option 2 TypeScript code block — no markdown fences>\n\n" +
    "PATCH_DIFF_OPT2:\n" +
    "@@ -1,1 +1,1 @@\n" +
    "<Option 2 unified diff block — write 'None' for new files>\n\n" +
    "WHY_NEEDED_OPT2:\n" +
    "<why Option 2 is robust/alternative approach>\n"
  );
}


/**
 * Sanitizes unescaped quotes in test('title', ...) lines inside generated test code blocks.
 */
function escapeTestTitlesInCode(code: string): string {
  const regex = /\b(test(?:\.(?:only|skip|fixme|fail))?\s*\(\s*)(['"`])([\s\S]*?)\2(\s*,\s*(?:async\s*)?\((?:[\s\S]*?)\)\s*=>\s*\{)/g;
  return code.replace(regex, (match, p1, p2, p3, p4) => {
    const escapedTitle = p3.replace(new RegExp(`(?<!\\\\)${p2}`, 'g'), `\\${p2}`);
    return `${p1}${p2}${escapedTitle}${p2}${p4}`;
  });
}

/**
 * Parses a key-value section within a requirement block (LOCATOR_UPDATES, CONFIG_UPDATES, ENV_UPDATES).
 */
function parseBlockKeyValues(secText: string): { key: string; value: string; requirementId?: string }[] {
  const results: { key: string; value: string; requirementId?: string }[] = [];
  if (!secText || secText.trim().toLowerCase() === 'none') return results;
  const lines = secText.split('\n');
  for (const l of lines) {
    const trimmed = l.trim();
    if (!trimmed || trimmed.toLowerCase() === 'none' || trimmed.startsWith('<') || trimmed.startsWith('(')) continue;
    let requirementId: string | undefined;
    let rest = trimmed;
    const tagMatch = trimmed.match(/^\[([A-Z0-9,\s\-]+)\]\s*/i);
    if (tagMatch) {
      const firstId = tagMatch[1].split(',')[0].trim();
      const normId = firstId.match(/\b(R\d+|FR-\d+|TS-\d+|TC-[A-Z0-9\-]+)\b/i);
      requirementId = normId ? normId[1].toUpperCase() : undefined;
      rest = trimmed.substring(tagMatch[0].length);
    }
    const colonIdx = rest.indexOf(':');
    if (colonIdx !== -1) {
      const key = rest.substring(0, colonIdx).trim().replace(/\s*\(.*?\)/, '').trim();
      const value = rest.substring(colonIdx + 1).trim();
      if (key) results.push({ key, value, requirementId });
    }
  }
  return results;
}

/**
 * Extracts the value of a named field from a block of text.
 * Reads from the field label until the next known field label or end of text.
 */
function extractBlockField(blockText: string, fieldName: string, nextFields: string[]): string {
  const fieldPattern = new RegExp(`(?:^|\\n)${fieldName}:\\s*`, 'i');
  const match = fieldPattern.exec(blockText);
  if (!match) return '';
  const start = match.index + match[0].length;
  let end = blockText.length;
  for (const next of nextFields) {
    const nextPattern = new RegExp(`(?:^|\\n)${next}:`, 'i');
    const nextMatch = nextPattern.exec(blockText.substring(start));
    if (nextMatch) {
      const candidate = start + nextMatch.index;
      if (candidate < end) end = candidate;
    }
  }
  return blockText.substring(start, end).trim();
}

/**
 * Text parser to convert the per-requirement structured block format into Suggestion[] array.
 * Handles blocks of the form:
 *   --- FR-XX ---
 *   REQUIREMENT: ...
 *   IMPLEMENTATION: ...
 *   DIFFERENCES: ...
 *   IMPACT ANALYSIS: ...
 *   CLASSIFICATION: IN_SYNC | REVIEW | MODIFY | ADD | DELETED
 *   CODE_CHANGES_REQUIRED: Yes | No
 *   REASON: ...
 *   TARGET_FILE: ...
 *   LOCATOR_UPDATES: ...
 *   CONFIG_UPDATES: ...
 *   ENV_UPDATES: ...
 *   PROPOSED_CODE: ...
 */
function parseAndHydrateSuggestions(
  text: string,
  requirements: Requirement[],
  tests: ParsedTest[]
): Suggestion[] {
  const suggestions: Suggestion[] = [];
  const existingFiles = Array.from(new Set(tests.map(t => t.filePath)));

  // Load previous snapshot from cache to perform local comparison
  const cachePath = path.join(process.cwd(), '.qa-sync-cache.json');
  let prevRequirements: Requirement[] = [];
  if (fs.existsSync(cachePath)) {
    try {
      prevRequirements = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    } catch (e) {
      // ignore
    }
  }
  const reqDiffs = compareRequirementDocuments(prevRequirements, requirements);

  const getReqTitle = (reqId: string): string => {
    const found = requirements.find(r => r.id.toUpperCase() === reqId.toUpperCase());
    if (found) return found.title;
    const foundTest = tests.find(t => t.title.toUpperCase() === reqId.toUpperCase() || t.requirementIds.map(normalizeRequirementId).includes(normalizeRequirementId(reqId)));
    return foundTest ? foundTest.title : reqId;
  };

  // ── Parse per-requirement structured blocks: --- <any header> --- ────────
  // Match ANY --- text --- delimiter so the parser works regardless of whether
  // Gemini uses "--- FR-1 ---", "--- I. Homepage Brand Identity ---", or any other format.
  const blockSplitter = /(?:^|\n)---\s*([^\n\-][^\n]*?)\s*---/gi;
  const blockMatches: { id: string; start: number }[] = [];
  let bm: RegExpExecArray | null;
  while ((bm = blockSplitter.exec(text)) !== null) {
    const headerText = bm[1].trim();

    // 1. Try to extract an explicit requirement ID from the header text (FR-1, FR-01, R1, TS-1, TC-XX)
    const idMatch = headerText.match(/\b((?:FR|R|TS|TC)-?\d+[A-Z0-9\-]*)\b/i);
    if (idMatch) {
      blockMatches.push({ id: idMatch[1].toUpperCase(), start: bm.index + bm[0].length });
      continue;
    }

    // 2. Strip leading roman numerals or numbers (e.g. "I. ", "II. ", "1. ", "1) ") from header
    const strippedTitle = headerText
      .replace(/^[IVXLCDMivxlcdm]{1,10}\.\s+/, '')
      .replace(/^\d+[.)]\s+/, '')
      .trim()
      .toLowerCase();

    // 3. Match by title against the known requirements list
    const matchedReq = requirements.find(r => {
      const rTitle = r.title.toLowerCase();
      return rTitle === strippedTitle
        || strippedTitle.includes(rTitle)
        || rTitle.includes(strippedTitle);
    });
    if (matchedReq) {
      blockMatches.push({ id: matchedReq.id, start: bm.index + bm[0].length });
      continue;
    }

    // 3.5 Match by title against the known test cases (for deleted/orphan test cases)
    const matchedTest = tests.find(t => {
      const tTitle = t.title.toLowerCase();
      return tTitle === strippedTitle
        || strippedTitle.includes(tTitle)
        || tTitle.includes(strippedTitle);
    });
    if (matchedTest) {
      blockMatches.push({ id: matchedTest.requirementId || matchedTest.title, start: bm.index + bm[0].length });
      continue;
    }

    // 4. Try matching header against requirement title using word overlap (fuzzy fallback)
    const headerWords = strippedTitle.split(/\s+/).filter(w => w.length >= 4);
    if (headerWords.length > 0) {
      let bestReq: Requirement | undefined;
      let bestScore = 0;
      for (const req of requirements) {
        const reqWords = req.title.toLowerCase().split(/\s+/);
        const score = headerWords.filter(hw => reqWords.some(rw => rw.includes(hw) || hw.includes(rw))).length;
        if (score > bestScore) { bestScore = score; bestReq = req; }
      }
      if (bestReq && bestScore >= 2) {
        blockMatches.push({ id: bestReq.id, start: bm.index + bm[0].length });
      }
    }
  }


  if (blockMatches.length > 0) {
    // New structured block format
    for (let bi = 0; bi < blockMatches.length; bi++) {
      const blockId = blockMatches[bi].id;
      const blockStart = blockMatches[bi].start;
      const blockEnd = bi + 1 < blockMatches.length ? blockMatches[bi + 1].start - blockMatches[bi + 1].id.length - 10 : text.length;
      const blockText = text.substring(blockStart, blockEnd);

      const allFields = [
        'REQUIREMENT',
        'MATCHED TEST',
        'IMPLEMENTATION',
        'ATOMIC CHECKS',
        'DIFFERENCES',
        'IMPACT ANALYSIS',
        'DETECTED_CHANGES',
        'CLASSIFICATION',
        'CODE_CHANGES_REQUIRED',
        'REASON',
        'TARGET_FILE',
        'LOCATOR_UPDATES',
        'CONFIG_UPDATES',
        'ENV_UPDATES',
        'PROPOSED_CODE',
        'PATCH_DIFF',
        'WHY_NEEDED',
        'PROPOSED_CODE_OPT2',
        'PATCH_DIFF_OPT2',
        'WHY_NEEDED_OPT2'
      ];

      const requirementText = extractBlockField(blockText, 'REQUIREMENT', allFields.slice(1));
      let implementationSummary = extractBlockField(blockText, 'MATCHED TEST', allFields.slice(3));
      if (!implementationSummary) {
        implementationSummary = extractBlockField(blockText, 'IMPLEMENTATION', allFields.slice(3));
      }
      let differencesFound = extractBlockField(blockText, 'ATOMIC CHECKS', allFields.slice(5));
      if (!differencesFound) {
        differencesFound = extractBlockField(blockText, 'DIFFERENCES', allFields.slice(5));
      }
      const impactAnalysis = extractBlockField(blockText, 'IMPACT ANALYSIS', allFields.slice(6)) || 'None';
      const detectedChanges = extractBlockField(blockText, 'DETECTED_CHANGES', allFields.slice(7)) || 'None';
      const classificationRaw = extractBlockField(blockText, 'CLASSIFICATION', allFields.slice(8)).split('\n')[0].trim().toUpperCase();
      const reason = extractBlockField(blockText, 'REASON', allFields.slice(9));
      const targetFileRaw = extractBlockField(blockText, 'TARGET_FILE', allFields.slice(10)).split('\n')[0].trim();
      const locatorUpdatesRaw = extractBlockField(blockText, 'LOCATOR_UPDATES', allFields.slice(11));
      const configUpdatesRaw = extractBlockField(blockText, 'CONFIG_UPDATES', allFields.slice(12));
      const envUpdatesRaw = extractBlockField(blockText, 'ENV_UPDATES', allFields.slice(13));
      const proposedCodeRaw = extractBlockField(blockText, 'PROPOSED_CODE', ['PATCH_DIFF', 'WHY_NEEDED', 'PROPOSED_CODE_OPT2', 'PATCH_DIFF_OPT2', 'WHY_NEEDED_OPT2']);
      const patchDiffRaw = extractBlockField(blockText, 'PATCH_DIFF', ['WHY_NEEDED', 'PROPOSED_CODE_OPT2', 'PATCH_DIFF_OPT2', 'WHY_NEEDED_OPT2']);
      const whyNeededRaw = extractBlockField(blockText, 'WHY_NEEDED', ['PROPOSED_CODE_OPT2', 'PATCH_DIFF_OPT2', 'WHY_NEEDED_OPT2']);
      const proposedCodeOpt2Raw = extractBlockField(blockText, 'PROPOSED_CODE_OPT2', ['PATCH_DIFF_OPT2', 'WHY_NEEDED_OPT2']);
      const patchDiffOpt2Raw = extractBlockField(blockText, 'PATCH_DIFF_OPT2', ['WHY_NEEDED_OPT2']);
      const whyNeededOpt2Raw = extractBlockField(blockText, 'WHY_NEEDED_OPT2', []);

      const classification = classificationRaw.includes('MODIFY') ? 'MODIFY'
        : classificationRaw.includes('ADD') ? 'ADD'
          : classificationRaw.includes('DELETED') || classificationRaw.includes('DELETE') ? 'DELETED'
            : classificationRaw.includes('REVIEW') ? 'REVIEW'
              : 'IN_SYNC';

      const codeChangesRequired = (classification === 'MODIFY' || classification === 'ADD') ? 'Yes' : 'No';

      const action: Suggestion['action'] =
        classification === 'MODIFY' ? 'MODIFY' :
          classification === 'ADD' ? 'ADD' :
            classification === 'DELETED' ? 'REMOVE' :
              'NONE';

      const normId = normalizeRequirementId(blockId);
      const diff = reqDiffs.get(normId);
      const matchedTest = tests.find(t => t.requirementIds.map(normalizeRequirementId).includes(normId) || t.title.toUpperCase() === blockId.toUpperCase());

      // Resolve target file path
      let targetPath: string | undefined;
      if (targetFileRaw && targetFileRaw.toLowerCase() !== 'none') {
        const baseName = path.basename(targetFileRaw);
        const matchedFile = existingFiles.find(f => path.basename(f) === baseName);
        targetPath = matchedFile || (targetFileRaw.endsWith('.spec.ts') ? path.join('tests', targetFileRaw).replace(/\\/g, '/') : undefined);
      }
      if (!targetPath) targetPath = matchedTest?.filePath;
      if (!targetPath) targetPath = guessFilePath(blockId, getReqTitle(blockId), existingFiles);

      let proposedCode: string | undefined;
      let patchDiff: string | undefined;
      let whyNeeded: string | undefined;
      let proposedCodeOpt2: string | undefined;
      let patchDiffOpt2: string | undefined;
      let whyNeededOpt2: string | undefined;

      if (proposedCodeRaw && proposedCodeRaw.trim().toLowerCase() !== 'none' && proposedCodeRaw.trim() !== '') {
        proposedCode = escapeTestTitlesInCode(proposedCodeRaw.trim());
      }
      if (patchDiffRaw && patchDiffRaw.trim().toLowerCase() !== 'none' && patchDiffRaw.trim() !== '') {
        patchDiff = patchDiffRaw.trim();
      }
      if (whyNeededRaw && whyNeededRaw.trim().toLowerCase() !== 'none' && whyNeededRaw.trim() !== '') {
        whyNeeded = whyNeededRaw.trim();
      }

      if (proposedCodeOpt2Raw && proposedCodeOpt2Raw.trim().toLowerCase() !== 'none' && proposedCodeOpt2Raw.trim() !== '') {
        proposedCodeOpt2 = escapeTestTitlesInCode(proposedCodeOpt2Raw.trim());
      }
      if (patchDiffOpt2Raw && patchDiffOpt2Raw.trim().toLowerCase() !== 'none' && patchDiffOpt2Raw.trim() !== '') {
        patchDiffOpt2 = patchDiffOpt2Raw.trim();
      }
      if (whyNeededOpt2Raw && whyNeededOpt2Raw.trim().toLowerCase() !== 'none' && whyNeededOpt2Raw.trim() !== '') {
        whyNeededOpt2 = whyNeededOpt2Raw.trim();
      }

      // Parse locator/config/env updates from this block
      const locatorUpdates = parseBlockKeyValues(locatorUpdatesRaw);
      const configUpdates = parseBlockKeyValues(configUpdatesRaw);
      const envUpdates = parseBlockKeyValues(envUpdatesRaw);

      // Build the description (shown in terminal)
      const noDiff = differencesFound.trim().toLowerCase() === 'no differences detected.' ||
        differencesFound.trim().toLowerCase() === 'none' ||
        differencesFound.trim() === '';
      const description = [
        `• Requirement:\n  ${requirementText.replace(/\n/g, '\n  ')}`,
        `• Implementation:\n  ${implementationSummary.replace(/\n/g, '\n  ')}`,
        `• Differences: ${noDiff ? 'No differences detected.' : '\n  ' + differencesFound.replace(/\n/g, '\n  ')}`,
        `• Impact Analysis:\n  ${impactAnalysis.replace(/\n/g, '\n  ')}`,
        `• Recommendation: ${classification}  |  Code Changes Required: ${codeChangesRequired}`,
        `• Reason:\n  ${reason.replace(/\n/g, '\n  ')}`,
      ].join('\n');

      suggestions.push({
        requirementId: blockId,
        action,
        title: getReqTitle(blockId),
        description,
        filePath: targetPath,
        testTitle: matchedTest?.title,
        startLine: matchedTest?.startLine,
        endLine: matchedTest?.endLine,
        originalCode: matchedTest?.fullText,
        proposedCode,
        patchDiff,
        whyNeeded,
        proposedCodeOpt2,
        patchDiffOpt2,
        whyNeededOpt2,
        requirementText,
        implementationSummary,
        differencesFound: noDiff ? 'No differences detected.' : differencesFound,
        impactAnalysis,
        codeChangesRequired,
        classification,
        detectedChanges: (diff ? formatDetectedChanges(diff) : (detectedChanges || 'None')),
        locatorUpdates: locatorUpdates.length > 0 ? locatorUpdates.map(u => ({ key: u.key, value: u.value })) : undefined,
        configUpdates: configUpdates.length > 0 ? configUpdates.map(u => ({ key: u.key, value: u.value })) : undefined,
        envUpdates: envUpdates.length > 0 ? envUpdates.map(u => ({ key: u.key, value: u.value })) : undefined,
      });
    }
  } else {
    // ── Fallback: legacy flat-section format (IN_SYNC: / NEW: / MODIFIED: etc.) ──
    const headers = ['IN_SYNC:', 'NEW:', 'MODIFIED:', 'DELETED:',
      'LOCATOR_UPDATES:', 'CONFIG_UPDATES:', 'ENV_UPDATES:', 'NEW_TEST_CODE:', 'MODIFIED_TEST_CODE:'];
    const indices: { header: string; index: number }[] = [];
    for (const h of headers) {
      const idx = text.indexOf(h);
      if (idx !== -1) indices.push({ header: h, index: idx });
    }
    indices.sort((a, b) => a.index - b.index);
    const sections: { [key: string]: string } = {};
    for (let i = 0; i < indices.length; i++) {
      const start = indices[i].index + indices[i].header.length;
      const end = (i + 1 < indices.length) ? indices[i + 1].index : text.length;
      sections[indices[i].header] = text.substring(start, end).trim();
    }
    const parseLines = (secText: string | undefined): { id: string; reason: string; filePath?: string }[] => {
      if (!secText) return [];
      const results: { id: string; reason: string; filePath?: string }[] = [];
      let current: { id: string; reason: string; filePath?: string } | null = null;
      for (const l of (secText.split('\n'))) {
        const trimmed = l.trim();
        if (!trimmed || trimmed.toLowerCase() === 'none' || trimmed.startsWith('<')) continue;
        // Fix Issue 15: R-?\d+ covers both R01 and hyphenated R-01 forms in the AI response
        const pMatch = trimmed.match(/^[-*\s]*\b(R-?\d+|FR-\d+|TS-\d+|TC\-[A-Z0-9\-]+)\b(?:\s*\(([^)]+)\))?\s*[:\-]\s*(.+)$/i);
        if (pMatch) {
          if (current) results.push(current);
          current = { id: pMatch[1].toUpperCase(), filePath: pMatch[2]?.trim(), reason: pMatch[3].trim() };
        } else if (current) {
          current.reason = current.reason ? current.reason + '\n' + trimmed : trimmed;
        }
      }
      if (current) results.push(current);
      return results;
    };
    const parseKV = (secText: string | undefined) => parseBlockKeyValues(secText || '');

    const inSync = parseLines(sections['IN_SYNC:']);
    const newReqs = parseLines(sections['NEW:']);
    const modifiedReqs = parseLines(sections['MODIFIED:']);
    const deletedReqs = parseLines(sections['DELETED:']);
    const locatorUpdates = parseKV(sections['LOCATOR_UPDATES:']);
    const configUpdates = parseKV(sections['CONFIG_UPDATES:']);
    const envUpdates = parseKV(sections['ENV_UPDATES:']);
    const newTestCode = escapeTestTitlesInCode(sections['NEW_TEST_CODE:'] || '');
    const modifiedTestCode = escapeTestTitlesInCode(sections['MODIFIED_TEST_CODE:'] || '');

    const allTestCode = newTestCode + '\n' + modifiedTestCode;
    const allSegments = parseTestFileStructure(allTestCode, 'dummy.spec.ts');
    const allTestCases: any[] = [];
    for (let i = 0; i < allSegments.length; i++) {
      if (allSegments[i].type === 'testCase') allTestCases.push((allSegments[i] as any).structure);
    }

    for (const item of inSync) {
      const normId = normalizeRequirementId(item.id);
      const mt = tests.find(t => t.requirementIds.map(normalizeRequirementId).includes(normId));
      suggestions.push({
        requirementId: item.id, action: 'NONE', title: getReqTitle(item.id),
        description: item.reason || `Covered by ${item.id}.`, filePath: mt?.filePath, testTitle: mt?.title,
        startLine: mt?.startLine, endLine: mt?.endLine
      });
    }
    for (const item of newReqs) {
      const tc = findMatchingTestCase(item.id, allTestCases);
      const title = getReqTitle(item.id);
      let fp = item.filePath;
      if (fp) { const bn = path.basename(fp); const mf = existingFiles.find(f => path.basename(f) === bn); fp = mf || (fp.endsWith('.spec.ts') ? path.join('tests', fp).replace(/\\/g, '/') : undefined); }
      suggestions.push({
        requirementId: item.id, action: 'ADD', title,
        description: `• Why this is needed: ${item.reason || `Requirement ${item.id} is missing from the test script.`}\n• Code updates: Adding a new test block for ${title}`,
        filePath: fp || guessFilePath(item.id, title, existingFiles),
        proposedCode: tc?.fullText, testTitle: tc?.title
      });
    }
    for (const item of modifiedReqs) {
      const tc = findMatchingTestCase(item.id, allTestCases);
      const title = getReqTitle(item.id);
      const normId = normalizeRequirementId(item.id);
      const mt = tests.find(t => t.requirementIds.map(normalizeRequirementId).includes(normId) || (tc && t.title === tc.title));
      let fp = item.filePath;
      if (fp) { const bn = path.basename(fp); const mf = existingFiles.find(f => path.basename(f) === bn); fp = mf || (fp.endsWith('.spec.ts') ? path.join('tests', fp).replace(/\\/g, '/') : undefined); }
      suggestions.push({
        requirementId: item.id, action: 'MODIFY', title,
        description: `• Why this is needed: ${item.reason || `Test for ${item.id} needs updating.`}\n• Code updates: Modifying the test block.`,
        filePath: fp || mt?.filePath || guessFilePath(item.id, title, existingFiles),
        testTitle: mt?.title || tc?.title, proposedCode: tc?.fullText,
        startLine: mt?.startLine, endLine: mt?.endLine,
        originalCode: mt?.requirementBlocks?.find(b => normalizeRequirementId(b.id) === normId)?.code || mt?.fullText
      });
    }
    for (const item of deletedReqs) {
      const normId = normalizeRequirementId(item.id);
      const mt = tests.find(t => t.requirementIds.map(normalizeRequirementId).includes(normId) || t.title.includes(item.id));
      suggestions.push({
        requirementId: item.id, action: 'REMOVE', title: `Orphan test: ${item.id}`,
        description: `• Why this is needed: ${item.reason || `Requirement ${item.id} not in the document.`}\n• Code updates: Removing the corresponding test block.`,
        filePath: mt?.filePath, testTitle: mt?.title,
        startLine: mt?.startLine, endLine: mt?.endLine,
        originalCode: mt?.requirementBlocks?.find(b => b.id === item.id)?.code || mt?.fullText
      });
    }
    // Route locator/config/env updates in fallback mode
    const reqIdToIndex = new Map<string, number>();
    for (let i = 0; i < suggestions.length; i++) reqIdToIndex.set(normalizeRequirementId(suggestions[i].requirementId), i);
    const fallbackIdx = suggestions.findIndex(s => s.action === 'ADD' || s.action === 'MODIFY');
    const fallback = fallbackIdx !== -1 ? fallbackIdx : (suggestions.length > 0 ? 0 : -1);
    for (const upd of [...locatorUpdates, ...configUpdates, ...envUpdates]) {
      let ti = fallback;
      if (upd.requirementId) { const f = reqIdToIndex.get(normalizeRequirementId(upd.requirementId)); if (f !== undefined) ti = f; }
      if (ti === -1) continue;
      const tgt = suggestions[ti];
      // route by type based on presence of dots (locator style) vs plain key
      if (upd.key.includes('.')) { if (!tgt.locatorUpdates) tgt.locatorUpdates = []; tgt.locatorUpdates.push({ key: upd.key, value: upd.value }); }
      else { if (!tgt.configUpdates) tgt.configUpdates = []; tgt.configUpdates.push({ key: upd.key, value: upd.value }); }
    }
  }

  // File routing validation pass (for structured blocks too)
  const updatedSuggestions: Suggestion[] = [];
  for (const s of suggestions) {
    if ((s.action === 'NONE' || s.action === 'MODIFY') && s.filePath) {
      const correctFile = guessFilePath(s.requirementId, s.title, existingFiles, s.filePath);
      if (path.basename(s.filePath) !== path.basename(correctFile)) {
        updatedSuggestions.push({
          ...s, action: 'REMOVE',
          description: `• Why this is needed: Requirement ${s.requirementId} is in the wrong spec file (${path.basename(s.filePath)}).\n• Code updates: Removing the test from ${path.basename(s.filePath)}.`
        });
        updatedSuggestions.push({
          ...s, action: 'ADD', filePath: correctFile,
          description: `• Why this is needed: Requirement ${s.requirementId} belongs in ${path.basename(correctFile)}.\n• Code updates: Adding the test to ${path.basename(correctFile)}.`,
          testTitle: `TS-${s.requirementId.replace(/^\D+/, '')}: Verify ${s.title}`
        });
      } else {
        updatedSuggestions.push(s);
      }
    } else {
      updatedSuggestions.push(s);
    }
  }

  // ── Post-processing: Enforce deterministic classifications ────────────────
  for (const s of updatedSuggestions) {
    const normId = normalizeRequirementId(s.requirementId);
    const diff = reqDiffs.get(normId);

    if (diff && diff.classification !== 'IN_SYNC') {
      s.classification = diff.classification;
      if (diff.classification === 'MODIFY') {
        s.action = 'MODIFY';
      } else if (diff.classification === 'ADD') {
        s.action = 'ADD';
      } else if (diff.classification === 'REVIEW') {
        s.action = 'NONE';
      }
    }
  }

  return updatedSuggestions;
}

function findMatchingTestCase(reqId: string, testCases: any[]): any | null {
  const normReqId = normalizeRequirementId(reqId);
  for (const tc of testCases) {
    const ids = extractAllRequirementIds(tc.title, '', tc.fullText).map(normalizeRequirementId);
    if (ids.includes(normReqId)) {
      return tc;
    }
  }
  return null;
}

/**
 * Dynamically matches a requirement to an existing spec file using the file's own name as keyword source.
 * Works for any project — does NOT rely on a hardcoded keyword list.
 */
function guessFilePath(reqId: string, title: string, existingFiles: string[], defaultPath?: string): string {
  const dir = existingFiles.length > 0 ? path.dirname(existingFiles[0]) : 'tests';

  // Layer 1: If files follow a strict per-ID naming convention (e.g., FR-01.spec.ts),
  // route the requirement to its own dedicated spec file.
  const hasIdPattern = existingFiles.some(f => {
    const base = path.basename(f);
    return /^(FR|TS|TC|R)-?\d+\.spec\.ts$/i.test(base);
  });

  if (hasIdPattern) {
    return path.join(dir, `${reqId.toUpperCase()}.spec.ts`).replace(/\\/g, '/');
  }

  // Layer 2: Match keywords from requirement ID + title + description against the ACTUAL
  // existing spec file names in this project.
  const reqLower = (reqId + ' ' + title).toLowerCase();
  
  // Clean title to extract key terms
  const stopWords = new Set(['and', 'the', 'for', 'with', 'from', 'page', 'verification', 'spec', 'test', 'operations', 'pages', 'flows', 'cases', 'checking']);
  const reqWords = reqLower.split(/[^a-z0-9]+/).filter(w => w.length >= 3 && !stopWords.has(w));

  let bestMatch = '';
  let bestScore = 0;

  for (const f of existingFiles) {
    const base = path.basename(f, '.spec.ts').toLowerCase();
    const fileKeywords = base.split(/[-_.\s]+/).filter(kw => kw.length >= 3 && !stopWords.has(kw));

    let score = 0;
    
    // Synonym and abbreviation matching for standard e-commerce features
    if (base === 'pdp' && (reqLower.includes('product page') || reqLower.includes('product detail') || reqLower.includes('pdp'))) {
      score += 4;
    }
    if (base === 'plp' && (reqLower.includes('product listing') || reqLower.includes('category') || reqLower.includes('plp'))) {
      score += 4;
    }

    for (const kw of fileKeywords) {
      if (reqWords.includes(kw) || reqLower.includes(kw)) {
        score += 2; // Exact word match
      } else {
        // Partial substring match
        for (const rw of reqWords) {
          if (rw.includes(kw) || kw.includes(rw)) {
            score += 1;
          }
        }
      }
    }

    if (score > bestScore) {
      bestScore = score;
      bestMatch = f;
    }
  }

  // Only use keyword match if a significant match was found
  if (bestScore > 0 && bestMatch) {
    return bestMatch.replace(/\\/g, '/');
  }

  // If the requirement already specifies an existing spec file and it was previously mapped,
  // check if the defaultPath is actually a suitable file.
  if (defaultPath) {
    const resolvedDefault = path.resolve(defaultPath);
    if (existingFiles.some(f => path.resolve(f) === resolvedDefault)) {
      return defaultPath.replace(/\\/g, '/');
    }
  }

  // Layer 3: No suitable existing file exists.
  // Generate a clean, appropriate name for a new spec file based on the requirement title.
  const cleanTitle = title
    .toLowerCase()
    .replace(/^(fr|ts|tc|r)-?\d+\s*[:\-]?\s*/, '') // remove leading ID if any
    .replace(/[^a-z0-9\s-]/g, '') // remove special chars
    .trim();

  const titleWords = cleanTitle.split(/\s+/).filter(w => w.length >= 2 && !stopWords.has(w));
  
  let baseName = '';
  if (titleWords.length > 0) {
    baseName = titleWords.slice(0, 3).join('-');
  } else {
    baseName = cleanTitle.replace(/\s+/g, '-');
  }

  if (!baseName || baseName === '-') {
    baseName = reqId.toLowerCase() || 'general';
  }

  // Ensure it ends with .spec.ts
  const newFileName = `${baseName}.spec.ts`;
  return path.join(dir, newFileName).replace(/\\/g, '/');
}

async function callWithRetry<T>(fn: () => Promise<T>, maxRetries = 3, delayMs = 2000): Promise<T> {
  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (error: any) {
      attempt++;

      // If the error is 404 (Not Found / unsupported model), throw immediately to skip retrying this model
      const is404 = error.status === 404 ||
        (error.message && (error.message.includes('404') || error.message.toLowerCase().includes('not found') || error.message.toLowerCase().includes('not supported')));
      if (is404) {
        throw error;
      }

      // If the error is a quota/rate limit error (429 or quota/billing limits), throw immediately to fallback
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
      console.warn(`  ⚠️ API call failed (attempt ${attempt}/${maxRetries}): ${error.message || error}. Retrying in ${delayMs}ms...`);
      await new Promise(resolve => setTimeout(resolve, delayMs));
      delayMs *= 2; // exponential backoff
    }
  }
}

/**
 * Uses Gemini API to perform comparison between RD requirements and Playwright tests.
 */
async function analyzeWithGemini(
  apiKey: string,
  requirements: Requirement[],
  tests: ParsedTest[],
  rawRdContent?: string,
  projectRoot: string = process.cwd(),
  missingLocators: { key: string; file: string }[] = [],
  rdChanged: boolean = false
): Promise<Suggestion[]> {
  // Generate a unique nonce for this invocation.
  // This is passed into buildUserAnalysisPrompt and injected as the first token of the
  // prompt, busting the Gemini server-side response cache on every run.
  const runNonce = Date.now().toString(36) + '-' + process.pid.toString(36);
  const genAI = new GoogleGenerativeAI(apiKey);
  const systemInstruction = [
    "ROLE: You are a senior QA automation engineer.",
    "Your sole responsibility is to sync Playwright test scripts with a requirements document.",
    "",
    "═══ PHASE 0 — PROJECT COMPREHENSION (MANDATORY — complete before any analysis) ═══",
    "CRITICAL: Before analyzing any requirement, you MUST perform a complete read-and-catalogue pass over ALL provided context.",
    "Read the following in full and build an internal catalogue:",
    "  CATALOGUE 1 — IMPORTS: Every import statement in EXISTING TEST CODE and locator files. Record: symbol name + import path.",
    "  CATALOGUE 2 — HELPERS: Every helper or utility function defined at module scope (e.g. dismissMembershipPopup, loginUser). Record: name, parameters, purpose.",
    "  CATALOGUE 3 — FIXTURES: Every test.beforeEach, test.afterEach, and test.use(...) block. Record: what state they set up or tear down.",
    "  CATALOGUE 4 — LOCATOR GROUPS: Every parent object and child key from ALL EXISTING LOCATOR KEYS. Record: parent.child -> selector.",
    "  CATALOGUE 5 — TEST TITLES: Every test('...') and test.describe('...') title already present. Record: title -> file.",
    "  CATALOGUE 6 — CONFIG: Every property found in CONFIG FILE CONTENT. Record: key -> value.",
    "  CATALOGUE 7 — TSCONFIG: compilerOptions (target, lib, strict, module type).",
    "  CATALOGUE 8 — PACKAGE.JSON: dependencies and devDependencies (allowable libraries and APIs).",
    "  CATALOGUE 9 — PROJECT HELPERS: helper/utility files found under tests/helpers/ (signatures and usage).",
    "RULES FROM THIS CATALOGUE:",
    "  - Do NOT generate any import that is already in CATALOGUE 1.",
    "  - Do NOT redefine any helper already in CATALOGUE 2 or CATALOGUE 9. Reuse it by name (e.g., if dismissMembershipPopup already exists in tests/helpers/dismissPopup.ts, reuse/import it).",
    "  - Do NOT duplicate any test title already in CATALOGUE 5.",
    "  - Do NOT reference any locator key not in CATALOGUE 4 unless it was first added to LOCATOR_UPDATES.",
    "  - FORBIDDEN BROWSER GLOBALS: Never use window, document, HTMLElement, querySelector, or other browser globals directly in Playwright test files. These globals are only valid inside page.evaluate(). All DOM interactions must use Playwright APIs (page.locator(), getByRole(), getByText(), etc.). Generated code must compile under Node.js without requiring DOM libraries.",
    "  - Never use external libraries or APIs that are not defined in package.json or supported by tsconfig.json.",
    "  - Adapt generated code to the project's configuration (tsconfig, package.json dependencies, target language options, existing directory structures) rather than expecting the project to adapt to the generated code.",
    "Only after completing this catalogue pass, proceed to Stage 1.",
    "",
    "═══ ATOMIC WORD-LEVEL ANALYSIS METHODOLOGY ═══",
    "CRITICAL: You must perform true atomic, word-level analysis on the entire Requirement Document.",
    "- Analyze every Title, Goal, Requirement, and Expected Result individually. NEVER merge these sections together. Each section must be analyzed independently.",
    "- Read the complete Requirement Document word-by-word, line-by-line, and statement-by-statement before starting the comparison.",
    "- Detect every added, removed, or modified word, token, sentence, line, or section, even if only a single character or word changes.",
    "- Identify exactly where the change occurred (Title, Goal, Requirement, or Expected Result).",
    "- Compare every atomic statement with the corresponding Playwright implementation.",
    "- Use the semantic meaning of each section:",
    "    Requirement: maps to executable behavior.",
    "    Goal: maps to the purpose of the test.",
    "    Expected Result: maps to the expected outcome.",
    "    Title: maps to the test scope and mapping.",
    "- If a Goal, Title, or Expected Result changes, determine whether the existing test still satisfies the new intent.",
    "- REVIEW : Only documentation changed (Goal, Expected Result, headings, numbering, formatting, grammar, or wording that does not change behavior). Never classify the addition, removal, or modification of executable requirements as REVIEW.",
    "- IN_SYNC : The existing test fully satisfies the latest Requirement Document, and there is no obsolete, extra, or removed step implementation in the test. If any step was removed from the Requirement Document, the corresponding code in the test is now obsolete/extra, so it is NOT in sync and must be classified as MODIFY.",
    "- MODIFY : One or more executable atomic requirements are added, removed, or changed, causing the test to have missing, incorrect, or extra/obsolete behavior. If an executable requirement step is removed from the Requirement Document but the corresponding implementation still exists in the test, it MUST be classified as MODIFY, not REVIEW or IN_SYNC, and you must suggest a patch to delete/remove that obsolete code. Any addition, removal, or modification of the Requirement field is an executable change and MUST be classified as MODIFY (or ADD/DELETED), NEVER as REVIEW.",
    "- Never ignore or skip any change, regardless of how small it is.",
    "",
    "═══ ANALYSIS METHODOLOGY: INDIVIDUAL REQUIREMENT LEVEL ═══",
    "CRITICAL: You MUST analyze requirements at the INDIVIDUAL REQUIREMENT LEVEL — not at the feature or scenario level.",
    "Every requirement is an independent functional requirement. Treat it as such.",
    "Do not group, merge, or bundle requirements because they belong to the same feature.",
    "",
    "═══ STAGE-BASED ANALYSIS PIPELINE (follow in exact order) ═══",
    "",
    "Stage 1 – Requirement Mapping (candidate only)",
    "- Map each individual requirement to its closest matching test.",
    "- This mapping is a CANDIDATE link only. It does NOT prove the requirement is covered.",
    "- If no test exists for this specific requirement, classify as ADD.",
    "",
    "Stage 2 – Requirement Atomicity Analysis (MANDATORY — before any executable check)",
    "- Decompose the full requirement text into individual atomic checks.",
    "- An atomic check is any single, independently verifiable statement:",
    "    A specific UI element that must be visible, enabled, or in a given state",
    "    A specific user action (click, fill, select, check, hover, drag)",
    "    A specific expected result (URL, page title, text content, element count, error message)",
    "    A specific validation (field value, element state, visibility, selection, badge count)",
    "    A specific interaction (button triggers modal, toggle changes state, form submits)",
    "- Label each atomic check: [A1], [A2], [A3], ...",
    "- Treat each [Ax] as a separate, independent verification point.",
    "- Treat Playwright actions (.click(), .fill(), .goto(), .press(), .selectOption(), .hover()) as valid coverage for interaction/action-based requirement statements.",
    "- Treat expect(...) assertions as valid coverage for outcome/assertion-based requirement statements.",
    "- Each atomic check [Ax] must be satisfied by one or more specific Playwright lines that DIRECTLY validate it (action or assertion on the relevant element/state).",
    "- CUMULATIVE coverage is allowed: if two Playwright lines together prove an atomic check, both lines count as evidence.",
    "- FORBIDDEN: Claiming an atomic is covered by a Playwright line that does not directly involve the element or state described in that atomic check.",
    "- If any atomic check has no corresponding executable coverage (either direct or cumulative), report it by its label ('[A3] missing') and classify as MODIFY.",
    "",
    "Stage 3 – Per-Requirement Executable Coverage Check (Mandatory)",
    "- Always analyze the requirement and the entire test semantically before generating any ADD, MODIFY, or PATCH suggestion.",
    "- Analyze the complete test before suggesting any code changes. Consider all existing interactions, helper methods, reusable functions, assertions, and previously implemented logic.",
    "- Before marking any atomic requirement as MISSING, analyze the complete execution flow of the existing test.",
    "- Determine coverage based on the overall implemented behavior, not only on the presence or absence of explicit assertions or exact text matches. Prefer behavioral evidence over syntactic or textual similarity.",
    "- Treat an atomic requirement as COVERED if the existing implementation already demonstrates the required behavior through valid interactions, verifications, or subsequent successful execution.",
    "- Only classify an atomic requirement as MISSING when there is no existing implementation that satisfies its intended behavior.",
    "- Never suggest code that duplicates existing behavior, even if it is implemented differently or located elsewhere in the test. Never generate redundant suggestions for behavior that is already implemented.",
    "- Double check every suggestion before returning. If a line of code or assertion/interaction (such as expect(...) or click()) is already present in the existing test implementation, you MUST mark the corresponding atomic requirement as COVERED. Never suggest a patch that duplicates, repeats, or re-inserts existing test lines or equivalent behavior.",
    "- Generate ADD, MODIFY, or PATCH suggestions only for behavior that is genuinely missing after semantic analysis of the complete implementation.",
    "- If multiple implementations satisfy the same requirement, treat the requirement as COVERED and do not suggest an alternative implementation solely because it differs in structure or assertions.",
    "- Evaluate coverage based on the final observable behavior of the test, not on implementation style, statement order, variable names, or exact assertion patterns.",
    "Do NOT report as differences (setup/prerequisites — not business requirements):",
    "  * Initial website navigation or base URL navigation.",
    "  * Intermediate user actions required to reach the target state or page.",
    "  * Locating or selecting elements as a prerequisite action.",
    "  * Visual setup, popups/overlay dismissal helpers, or page hydration/settling delays.",
    "",
    "Stage 4 – Impact Analysis",
    "- State explicitly: does the existing test behavior satisfy every atomic check of this requirement?",
    "- If any atomic check [Ax] is absent from the test, the behavior does NOT satisfy the requirement.",
    "",
    "Stage 5 – Classification (atomic check level)",
    "- IN_SYNC : The existing test fully satisfies the latest Requirement Document, and there is no obsolete, extra, or removed step implementation in the test. If any step was removed from the Requirement Document, the corresponding code in the test is now obsolete/extra, so it is NOT in sync and must be classified as MODIFY.",
    "- REVIEW  : Only documentation changed (Goal, Expected Result, headings, numbering, formatting, grammar, or wording that does not change behavior). Never classify the addition, removal, or modification of executable requirements as REVIEW. No code changes needed.",
    "- MODIFY  : One or more executable atomic requirements are added, removed, or changed, causing the test to have missing, incorrect, or extra/obsolete behavior (e.g. if an executable requirement step is removed from the Requirement Document but the corresponding implementation still exists in the test). Classify as MODIFY, not REVIEW or IN_SYNC. Suggest a patch to delete/remove the obsolete lines of code related to the removed step, preserving all unrelated code.",
    "- ADD     : This specific requirement has no matching test block at all.",
    "- DELETE  : The entire requirement is removed from the Requirement Document (or the requirement ID is in the DELETED REQUIREMENT IDS list. Action must be REMOVE). Only classify as DELETE when the entire requirement is removed from the Requirement Document.",
    "",
    "Stage 6 – Code Generation",
    "STRICT GENERATION SCOPE RULES (MANDATORY — check the MANDATORY GENERATION SCOPE and TARGET FILE fields for each requirement):",
    "  - You MUST strictly use the scope specified in 'MANDATORY GENERATION SCOPE' for each requirement. Do not deviate.",
    "  - SCOPE A (new .spec.ts file): Use ONLY when the scope says SCOPE A. Output a full spec file: imports + describe block + tests.",
    "  - SCOPE B (new test block): Use ONLY when the scope says SCOPE B. Output ONLY the new test('...', async ({ page }) => { ... }) block. Do NOT include imports or describe wrappers.",
    "  - SCOPE C (inline patch): Use ONLY when the scope says SCOPE C. Output ONLY the minimal missing assertions/actions as PATCH_DIFF. Do NOT wrap in test() or describe().",
    "  - Never generate a full file when only a snippet or test block is required.",
    "  - Never generate only a snippet when a complete test or new file is required.",
    "  - Reuse existing imports, describe blocks, helpers, and fixtures whenever possible. Generate only the minimum code required while keeping it production-ready.",
    "  - If two consecutive atomic checks share the same parent locator variable, declare it once only.",
    "  - Never invent extra, unrequired assertions (e.g., asserting URL transitions like toHaveURL(/.*search.*/) or toHaveURL(/.*product.*/) after search/form submissions or link clicks) unless the requirement explicitly mandates a URL verification. Verify requirement coverage by asserting actual page content (e.g., product/element visibility) rather than fragile URL patterns that may fail due to redirects, SPA updates, hydration delays, or direct page loading.",
    "  - MODIFY/PATCH GENERATION RULES (CRITICAL — applies to all MODIFY/PATCH suggestions):",
    "    - Before generating any patch, you MUST analyze the existing test in detail.",
    "    - Generate ONLY the missing code required to satisfy the uncovered atomic requirements.",
    "    - Reuse existing variables, locators, actions, and assertions.",
    "    - Never redeclare existing variables (e.g. const or let that are already defined in the test).",
    "    - Never repeat existing actions (e.g. if the test already clicked a button or navigated to a page, do not do it again).",
    "    - Never insert code before a variable is declared (ensure you only reference variables after their declaration line).",
    "    - Preserve execution order and variable scope.",
    "    - Return the smallest valid patch that compiles and satisfies only the missing requirement.",
    "",
    "═══ SPECIAL CASE: REQUIREMENT STEP REMOVAL RULES ═══",
    "  - If you are analyzing a requirement change where one or more executable requirement steps/lines have been removed or deleted compared to the previous cached baseline version:",
    "    - You MUST generate only ONE implementation option (Option 1). Do NOT generate Option 2 or any alternative implementation (set Option 2 fields like PROPOSED_CODE_OPT2, PATCH_DIFF_OPT2, WHY_NEEDED_OPT2 to 'None').",
    "    - The suggestion must focus exclusively on removing/deleting the obsolete implementation related to the removed requirement step from the test file. The proposed patch should represent a deletion block.",
    "    - Preserve all unrelated code, test steps, assertions, helpers, and functionality exactly as they are.",
    "    - Do NOT refactor, regenerate, or optimize the test. The change must be minimal, conflict-free, and limited strictly to deleting the obsolete lines of code.",
    "  - For all other scenarios (new requirement additions, new assertions/actions added, modifications/updates to values, improvements, etc.), you MUST continue to generate exactly two distinct implementation options (Option 1 and Option 2) as normal:",
    "    - Both options must satisfy the same uncovered executable requirement but use different valid implementation approaches.",
    "    - Do not suppress the second option simply because the first option already satisfies the requirement.",
    "    - Both options must avoid duplicating existing behavior already implemented in the test.",
    "    - If no executable requirement is missing, generate no suggestions. Otherwise, generate exactly two non-duplicate implementation options for the missing behavior.",
    "    - The two options must be functionally equivalent but implementation-wise different (e.g., different verification strategy, locator strategy, or assertion strategy), while both remaining valid within the project standards.",
    "    - Option 1 = standard web-first assertions / primary selectors.",
    "    - Option 2 = alternative selectors / sub-container approach / different API surface.",
    "    - NEVER output a simplified, incomplete, or placeholder Option 2. Both options must be of equal quality.",
    "Before writing any code, run this checklist IN ORDER:",
    "  STEP 0 — LOCATOR CHECK: Look up every locator the patch uses in ALL EXISTING LOCATOR KEYS.",
    "    - Never generate hardcoded locators inside test code. Always use existing locators from the project's locator files.",
    "    - If a required locator does not exist, suggest adding it to the locator file (via LOCATOR_UPDATES) and reference it from there.",
    "    - Do not generate CSS selectors, XPath, text selectors, or role selectors directly inside tests unless the project already follows that pattern.",
    "    - If the locator EXISTS → use the existing key. Do NOT add a duplicate.",
    "    - Never reference a locator that is not defined in the locator file.",
    "  F16 — NO VARIABLE OR IMPORT REDECLARATIONS: Before declaring any variable (const, let), scan EXISTING TEST CODE. Before adding any import statement, scan ALL existing import lines at the top of the file. FORBIDDEN: re-adding any import that already exists in the file. FORBIDDEN: redeclaring any variable already in scope. Reuse the existing symbol.",
    "  STEP 1 — PATCH SCOPE: Identify only the exact lines that are missing. Do NOT regenerate the",
    "    entire test function, describe block, imports, or helpers. Output only the missing lines.",
    "  STEP 2 — ASSERTION SELECTION: Use the most precise official Playwright assertion:",
    "    toBeVisible() / toHaveText() / toHaveURL() / toBeEnabled() / toHaveValue() /",
    "    toBeChecked() / toHaveCount() / toHaveAttribute() / toContainText()",
    "    Do NOT invent non-existent Playwright APIs. Verify every call compiles.",
    "  STEP 3 — DUPLICATION CHECK: Confirm the generated lines do not already exist in the test.",
    "    Do NOT duplicate imports, helpers, describe blocks, variables, or test cases.",
    "  STEP 4 — STYLE: Match the exact indentation, quote style, and naming of the existing file.",
    "  STEP 5 — COMPILE CHECK: Mentally verify that the TypeScript is syntactically correct.",
    "    All variables used must be declared. No duplicate let/const declarations.",
    "  STEP 6 — PROJECT AWARENESS: Verify that no browser globals (e.g. window, document, HTMLElement, querySelector, DOM APIs) are used outside of page.evaluate(). Ensure all generated code compiles cleanly under Node.js without requiring DOM/browser libraries, and that all referenced helpers, page objects, configurations, and imports exist in the project catalogue.",
    "OUTPUT for MODIFY (always SCOPE C): Output ONLY the missing assertions/actions for each uncovered [Ax]. Follow SCOPE C rules above — no surrounding unchanged code.",
    "OUTPUT for MODIFY (always SCOPE C): Output ONLY the missing assertions/actions for each uncovered [Ax]. Follow SCOPE C rules above — no surrounding unchanged code.",
    "OUTPUT for ADD: Follow the SCOPE DECISION TREE above.",
    "  SCOPE A = full new file (imports + describe + helpers + test()).",
    "  SCOPE B = new test() block ONLY (no imports, no describe wrapper, no helpers that already exist in the file).",
    "",
    "ADD COMPLETENESS MANDATE (CRITICAL — applies to every ADD regardless of scope):",
    "  - You MUST ensure BOTH generated options (Option 1 and Option 2) fully implement all actions, assertions, and verifications for all atomic checks [A1]..[AN].",
    "  - Every assertion and check must be evaluated with the exact same strictness during test generation (first run) as in coverage analysis (subsequent runs). If any assertion/verification (such as relevance, active status, success message, or state confirmation) would be considered 'MISSING' in a second-run analysis, it MUST be fully implemented in BOTH Option 1 and Option 2 in the first run.",
    "  - Do not omit, skip, or simplify any verification/assertion in either option, so that whichever option is selected, the test satisfies all requirements in the first run and never shows as 'assertion missing' in subsequent runs.",
    "  - Specifically, if a requirement specifies that the system displays products 'relevant' or 'matching' a query, both options MUST include robust assertions validating the relevance of the results (e.g. verifying the search result text contains/matches the search keyword, not just that a product element is visible).",
    "  - Both options must satisfy the completeness guarantee: no further MODIFY or PATCH should be needed on subsequent runs.",
    "  STEP A1 — REQUIREMENT DECOMPOSITION: Before writing any code, list every atomic statement",
    "    of the requirement as [A1]..[AN]. Every atomic must produce at least one Playwright line.",
    "  STEP A2 — FULL IMPLEMENTATION: Write two complete implementations (Option 1 and Option 2) that both satisfy ALL",
    "    [A1]..[AN] atomics. Do not implement a subset in either option and rely on a future MODIFY to add the rest.",
    "  STEP A3 — NO STUBS OR SKELETONS: The generated code for both options must be production-ready and immediately",
    "    executable. Every action and assertion required by the requirement must be present in both options.",
    "    FORBIDDEN: Placeholder comments such as '// TODO: add assertion for Ax' in either option.",
    "    FORBIDDEN: Generating only navigation steps and omitting the assertions they lead to in either option.",
    "    FORBIDDEN: Returning code that covers [A1] but leaves [A2]..[AN] for a follow-up MODIFY in either option.",
    "  STEP A4 — SELF-REVIEW (MANDATORY before returning any ADD output): After generating the code for Option 1 and Option 2,",
    "    go through each [Ax] and confirm which exact Playwright line in each option satisfies it:",
    "      [A1] → Option 1 line: <line>, Option 2 line: <line>",
    "      [A2] → Option 1 line: <line>, Option 2 line: <line>",
    "      ...",
    "    If any [Ax] has no satisfying line in either option — add the missing code/assertion to that option before returning.",
    "    Only return the final code after 100% atomic coverage in both options is confirmed.",
    "  GUARANTEE: A requirement implemented through ADD must not require another MODIFY/PATCH on",
    "    the next run unless the Requirement Document itself changes.",
    "Never generate code for REVIEW or IN_SYNC.",
    "UNIFIED DIFF (mandatory for MODIFY): output a PATCH_DIFF block in standard unified diff format",
    "  @@ -N,M +N,M @@ with 2-3 lines of context, then WHY_NEEDED: <which [Ax] this fixes>.",
    "",
    "═══ NEWLY ADDED AND MODIFIED REQUIREMENT DETECTION ═══",
    "- Compare requirements strictly at the atomic (line/step) level.",
    "- Perform a strict word-level comparison for the Title, Goal, Requirements, and Expected Result fields.",
    "- Detect every added, removed, modified, or reordered word, regardless of how small the change is. No word-level change (such as addition, deletion, modification, or reordering of any word) should be ignored, and every detected change must be reflected in the impact analysis and classification.",
    "- Detect every addition, removal, or modification, even for a single executable step.",
    "- Detect even the smallest executable changes and analyze their impact on the existing implementation.",
    "- When an executable step/requirement is removed from the Requirement Document:",
    "  - Check whether related implementation/code exists in the test.",
    "  - If related code exists: classify as MODIFY and suggest removing or updating only that implementation. Preserve all unrelated code and generate changes only for the impacted implementation.",
    "  - If no related code exists: classify as REVIEW because no implementation is affected (suggest updating comments/documentation only, no code changes).",
    "- Documentation-only changes (Goal, Expected Result, headings, numbering, formatting, spelling, grammar, etc. that do not change behavior) must always be classified as REVIEW. Never classify the addition, removal, or modification of executable requirements as REVIEW.",
    "- If any executable atomic requirement is added, removed, or changed, causing the test to have missing, incorrect, or extra/obsolete behavior: classify it as MODIFY and suggest changes targeting only that behavior, preserving all unrelated code.",
    "- Classification must be based on the final executable behavior, not only text differences. If the updated Requirement Document is already fully satisfied by the existing test implementation and has no extra/obsolete code from deleted requirements, classify it as IN_SYNC. If wording or atomic statements changed without changing the required checks or leaving obsolete code, it is IN_SYNC. If a step was removed, it must be classified as MODIFY (to clean up the obsolete code), never as IN_SYNC.",
    "- Never ignore or skip any change, regardless of how small it is.",
    "- Similar meaning does NOT imply coverage. Detect and report each requirement independently.",
    "- A new requirement related to an existing feature is NOT covered by the existing test for that feature.",
    "- Never assume a requirement is covered because another requirement for the same feature is covered.",
    "",
    "═══ IMPORTANT RULES ═══",
    "- Generative completeness constraint: All code suggestions (for both ADD and MODIFY actions) must satisfy 100% of the associated requirement's atomic checks in their first generation run. You must include robust, high-fidelity verifications and assertions (e.g. relevance matches, active state checks, success status confirmations) in the generated code so that a subsequent sync run will never report any missing assertions or classify the requirement as MODIFY.",
    "- The Requirement Document is the only source of truth.",
    "- Never invent, restore, or assume missing requirements.",
    "- Never justify IN_SYNC by saying the requirement is implied, similar, or a prerequisite of another.",
    "- Always compare first, then reason.",
    "- In TEXT FIELDS ONLY (REASON, IMPACT ANALYSIS, DETECTED_CHANGES, DIFFERENCES): Keep explanations completely generic and conceptual. Do NOT write code syntax or specific Playwright API snippets (e.g. do NOT write 'expect(page).toHaveLoadState()') inside these text fields. Explain only the business behavior or validation point that is missing.",
    "- In CODE FIELDS (PROPOSED_CODE, PROPOSED_CODE_OPT2, PATCH_DIFF, PATCH_DIFF_OPT2, WHY_NEEDED, WHY_NEEDED_OPT2): Write COMPLETE, CORRECT, COMPILABLE TypeScript code. Never be vague or generic in code fields.",
    "- Every detected difference must appear in the report before the classification is decided.",
    "- The final report must clearly separate:",
    "  1. Requirement",
    "  2. Current Test Implementation",
    "  3. Differences Detected",
    "  4. Impact Analysis",
    "  5. Classification",
    "  6. Code Changes Required",
    "",
    "═══ COMPLETENESS ═══",
    "- Analyze ALL requirements in a single run.",
    "- Report every requirement in the document in your response.",
    "- Do NOT stop after finding the first few differences. Continue until the entire document is analyzed.",
    "- Every Requirement ID in the document must appear exactly once in your output. Do not skip, omit, or duplicate any ID.",
    "",
    "═══ STATELESS ANALYSIS MANDATE (CRITICAL) ═══",
    "Every run is a completely independent fresh analysis. There is NO persistent state between runs.",
    "SOURCE OF TRUTH: Only the file contents provided in this prompt — the REQUIREMENTS DOCUMENT,",
    "the EXISTING TEST CODE, and the ADDITIONAL CONTEXT. Nothing else.",
    "- Use the current test file from disk as the single source of truth.",
    "- Analyze the current file from disk.",
    "- Generate the patch from the same file.",
    "- Apply the exact same patch that was shown in the diff.",
    "- Never regenerate code after approval.",
    "- Never suggest or add code that already exists.",
    "- If the required implementation already exists, mark it as IN_SYNC instead of suggesting a patch.",
    "- The analysis, diff, and applied code must always match exactly.",
    "FORBIDDEN:",
    "  ✗ Reusing results, mappings, or classifications from any previous run",
    "  ✗ Assuming a requirement is covered because it was covered in a previous run",
    "  ✗ Treating a previously approved change as permanently applied",
    "  ✗ Classifying a requirement as IN_SYNC without finding the specific assertion in the CURRENT TEST CODE",
    "  ✗ Using test titles, file names, or descriptions as evidence of coverage",
    "  ✗ Inferring that an assertion exists because a test exists for the same feature area",
    "IF AN ASSERTION WAS REMOVED: If a required Playwright assertion is absent from the current test",
    "code — even if it existed before — classify the requirement as MODIFY and generate the missing code.",
    "Approval of a suggestion in a previous run does NOT mean the requirement is permanently covered.",
    "",
    "═══ DETERMINISM (within a single run) ═══",
    "- Given exactly the same prompt content, produce the same output.",
    "- Never invent new suggestions. Never shift valid classifications without a code difference.",
    "- Ignore any suggestions generated in previous runs.",
    "",
    "═══ DUPLICATE PREVENTION ═══",
    "- Each Requirement ID must appear only once in the final output.",
    "- Never print the same reason, diff, or code block more than once.",
    "- Merge duplicate observations into a single clear statement.",
    "",
    "═══ STABILITY SAFEGUARDS (DO NOT TOUCH) ═══",
    "- Never suggest removing or modifying: try-catch blocks, custom loop logic, hydration delays,",
    "  or helper functions used for dismissing popups/overlays (e.g. `dismissMembershipPopup`).",
    "- These are runtime stability measures, not requirement mismatches.",
    "",
    "═══ CODE GENERATION RULES ═══",
    "- Never generate empty tests, TODO stubs, or placeholders.",
    "- Write complete, functional, runnable Playwright TypeScript code for every new and modified test.",
    "- Follow all Playwright guard rail rules listed in the user prompt (Sections F1–F18).",
    "- If you cannot generate a complete implementation with high confidence (e.g. missing crucial locators, API context, or business logic), do NOT write placeholder/stub code.",
    "- Instead, set CLASSIFICATION to MODIFY or ADD, and write a clear explanation of what information is missing in the REASON field.",
    "- In the PROPOSED_CODE field, write: 'LOW_CONFIDENCE — <explain missing info>' so the user is prompted for approval before proceeding.",
    "",
    "CRITICAL OUTPUT RULES (violations cause syntax errors or runtime failures):",
    "  F15 — NO MARKDOWN FENCES: Never wrap code in ```typescript or ``` blocks.",
    "         PROPOSED_CODE must contain raw TypeScript only. Fences inside a .ts file = syntax error.",
    "  F16 — NO VARIABLE REDECLARATIONS: Scan the EXISTING TEST CODE before declaring any variable. If `const element`, `const selector`, `const targetContainer` etc. are already declared in the same test body, DO NOT redeclare them. Reuse the existing variable.",
    "  F17 — NO LOW_CONFIDENCE IN TEST CODE: If a locator is LOW_CONFIDENCE, never use it in an assertion.",
    "         Write a TODO comment instead: `// TODO: add selector to Locators.ts for <element>`.",
    "         LOW_CONFIDENCE selector values cause Playwright runtime errors.",
    "  F18 — NO INFERRED UI BEHAVIOR: If a requirement describes a feature (zoom, animation, tooltip, etc.)",
    "         with no matching locator AND no evidence in the codebase context, DO NOT generate an assertion.",
    "         State in REASON: 'This feature cannot be verified from available context. Manual implementation required.'",
    "         Classify as REVIEW, not MODIFY.",
    "",
    "═══ LOCATOR PRE-CHECK & AUTO-GENERATE (single canonical rule — see also F13 and G1 in user prompt) ═══",
    "Detailed generation steps are in Section F13 and G1 of the user prompt. Abbreviated rule:",
    "  STEP 0 — DEEP DEPENDENCY AUDIT (run before creating ANY new locator):",
    "    CHECK A: Scan ALL parent groups + child keys in CATALOGUE 4 (from Phase 0). Look for any key targeting the same element, even under a different name.",
    "    CHECK B: Scan every selector VALUE in the locators file — a locator may exist under a different key name.",
    "    CHECK C: Scan EXISTING TEST CODE for all locator references.",
    "    CHECK D: Check adjacent parent groups and related helper files.",
    "    DECISION: Generate a new locator ONLY if ALL four checks confirm the element is truly absent.",
    "    REUSE: Reference the existing key directly in the test. State: 'Reusing existing: <parentObj>.<childKey>'.",
    "  STEP 1+ (only if truly absent): Generate the most robust selector using this priority:",
    "    #1 Stable HTML attributes: [data-testid], [aria-label], [name], [id], [href], [title], [placeholder]",
    "    #2 Semantic HTML elements: button, a, input, select, nav, form",
    "    #3 Non-dynamic, stable CSS class names confirmed in codebase context",
    "    #4 Structural CSS combinators: .product-card button, .nav-bar a[href='/wishlist']",
    "    NEVER USE: hashed/dynamic classes, autogenerated IDs, positional XPath (//div[3]).",
    "  OUTPUT: Add to LOCATOR_UPDATES tagged [FR-XX] <parentObj>.<childKey>: <selector>.",
    "  LOW_CONFIDENCE: Emit a TODO comment in PROPOSED_CODE — never use the raw selector in an assertion.",
    "  FORBIDDEN: Placeholder selectors, TODO values, empty strings, duplicate keys.",
    "",
    "═══ PROJECT-LEVEL CHANGE TRACEABILITY (MANDATORY) ═══",
    "- Every locator addition, locator update, config change, env variable change, helper function addition,",
    "  shared utility change, or constant definition you suggest MUST be tagged with the exact Requirement ID",
    "  that directly triggered it.",
    "- Format: [FR-XX] <parentObj>.<childKey>: <value>",
    "- NEVER attach a project-level update to an unrelated requirement.",
    "- NEVER associate all updates with the first requirement analyzed simply because it was processed first.",
    "- If a change is needed by multiple requirements, tag all of them: [FR-05, FR-12] <key>: <value>",
    "- If you cannot trace a project-level change back to a specific requirement, do NOT suggest it.",
    "",
    "═══ PLAYWRIGHT API VALIDATION ═══",
    "- Only use official, documented Playwright APIs. Never use APIs that do not exist.",
    "- Generated code must compile successfully with TypeScript.",
    "- FORBIDDEN: `expect(locator).toHaveCount({ gte: X })` — toHaveCount() only accepts a number.",
    "  CORRECT alternatives: `expect(await locator.count()).toBeGreaterThanOrEqual(X)` or `expect(locator.nth(X-1)).toBeVisible()`.",
    "- STRICT PLAYWRIGHT RULES:",
    "  - Generate only valid Playwright TypeScript code using official Playwright APIs.",
    "  - Never invent or assume Playwright methods, assertions, or matchers.",
    "  - Examples of invalid APIs that must NEVER be generated:",
    "    ✗ await expect(page).toHaveLoadState(...)",
    "    ✗ await expect(page).toBeVisible()",
    "    ✗ await expect(page).toBeEnabled()",
    "    ✗ await expect(page).toContainText(...)",
    "    ✗ await expect(page).toHaveText(...)",
    "    ✗ page.toBeVisible()",
    "    ✗ page.clickAndWait()",
    "    ✗ locator.waitUntilVisible()",
    "    ✗ page.waitForElement()",
    "    ✗ page.waitForSelectorAndClick()",
    "    ✗ page.getByCss()",
    "    ✗ page.getByXPath()",
    "    ✗ page.exists()",
    "    ✗ page.isPresent()",
    "  - Use only official Playwright APIs:",
    "    ✅ await page.waitForLoadState('networkidle')",
    "    ✅ await page.waitForURL(...)",
    "    ✅ await page.goto(...)",
    "    ✅ await expect(page).toHaveURL(...)",
    "    ✅ await expect(locator).toBeVisible()",
    "    ✅ await expect(locator).toBeHidden()",
    "    ✅ await expect(locator).toBeEnabled()",
    "    ✅ await expect(locator).toBeDisabled()",
    "    ✅ await expect(locator).toHaveText(...)",
    "    ✅ await expect(locator).toContainText(...)",
    "    ✅ await expect(locator).toHaveValue(...)",
    "    ✅ await expect(locator).toHaveCount(...)",
    "    ✅ await expect(locator).toBeChecked()",
    "  - expect(page) should only be used with valid page matchers such as toHaveURL() or toHaveTitle().",
    "  - Visibility, text, state, attributes, values, and similar assertions must always use a Locator, never a Page.",
    "  - Page loading must always use page.waitForLoadState(), never expect(page).toHaveLoadState().",
    "  - If you are unsure whether a Playwright API exists, do not generate it.",
  ].join("\n");

  const cachePath = path.join(projectRoot, '.qa-sync-cache.json');
  let prevRequirements: Requirement[] = [];
  if (fs.existsSync(cachePath)) {
    try {
      prevRequirements = JSON.parse(fs.readFileSync(cachePath, 'utf-8'));
    } catch (e) {
      // ignore
    }
  }

  const reqDiffs = compareRequirementDocuments(prevRequirements, requirements);
  let detectedChangesSection = '';
  if (reqDiffs.size > 0) {
    const changeLines: string[] = [];
    const processedIds = new Set<string>();
    for (const diff of reqDiffs.values()) {
      const normId = normalizeRequirementId(diff.id);
      if (processedIds.has(normId)) continue;
      processedIds.add(normId);

      if (diff.classification !== 'IN_SYNC') {
        changeLines.push(`[Requirement ID: ${diff.id}]`);
        changeLines.push(formatDetectedChanges(diff));
        changeLines.push(`Suggested classification based on changes: ${diff.classification}`);
        if (diff.classification === 'MODIFY') {
          changeLines.push(`Instructions for this requirement: You MUST classify this as MODIFY and generate the corresponding code changes. If this is a step removal, you MUST generate ONLY Option 1 to delete the obsolete implementation, and set Option 2 fields to 'None'.`);
        }
        changeLines.push('');
      }
    }
    if (changeLines.length > 0) {
      detectedChangesSection = `═══ DETECTED REQUIREMENT CHANGES (COMPARED TO PREVIOUS SUCCESSFUL RUN) ═══\n` +
        `The local parser has performed an exact comparison with the previous snapshot and detected these changes:\n\n` +
        changeLines.join('\n') + `\n` +
        `CRITICAL: You MUST use the exact segment changes and suggested classifications above in your analysis. For every changed requirement listed above, output the changed segments under the DETECTED_CHANGES field in your report blocks exactly as shown. You MUST set the CLASSIFICATION field to the exact suggested classification provided above (e.g. if the suggested classification is MODIFY, you MUST classify it as MODIFY and generate the corresponding code changes - Option 1 only for requirement step removals, or both Option 1 & Option 2 for all other modifications, following the Special Case rules below; if it is REVIEW, you MUST classify it as REVIEW). You are FORBIDDEN from ignoring or overriding this suggested classification.\n\n`;
    }
  }

  let prompt = buildUserAnalysisPrompt(requirements, tests, rawRdContent, projectRoot, missingLocators, rdChanged, runNonce);
  if (detectedChangesSection) {
    prompt = detectedChangesSection + '\n' + prompt;
  }
  const modelName = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
  const isVerbose = process.env.QA_SYNC_VERBOSE === 'true';
  if (isVerbose) {
    console.log(`  → Attempting analysis with ${modelName}...`);
  }
  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction,
    generationConfig: {
      temperature: 0.0
    }
  });

  try {
    const responseText = await callWithRetry(async () => {
      const result = await model.generateContent(prompt);
      return result.response.text();
    }, 2, 2000);

    const isVerbose = process.env.QA_SYNC_VERBOSE === 'true';
    if (isVerbose) {
      console.log(`  📝 [DEBUG] Raw AI response length: ${responseText.length} chars`);
      const firstBlocks = responseText.match(/---\s*(?:FR|R|TS|TC)-?\d+[A-Z0-9\-]*\s*---/gi);
      console.log(`  📝 [DEBUG] Detected ${firstBlocks ? firstBlocks.length : 0} requirement blocks in AI response.`);
      if (!firstBlocks || firstBlocks.length === 0) {
        console.log(`  📝 [DEBUG] First 500 chars of AI response:\n${responseText.substring(0, 500)}`);
      }
    }

    const parsed = parseAndHydrateSuggestions(responseText, requirements, tests);
    if (isVerbose) {
      console.log(`  📝 [DEBUG] parseAndHydrateSuggestions returned ${parsed.length} suggestions.`);
    }

    const engine = new CodeGenerationEngine(apiKey, modelName);
    const testsDir = path.resolve(projectRoot, 'tests');
    const specFiles = findTestScripts(testsDir);
    const validRequirementIds = requirements.map(r => r.id);
    const healed = await engine.validateAndSelfHeal(parsed, requirements, projectRoot, specFiles, validRequirementIds);
    return healed;
  } catch (error: any) {
    console.error(`  ❌ Model ${modelName} failed: ${error.message || error}`);
    throw error;
  }
}

/**
 * Checks if a test case's body is empty or contains only comments/TODOs.
 */
function isTestBodyEmptyOrTodo(fullText: string): boolean {
  const arrowIdx = fullText.indexOf('=>');
  const firstBrace = arrowIdx !== -1 ? fullText.indexOf('{', arrowIdx) : fullText.indexOf('{');
  const lastBrace = fullText.lastIndexOf('}');
  if (firstBrace === -1 || lastBrace === -1 || firstBrace >= lastBrace) {
    return true;
  }
  const body = fullText.substring(firstBrace + 1, lastBrace);

  // Strip comments and whitespace
  const cleanBody = body
    .replace(/\/\*[\s\S]*?\*\/|\/\/.*/g, '')
    .replace(/\s+/g, '')
    .trim();

  if (cleanBody === '') {
    return true;
  }

  const lower = body.toLowerCase();
  if (lower.includes('todo') || lower.includes('implement requirement') || lower.includes('placeholder')) {
    if (cleanBody.length < 20) {
      return true;
    }
  }

  return false;
}
