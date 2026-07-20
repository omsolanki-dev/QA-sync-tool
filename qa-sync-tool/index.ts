import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';
import { chromium } from '@playwright/test';
import { findRequirementDocument, extractRequirements, findTestScripts, scanTestFiles, parseTestFileStructure, reconstructTestFile, FileSegment, resolveLocatorsPathForFile, normalizeRequirementId, findConfigFilePath, findLocatorsFilePath, getMissingLocators, Requirement, applyChanges, cleanProposedCode, ensureRequirementComment, updateConfigFile, updateLocatorsFile, updateEnvFile, findExternalDependencies, healSpecImports, Suggestion } from './testScanner';
import { analyzeCoverage, MappingError, currFingerprintToPrevFingerprintMap, currIdToPrevIdMap, compareRequirementDocuments } from './aiAnalyzer';

// Custom ANSI colors for sleek terminal output
const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const ORANGE = '\x1b[38;5;208m';
const DIM = '\x1b[2m';

/**
 * Prints a colored diff between original and proposed code.
 * - Green  (+) lines = proposed / new code
 * - Red    (-) lines = original / removed code
 * If only one side is provided, prints that side with its appropriate prefix.
 */
function printDiff(original?: string, proposed?: string, maxLines = 60): void {
  const DIFF_HEADER = `\x1b[2m${'─'.repeat(56)}${RESET}`;
  if (original && proposed) {
    console.log(`${BOLD}  📄 Code Diff:${RESET}`);
    console.log(DIFF_HEADER);
    const origLines = original.trim().split('\n');
    const propLines = proposed.trim().split('\n');
    const printLines = (lines: string[], color: string, limit: number) => {
      const shown = lines.slice(0, limit);
      shown.forEach(l => console.log(`${color}    ${l}${RESET}`));
      if (lines.length > limit) {
        console.log(`${DIM}  ... ${lines.length - limit} more lines hidden${RESET}`);
      }
    };
    printLines(origLines, RED, maxLines);
    console.log(DIFF_HEADER);
    printLines(propLines, GREEN, maxLines);
    console.log(DIFF_HEADER);
  } else if (original) {
    console.log(`${BOLD}  🔧 Suggested Patch (Removal):${RESET}`);
    console.log(DIFF_HEADER);
    const origLines = original.trim().split('\n').slice(0, maxLines);
    origLines.forEach(l => console.log(`${RED}    ${l}${RESET}`));
    if (original.trim().split('\n').length > maxLines) {
      console.log(`${DIM}  ... more lines hidden${RESET}`);
    }
    console.log(DIFF_HEADER);
  } else if (proposed) {
    console.log(`${BOLD}  ✨ New Code to Add:${RESET}`);
    console.log(DIFF_HEADER);
    const propLines = proposed.trim().split('\n').slice(0, maxLines);
    propLines.forEach(l => console.log(`${GREEN}    ${l}${RESET}`));
    if (proposed.trim().split('\n').length > maxLines) {
      console.log(`${DIM}  ... more lines hidden${RESET}`);
    }
    console.log(DIFF_HEADER);
  }
}

/**
 * Renders a unified-diff string (patchDiff) as clean TypeScript code.
 * Rules:
 *   - @@ hunk headers are skipped entirely — no patch metadata shown.
 *   - Lines that were additions (originally prefixed '+') → shown in GREEN, no prefix.
 *   - Lines that were removals (originally prefixed '-') → shown in RED, no prefix.
 *   - Context lines (originally prefixed ' ') → shown DIM, no prefix.
 *   - The displayed code is exactly the TypeScript that will be applied.
 */
function printCleanCodeDiff(patchDiff: string, maxLines = 80): void {
  const DIFF_HEADER = `${DIM}${'─'.repeat(56)}${RESET}`;
  console.log(DIFF_HEADER);
  let shown = 0;
  for (const rawLine of patchDiff.split('\n')) {
    // Skip @@ hunk headers and diff file headers (--- / +++ paths)
    if (rawLine.startsWith('@@') || rawLine.startsWith('--- ') || rawLine.startsWith('+++ ')) continue;

    if (shown >= maxLines) {
      console.log(`${DIM}  ... more lines hidden${RESET}`);
      break;
    }

    if (rawLine.startsWith('+')) {
      // Added line — strip the leading '+', display in green
      console.log(`${GREEN}    ${rawLine.slice(1)}${RESET}`);
    } else if (rawLine.startsWith('-')) {
      // Removed line — strip the leading '-', display in red
      console.log(`${RED}    ${rawLine.slice(1)}${RESET}`);
    } else {
      // Context line — strip the leading ' ' if present, display dimmed
      const content = rawLine.startsWith(' ') ? rawLine.slice(1) : rawLine;
      console.log(`${DIM}    ${content}${RESET}`);
    }
    shown++;
  }
  console.log(DIFF_HEADER);
}

function deduplicateSuggestions(suggestions: Suggestion[]): Suggestion[] {
  const seen = new Map<string, Suggestion>();
  for (const sug of suggestions) {
    const normalizedPath = sug.filePath ? path.normalize(sug.filePath).toLowerCase().replace(/\\/g, '/') : 'no_file';
    const key = `${normalizeRequirementId(sug.requirementId)}::${normalizedPath}::${sug.action}`;

    if (!seen.has(key)) {
      seen.set(key, { ...sug });
    } else {
      const existing = seen.get(key)!;
      const lines = new Set<string>();
      if (existing.description) {
        existing.description.split('\n').forEach(l => lines.add(l.trim()));
      }
      if (sug.description) {
        sug.description.split('\n').forEach(l => lines.add(l.trim()));
      }
      existing.description = Array.from(lines).join('\n');

      if (!existing.proposedCode && sug.proposedCode) {
        existing.proposedCode = sug.proposedCode;
      }
      if (!existing.originalCode && sug.originalCode) {
        existing.originalCode = sug.originalCode;
      }

      if (sug.configUpdates && sug.configUpdates.length > 0) {
        existing.configUpdates = existing.configUpdates || [];
        for (const item of sug.configUpdates) {
          if (!existing.configUpdates.some(u => u.key === item.key)) {
            existing.configUpdates.push(item);
          }
        }
      }
      if (sug.locatorUpdates && sug.locatorUpdates.length > 0) {
        existing.locatorUpdates = existing.locatorUpdates || [];
        for (const item of sug.locatorUpdates) {
          if (!existing.locatorUpdates.some(u => u.key === item.key)) {
            existing.locatorUpdates.push(item);
          }
        }
      }
      if (sug.envUpdates && sug.envUpdates.length > 0) {
        existing.envUpdates = existing.envUpdates || [];
        for (const item of sug.envUpdates) {
          if (!existing.envUpdates.some(u => u.key === item.key)) {
            existing.envUpdates.push(item);
          }
        }
      }
    }
  }
  return Array.from(seen.values());
}

/**
 * Upserts (inserts or replaces) a list of requirements into the cache file.
 * Matching is done by stable requirement fingerprint (taking edits and renumbering into account).
 * Requirements already in the cache but NOT in `reqs` are left untouched.
 * Never overwrites the entire cache — only touches the specified entries.
 */
function upsertRequirementsInCache(projectRoot: string, reqs: Requirement[]): void {
  const cachePath = path.join(projectRoot, '.qa-sync-cache.json');
  let cacheRequirements: Requirement[] = [];
  if (fs.existsSync(cachePath)) {
    try {
      const content = fs.readFileSync(cachePath, 'utf-8').trim();
      if (content) {
        cacheRequirements = JSON.parse(content);
      }
    } catch (e) {
      // ignore parse error — start fresh
    }
  }

  const normLower = (s?: string): string =>
    (s || '')
      .replace(/\\'/g, "'")
      .replace(/\\"/g, '"')
      .replace(/\\`/g, '`')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();

  const getCleanTitle = (title: string): string => {
    return normLower(title).replace(/^(fr|ts|tc|r)-?\d+\s*[:\-]?\s*/, '');
  };

  // Deduplicate existing cacheRequirements using multi-signal logical checks to prevent duplicates
  const uniqueCacheReqs: Requirement[] = [];
  for (const r of cacheRequirements) {
    const rCleanTitle = getCleanTitle(r.title);
    const rNormId = normalizeRequirementId(r.id);
    const duplicateIdx = uniqueCacheReqs.findIndex(ur => 
      ur.fingerprint === r.fingerprint ||
      normalizeRequirementId(ur.id) === rNormId ||
      (rCleanTitle && getCleanTitle(ur.title) === rCleanTitle)
    );
    if (duplicateIdx !== -1) {
      uniqueCacheReqs[duplicateIdx] = r;
    } else {
      uniqueCacheReqs.push(r);
    }
  }
  cacheRequirements = uniqueCacheReqs;

  for (const req of reqs) {
    const prevFingerprint = currFingerprintToPrevFingerprintMap.get(req.fingerprint);
    const prevId = currIdToPrevIdMap.get(normalizeRequirementId(req.id));
    const reqCleanTitle = getCleanTitle(req.title);
    const normReqId = normalizeRequirementId(req.id);

    const idx = cacheRequirements.findIndex(r => {
      // 1. Exact fingerprint match
      if (r.fingerprint === req.fingerprint) return true;

      // 2. Mapped previous fingerprint match
      if (prevFingerprint && r.fingerprint === prevFingerprint) return true;

      // 3. Mapped previous ID match
      if (prevId && normalizeRequirementId(r.id) === prevId) return true;

      // 4. Exact normalized Requirement ID match
      if (normalizeRequirementId(r.id) === normReqId) return true;

      // 5. Clean title match
      if (reqCleanTitle && getCleanTitle(r.title) === reqCleanTitle) return true;

      return false;
    });

    if (idx !== -1) {
      // Overwrite the existing entry with the new one
      cacheRequirements[idx] = req;
    } else {
      // Genuinely new requirement — append it.
      cacheRequirements.push(req);
    }
  }

  try {
    fs.writeFileSync(cachePath, JSON.stringify(cacheRequirements, null, 2), 'utf-8');
  } catch (e) {
    // ignore write error
  }
}

/**
 * Removes a single requirement from the cache by fingerprint or requirement ID.
 * Called only when a REMOVE suggestion is approved and the test is actually deleted.
 */
function removeRequirementFromCache(projectRoot: string, fingerprint: string, requirementId?: string): void {
  const cachePath = path.join(projectRoot, '.qa-sync-cache.json');
  if (!fs.existsSync(cachePath)) return;
  try {
    const content = fs.readFileSync(cachePath, 'utf-8').trim();
    if (!content) return;
    let reqs: Requirement[] = JSON.parse(content);
    if (requirementId) {
      const normId = normalizeRequirementId(requirementId);
      reqs = reqs.filter(r => normalizeRequirementId(r.id) !== normId);
    } else {
      reqs = reqs.filter(r => r.fingerprint !== fingerprint);
    }
    fs.writeFileSync(cachePath, JSON.stringify(reqs, null, 2), 'utf-8');
  } catch (e) {
    // ignore
  }
}

/**
 * Updates the cache for a single approved suggestion — called ONLY after code changes
 * are successfully applied to disk (never before, never on skip/reject).
 *
 * - ADD / MODIFY: upserts the current requirement entry by fingerprint.
 * - REMOVE: deletes the orphaned requirement from the cache by requirement ID so it
 *   no longer appears as a baseline on the next run.
 */
function updateCacheForApprovedSuggestion(
  projectRoot: string,
  sug: Suggestion,
  currentRequirements: Requirement[],
  prevRequirements: Requirement[]
): void {
  if (sug.action === 'REMOVE') {
    // The orphaned test was deleted. Remove the stale baseline entry from cache by requirement ID.
    removeRequirementFromCache(projectRoot, '', sug.requirementId);
    return;
  }

  const normId = normalizeRequirementId(sug.requirementId);
  const currentReq = currentRequirements.find(r => normalizeRequirementId(r.id) === normId);
  if (currentReq) {
    upsertRequirementsInCache(projectRoot, [currentReq]);
  }
}


async function applyAndVerifySuggestion(
  projectRoot: string,
  sug: Suggestion,
  specFiles: string[],
  validRequirementIds: string[],
  requirements: Requirement[],
  prevRequirements: Requirement[],
  askQuestion: (query: string) => Promise<string>,
  skipValidation = false
): Promise<boolean> {
  // 1. Low confidence check
  const isLowConfidence = sug.proposedCode?.includes('LOW_CONFIDENCE') || sug.reason?.includes('LOW_CONFIDENCE') || sug.whyNeeded?.includes('LOW_CONFIDENCE');
  if (isLowConfidence) {
    console.log(`\n${YELLOW}${BOLD}⚠️  Low confidence suggestion for ${sug.requirementId}:${RESET}`);
    console.log(`Reason: ${sug.reason || 'Not specified'}`);
    const proceed = await askQuestion(`The implementation is partial or low confidence. Proceed anyway? (Y/N) → `);
    if (proceed !== 'y') {
      console.log(`${DIM}Skipped applying low confidence suggestion.${RESET}`);
      return false;
    }
  }

  // 2. External dependency check
  if (sug.proposedCode) {
    const importRegex = /import\s+[\s\S]*?from\s+(['"`]).*?\1\s*;?/g;
    const importsList: string[] = [];
    let importMatch;
    while ((importMatch = importRegex.exec(sug.proposedCode)) !== null) {
      importsList.push(importMatch[0]);
    }
    const externalDeps = findExternalDependencies(importsList);
    if (externalDeps.length > 0) {
      console.log(`\n${YELLOW}${BOLD}⚠️  External dependencies required for ${sug.requirementId}:${RESET}`);
      externalDeps.forEach(d => console.log(`   - ${d}`));
      const answer = await askQuestion(`Do you want to install these dependencies? (Y/N) → `);
      if (answer !== 'y') {
        console.log(`${RED}Aborting application of ${sug.requirementId} due to skipped dependencies.${RESET}`);
        return false;
      }
      console.log(`Installing dependencies: ${externalDeps.join(' ')}...`);
      try {
        const { execSync } = require('child_process');
        execSync(`npm install ${externalDeps.join(' ')}`, { stdio: 'inherit', cwd: projectRoot });
      } catch (e) {
        console.error(`❌ Failed to install dependencies:`, e);
        return false;
      }
    }
  }

  // 3. Apply changes
  try {
    applyChanges(projectRoot, [sug], specFiles, validRequirementIds);
    if (sug.action === 'REMOVE') {
      try {
        updateCacheForApprovedSuggestion(projectRoot, sug, requirements, prevRequirements);
      } catch (cacheError) {
        console.error(`⚠️ Failed to update cache for suggestion ${sug.requirementId}:`, cacheError);
      }
      return true;
    }
  } catch (error) {
    console.error(`❌ Failed to apply changes for suggestion ${sug.requirementId}:`, error);
    return false;
  }



  if (sug.codeChangesRequired === 'No') {
    console.log(`\n📝 Documentation-only change detected for ${sug.requirementId}. Skipping compilation and Playwright verification.\n`);
    return true;
  }

  if (skipValidation) {
    return true;
  }

  // 4. Validate TypeScript compilation
  console.log(`🔍 Validating TypeScript compilation for ${sug.requirementId}...`);
  let compiles = false;
  try {
    const { execSync } = require('child_process');
    execSync('npx tsc --noEmit', { stdio: 'pipe', cwd: projectRoot });
    compiles = true;
  } catch (e: any) {
    const errorOutput = ((e.stdout || '') + (e.stderr || '')).toString().trim();
    console.log(`\n${RED}${BOLD}❌ TypeScript compilation failed for ${sug.requirementId}:${RESET}`);
    if (errorOutput) {
      // Print each compiler error line so the user sees exactly what went wrong
      for (const line of errorOutput.split('\n').slice(0, 30)) {
        console.log(`   ${DIM}${line}${RESET}`);
      }
    }
    console.log(`${YELLOW}⚠️  Compilation failed — still running tests to show full picture.${RESET}\n`);
  }

  // 5. Playwright test verification — always run, even if compilation failed,
  //    so the user sees the full error picture before continuing to the next suggestion.
  console.log(`\n🧪 Verifying changes for ${sug.requirementId}...`);

  let testCmd = 'npx playwright test';
  if (sug.action === 'ADD' || sug.action === 'MODIFY') {
    if (sug.filePath) {
      const normalizedPath = sug.filePath.replace(/\\/g, '/');
      if (sug.testTitle) {
        const escapedTitle = sug.testTitle.replace(/["]/g, '.');
        testCmd = `npx playwright test "${normalizedPath}" -g "${escapedTitle}"`;
      } else {
        testCmd = `npx playwright test "${normalizedPath}"`;
      }
    }
  }

  console.log(`Running verification command: ${testCmd}`);

  let testsPass = false;
  try {
    const { execSync } = require('child_process');
    execSync(testCmd, { stdio: 'inherit', cwd: projectRoot });
    testsPass = true;
  } catch (error: any) {
    // Test output already printed via stdio: 'inherit' above
  }

  const success = compiles && testsPass;

  if (success) {
    console.log(`\n${GREEN}${BOLD}✅ Verification Successful for ${sug.requirementId}!${RESET}\n`);
  } else if (!compiles && !testsPass) {
    console.log(`\n${RED}${BOLD}❌ Verification Failed for ${sug.requirementId} — compilation errors and test failures.${RESET}\n`);
  } else if (!compiles) {
    console.log(`\n${RED}${BOLD}❌ Verification Failed for ${sug.requirementId} — compilation errors (tests may have passed).${RESET}\n`);
  } else {
    console.log(`\n${RED}${BOLD}❌ Verification Failed for ${sug.requirementId} — tests failed.${RESET}\n`);
  }

  // NOTE: Cache is updated at approval time (before this function is called),
  // not here — so it never depends on test results.

  return success;
}

async function main() {
  let projectRoot = process.cwd();
  if (path.basename(projectRoot) === 'qa-sync-tool') {
    projectRoot = path.dirname(projectRoot);
  }

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const askQuestion = (query: string): Promise<string> => {
    // Determine the allowed options for the current prompt
    const queryLower = query.toLowerCase();
    let allowed: string[] = [];

    if (queryLower.includes('(y/n)')) {
      allowed = ['y', 'n'];
    } else if (queryLower.includes('apply all changes at once')) {
      // (A), (R), (N)
      allowed = ['a', 'r', 'n'];
    } else if (queryLower.includes('1 = option 1') && queryLower.includes('2 = option 2')) {
      // 1, 2, n, s
      allowed = ['1', '2', 'n', 's'];
    } else if (queryLower.includes('y = yes') && queryLower.includes('n = skip') && queryLower.includes('s = skip')) {
      // y, n, s
      allowed = ['y', 'n', 's'];
    } else {
      // Generic backup parser for parenthesized single-character options
      const matches = query.match(/\(([A-Za-z0-9\/|,\s=]+)\)/);
      if (matches) {
        const content = matches[1].toLowerCase();
        if (content.includes('/')) {
          allowed = content.split('/').map(s => s.trim()).filter(Boolean);
        } else if (content.includes(',')) {
          const parts = content.split(',');
          for (const part of parts) {
            const eqIdx = part.indexOf('=');
            if (eqIdx !== -1) {
              allowed.push(part.substring(0, eqIdx).trim());
            } else {
              const words = part.trim().split(/\s+/);
              if (words[0] && words[0].length === 1) {
                allowed.push(words[0]);
              }
            }
          }
        }
      }
    }

    return new Promise((resolve) => {
      const ask = () => {
        rl.question(query, (ans) => {
          const sanitized = ans.trim().toLowerCase();
          // If we resolved allowed options, validate against them strictly.
          // Otherwise, allow anything (to prevent deadlock on unexpected prompts).
          if (allowed.length > 0 && !allowed.includes(sanitized)) {
            console.log(`❌ Invalid input. Please enter a valid option.`);
            ask();
          } else {
            resolve(sanitized);
          }
        });
      };
      ask();
    });
  };

  let shouldRunAnalysis = true;

  while (shouldRunAnalysis) {
    console.log(`\n${BOLD}${CYAN}🔍 Playwright QA Sync Tool${RESET}`);
    console.log(`${DIM}---------------------------------------------${RESET}`);

    // 1. Find Requirement Document
    console.log(`\n1. Scanning for Requirement Document...`);
    const rdFile = await findRequirementDocument(projectRoot);
    if (!rdFile) {
      console.error(`${RED}❌ Error: No requirement document (.md, .txt, .docx, requirement.docs) found in project root or subdirectories.${RESET}`);
      process.exit(1);
    }

    const relativeRdPath = path.relative(projectRoot, rdFile.filePath);
    console.log(`${GREEN}  ✓ Found Requirement Document: ${BOLD}${relativeRdPath}${RESET}`);

    const requirements = await extractRequirements(rdFile.content);
    console.log(`  ✓ Extracted ${BOLD}${requirements.length}${RESET} requirements.`);
    if (requirements.length === 0) {
      console.log(`${YELLOW}  ⚠️ Warning: No requirements found matching pattern (e.g., "R1: Title" or "[R2] Title").${RESET}`);
    }

    // ── Load previous snapshot from cache (for REMOVE cache cleanup and stable mapping) ──
    const cachePath = path.join(projectRoot, '.qa-sync-cache.json');
    let prevRequirements: Requirement[] = [];
    try {
      if (fs.existsSync(cachePath)) {
        const raw = fs.readFileSync(cachePath, 'utf-8').trim();
        if (raw && raw !== '[]') {
          prevRequirements = JSON.parse(raw);
        }
      }
    } catch (e) { /* ignore */ }

    // ── First-run baseline ────────────────────────────────────────────────────
    // If no cache exists yet or is empty, write every requirement as the baseline so the AI
    // has something to compare against on the very next run.
    const isCacheEmpty = prevRequirements.length === 0;
    if (isCacheEmpty && requirements.length > 0) {
      console.log(`\n${CYAN}${BOLD}📦 First run detected — creating requirement baseline cache...${RESET}`);
      upsertRequirementsInCache(projectRoot, requirements);
      console.log(`${GREEN}  ✓ Baseline cache written with ${requirements.length} requirements.${RESET}`);
      // Re-load after writing so prevRequirements is populated for this run.
      prevRequirements = [...requirements];
    }

    // Hash tracking is disabled — always perform a full analysis every run
    const rdChanged = true;

    // 2. Find Test Scripts
    console.log(`\n2. Scanning for Playwright Test Scripts...`);
    const specFiles = findTestScripts(projectRoot);
    console.log(`  ✓ Found ${BOLD}${specFiles.length}${RESET} test files (.spec.ts) in the project.`);

    // Auto-heal import paths for config/locators if they were relocated
    const filesToHeal = [...specFiles];
    const pwConfig = path.join(projectRoot, 'playwright.config.ts');
    if (fs.existsSync(pwConfig)) {
      filesToHeal.push(pwConfig);
    }
    healSpecImports(projectRoot, filesToHeal);

    const parsedTests = scanTestFiles(specFiles);
    console.log(`  ✓ Extracted ${BOLD}${parsedTests.length}${RESET} test cases.`);

    let missingLocators = getMissingLocators(projectRoot, specFiles);

    // 3. Analyzing Test Coverage and Code Options
    console.log(`\n3. Analyzing Test Coverage and Code Options...`);
    let rawSuggestions: Suggestion[];
    try {
      rawSuggestions = await analyzeCoverage(requirements, parsedTests, specFiles.length > 0, rdFile?.content, projectRoot, missingLocators, rdChanged);
    } catch (err: any) {
      if (err instanceof MappingError) {
        console.error(`\n${RED}${BOLD}❌ Mapping Error: Matching validation failed.${RESET}`);
        console.error(`${DIM}${err.message}${RESET}\n`);
        rl.close();
        process.exit(1);
      }
      throw err;
    }
    const suggestions = deduplicateSuggestions(rawSuggestions);

    // Enforce ADD suggestion rules and active requirement test preservation
    const activeRequirementIds = new Set(requirements.map(r => normalizeRequirementId(r.id)));
    const testCaseToSuggestions = new Map<string, Suggestion[]>();
    for (const sug of suggestions) {
      if (sug.filePath && sug.testTitle && sug.action !== 'NONE') {
        const key = `${path.resolve(projectRoot, sug.filePath).toLowerCase()}::${sug.testTitle.trim().toLowerCase()}`;
        const list = testCaseToSuggestions.get(key) || [];
        list.push(sug);
        testCaseToSuggestions.set(key, list);
      }
    }

    for (const sug of suggestions) {
      // An ADD suggestion must never remove, replace, or overwrite an existing active test.
      if (sug.action === 'ADD') {
        sug.originalCode = undefined;
        sug.patchDiff = undefined;
        sug.patchDiffOpt2 = undefined;
      }

      // A test for one active requirement must never be reused as the replacement target for another active requirement.
      if (sug.filePath && sug.testTitle && sug.action === 'MODIFY') {
        const key = `${path.resolve(projectRoot, sug.filePath).toLowerCase()}::${sug.testTitle.trim().toLowerCase()}`;
        const siblingSugs = testCaseToSuggestions.get(key) || [];
        const otherActiveSugs = siblingSugs.filter(s => 
          s.requirementId !== sug.requirementId && 
          activeRequirementIds.has(normalizeRequirementId(s.requirementId))
        );

        if (otherActiveSugs.length > 0) {
          sug.action = 'ADD';
          sug.originalCode = undefined;
          sug.patchDiff = undefined;
          sug.patchDiffOpt2 = undefined;
        }
      }
    }

    // Automatically update cache for REVIEW suggestions immediately (documentation changes only).
    // REVIEW = wording/title/goal change only, no code change needed, safe to update immediately.
    let reviewUpdatedCount = 0;
    const reqDiffs = compareRequirementDocuments(prevRequirements, requirements);
    for (const sug of suggestions) {
      if (sug.classification === 'REVIEW') {
        const normId = normalizeRequirementId(sug.requirementId);
        const diff = reqDiffs.get(normId);
        const isTrueReview = diff ? diff.classification === 'REVIEW' : true;

        if (isTrueReview) {
          updateCacheForApprovedSuggestion(projectRoot, sug, requirements, prevRequirements);
          reviewUpdatedCount++;
        }
      }
    }
    if (reviewUpdatedCount > 0) {
      console.log(`  ✓ Automatically updated cache baseline for ${BOLD}${reviewUpdatedCount}${RESET} REVIEW requirements (documentation changes only).`);
    }

    // 4. Show Suggestions
    console.log(`\n${BOLD}═══════════════════════════════════════════════════════${RESET}`);
    console.log(`${BOLD}         REQUIREMENT ANALYSIS REPORT${RESET}`);
    console.log(`${BOLD}═══════════════════════════════════════════════════════${RESET}\n`);

    let changeNeeded = false;

    for (const sug of suggestions) {
      const hasUpdates = (sug.locatorUpdates && sug.locatorUpdates.length > 0) ||
        (sug.configUpdates && sug.configUpdates.length > 0) ||
        (sug.envUpdates && sug.envUpdates.length > 0);

      if (sug.action !== 'NONE' || hasUpdates) changeNeeded = true;

      const isReview = sug.classification === 'REVIEW';
      const actionColor = sug.action === 'ADD' ? CYAN
        : sug.action === 'MODIFY' ? YELLOW
          : sug.action === 'REMOVE' ? RED
            : isReview ? ORANGE
              : GREEN;
      const actionLabel = sug.classification || (sug.action === 'NONE' ? 'IN_SYNC' : sug.action);

      // ── Header & Detailed Report ───────────────────────────────────
      const isVerbose = process.env.QA_SYNC_VERBOSE === 'true';
      if (actionLabel !== 'IN_SYNC' || isVerbose) {
        console.log(`${BOLD}${actionColor}▸ ${sug.requirementId}${RESET}  ${BOLD}${sug.title}${RESET}  ${BOLD}${actionColor}[${actionLabel}]${RESET}`);
        console.log(`${DIM}${'─'.repeat(60)}${RESET}`);

        // Show target file and matched test info
        if (sug.filePath) {
          const relPath = path.relative(projectRoot, sug.filePath).replace(/\\/g, '/');
          console.log(`  ${BOLD}Target File:${RESET}  ${CYAN}${relPath}${RESET}`);
        }
        if (sug.implementationSummary && sug.implementationSummary.toLowerCase() !== 'no test found.') {
          console.log(`  ${BOLD}Matched test:${RESET} ${sug.implementationSummary.trim()}`);
        } else {
          console.log(`  ${BOLD}Matched test:${RESET} ${RED}No test found.${RESET}`);
        }

        // ── Requirement Changes Section ─────────────────────────────────────
        const detectedChanges = sug.detectedChanges;
        const hasChange = detectedChanges && detectedChanges.trim().toLowerCase() !== 'none' && detectedChanges.trim() !== '';
        if (hasChange && detectedChanges) {
          console.log(`\n  ${BOLD}Requirement Changes:${RESET}`);
          console.log(`  ${detectedChanges.trim().replace(/\n/g, '\n  ')}`);

          let impactText = '';
          if (sug.classification === 'REVIEW') {
            impactText = 'Documentation only → REVIEW';
          } else if (sug.classification === 'MODIFY') {
            impactText = 'Executable requirement changed → MODIFY';
          } else if (sug.classification === 'ADD') {
            impactText = 'New requirement → ADD';
          } else if (sug.classification === 'ORPHAN') {
            impactText = 'Orphan test script → REMOVE';
          }
          console.log(`\n  ${BOLD}Determine the impact:${RESET}`);
          console.log(`  - ${impactText}`);
        }

        // ── Coverage Analysis Section ───────────────────────────────────────
        const showCoverageAnalysis = sug.classification === 'MODIFY' || sug.classification === 'ADD' || isVerbose;
        if (showCoverageAnalysis) {
          console.log(`\n  ${BOLD}Coverage Analysis:${RESET}`);

          // 1. Missing Implementation
          if (sug.classification === 'ADD') {
            console.log(`  - Missing implementation: Yes (No test script covers this requirement)`);
          } else if (sug.classification === 'MODIFY') {
            console.log(`  - Missing implementation: No (Test script exists but has missing assertions)`);
          }

          // 2. Missing assertions/actions
          if (sug.differencesFound) {
            const lines = sug.differencesFound.split('\n');
            const missingLines = lines.filter(l => l.includes('✗ MISSING') || l.includes('⚠ WORDING'));
            if (missingLines.length > 0) {
              console.log(`  - Missing assertions/actions:`);
              for (const ml of missingLines) {
                console.log(`    ${RED}✗${RESET}  ${ml.replace(/\[A\d+\]\s*(?:✗\s*MISSING|⚠\s*WORDING)\s*—\s*/i, '').trim()}`);
              }
            } else if (isVerbose) {
              console.log(`  - Missing assertions/actions: None`);
            }
          }

          // 3. Suggested code changes note
          if (sug.classification === 'MODIFY' || sug.classification === 'ADD') {
            console.log(`  - Suggested code changes: Available in Review mode`);
          }
        }

        // Show explanation (impact analysis) if available
        if (sug.impactAnalysis && sug.impactAnalysis.toLowerCase() !== 'none' && sug.impactAnalysis.trim() !== '') {
          console.log(`\n  ${BOLD}Explanation:${RESET} ${sug.impactAnalysis}`);
        }

        // Configuration Updates
        if (hasUpdates) {
          console.log(`\n${BOLD}  ⚙️  Configuration Updates:${RESET}`);
          sug.locatorUpdates?.forEach(u => console.log(`    ${GREEN}+ Locator: ${u.key} → '${u.value}'${RESET}`));
          sug.configUpdates?.forEach(u => console.log(`    ${GREEN}+ Config:  ${u.key} → '${u.value}'${RESET}`));
          sug.envUpdates?.forEach(u => console.log(`    ${GREEN}+ Env:     ${u.key} → '${u.value}'${RESET}`));
        }

        console.log(`\n${DIM}${'─'.repeat(60)}${RESET}\n`);
      }
    }

    // ── Phase 2: Analysis Summary Table ─────────────────────────────────────
    const inSyncSuggestions = suggestions.filter(s => s.classification === 'IN_SYNC' || (!s.classification && s.action === 'NONE' && (!s.differencesFound || s.differencesFound.includes('✓ COVERED') && !s.differencesFound.includes('✗ MISSING') && !s.differencesFound.includes('⚠ WORDING'))));
    const reviewSuggestions = suggestions.filter(s => s.classification === 'REVIEW' || (!s.classification && s.action === 'NONE' && (s.differencesFound && (s.differencesFound.includes('⚠ WORDING') || s.differencesFound.includes('Removed')))));
    const modifySuggestions = suggestions.filter(s => s.classification === 'MODIFY' || (!s.classification && s.action === 'MODIFY'));
    const addSuggestions = suggestions.filter(s => s.classification === 'ADD' || (!s.classification && s.action === 'ADD'));
    const orphanSuggestions = suggestions.filter(s => s.classification === 'ORPHAN' || (!s.classification && s.action === 'REMOVE'));

    console.log(`\n${BOLD}═══════════════════════════════════════════════════════${RESET}`);
    console.log(`${BOLD}  ANALYSIS SUMMARY${RESET}`);
    console.log(`${BOLD}═══════════════════════════════════════════════════════${RESET}`);
    console.log(`  ${GREEN}✅ IN_SYNC${RESET}   : ${inSyncSuggestions.length.toString().padEnd(2)} ${inSyncSuggestions.length > 0 ? '→ ' + inSyncSuggestions.map(s => s.requirementId).join(', ') : ''}`);
    console.log(`  ${ORANGE}🔍 REVIEW${RESET}    : ${reviewSuggestions.length.toString().padEnd(2)} ${reviewSuggestions.length > 0 ? '→ ' + reviewSuggestions.map(s => s.requirementId).join(', ') : ''}`);
    console.log(`  ${YELLOW}⚠  MODIFY${RESET}    : ${modifySuggestions.length.toString().padEnd(2)} ${modifySuggestions.length > 0 ? '→ ' + modifySuggestions.map(s => s.requirementId).join(', ') : ''}`);
    console.log(`  ${CYAN}➕ ADD${RESET}       : ${addSuggestions.length.toString().padEnd(2)} ${addSuggestions.length > 0 ? '→ ' + addSuggestions.map(s => s.requirementId).join(', ') : ''}`);
    console.log(`  ${RED}🗑  REMOVE${RESET}    : ${orphanSuggestions.length.toString().padEnd(2)} ${orphanSuggestions.length > 0 ? '→ ' + orphanSuggestions.map(s => s.testTitle || s.requirementId).join(', ') : ''}`);
    console.log(`${DIM}───────────────────────────────────────────────────────${RESET}`);
    const affectedFiles = Array.from(new Set(suggestions.filter(s => s.action !== 'NONE').map(s => s.filePath ? path.basename(s.filePath) : '').filter(Boolean)));
    console.log(`  ${BOLD}Affected files:${RESET} ${affectedFiles.length > 0 ? affectedFiles.join(', ') : 'None'}`);
    console.log(`${BOLD}═══════════════════════════════════════════════════════${RESET}\n`);

    if (!changeNeeded) {
      console.log(`${GREEN}${BOLD}✓ All test scripts are in sync with your Requirement Document.${RESET}`);
      console.log(`${DIM}  No differences detected. No code changes are required.${RESET}\n`);
      console.log(`${BOLD}🧪 Running Playwright tests to verify the suite...${RESET}\n`);
      let verificationSuccess = false;
      try {
        const { execSync } = require('child_process');
        execSync('npx playwright test --headed', { stdio: 'inherit', cwd: projectRoot });
        console.log(`\n${GREEN}${BOLD}✅ Verification Successful: All tests passed!${RESET}\n`);
        verificationSuccess = true;
      } catch (error: any) {
        console.log(`\n${RED}${BOLD}❌ Verification Failed: Some tests failed or an error occurred during execution.${RESET}\n`);
      }
      if (verificationSuccess) {
        // All requirements are confirmed in-sync and tests pass.
        // Upsert every requirement individually — never overwrite the whole cache.
        upsertRequirementsInCache(projectRoot, requirements);
      }
      shouldRunAnalysis = false;
      break;
    }

    // ── Per-suggestion approval ─────────────────────────────────────────────
    // Build the list of actionable suggestions in apply order:
    //   1. Project-level locator/config/env updates
    //   2. ADD (new tests)
    //   3. MODIFY (patch existing tests)
    //   4. REMOVE (delete orphaned tests)
    const newSuggestions = suggestions.filter(s => s.action === 'ADD');
    const modifiedSuggestions = suggestions.filter(s => s.action === 'MODIFY');
    const deletedSuggestions = suggestions.filter(s => s.action === 'REMOVE');
    const projectSuggestions = suggestions.filter(s => s.action === 'NONE' && (
      (s.locatorUpdates && s.locatorUpdates.length > 0) ||
      (s.configUpdates && s.configUpdates.length > 0) ||
      (s.envUpdates && s.envUpdates.length > 0)
    ));

    // ── Show summary of proposed actions ───────────────────────────────────
    const validRequirementIds = requirements.map(r => normalizeRequirementId(r.id));
    let changed = false;

    if (!changeNeeded) {
      // no action items — handled below
    } else {
      console.log(`\n${BOLD}PROPOSED ACTIONS SUMMARY:${RESET}`);
      console.log("-".repeat(60));
      if (newSuggestions.length > 0)
        console.log(`✨ Add new tests:       ${CYAN}${newSuggestions.map(s => s.requirementId).join(', ')}${RESET}`);
      if (modifiedSuggestions.length > 0)
        console.log(`⚙️  Patch existing tests: ${YELLOW}${modifiedSuggestions.map(s => s.requirementId).join(', ')}${RESET}`);
      if (deletedSuggestions.length > 0)
        console.log(`🗑  Remove orphan tests:  ${RED}${deletedSuggestions.map(s => s.requirementId).join(', ')}${RESET}`);
      if (projectSuggestions.length > 0)
        console.log(`🔧 Project-level updates: locators / config / env`);
      console.log("-".repeat(60));

      // ── Ask whether to review suggestions individually or apply all ────
      const bulkChoice = await askQuestion(
        `\nApply all changes at once (A), review each individually (R), or skip (N)? → `
      );

      if (bulkChoice.toLowerCase() === 'a') {
        const allSuggestions = [
          ...suggestions.filter(s => s.classification === 'REVIEW'),
          ...projectSuggestions,
          ...newSuggestions,
          ...modifiedSuggestions,
          ...deletedSuggestions
        ];

        // Deduplicate suggestions by requirementId to avoid double execution
        const uniqueSuggestions: Suggestion[] = [];
        const seenReqs = new Set<string>();
        for (const s of allSuggestions) {
          if (!seenReqs.has(s.requirementId)) {
            seenReqs.add(s.requirementId);
            uniqueSuggestions.push(s);
          }
        }

        const summary = {
          applied: [] as string[],
          failed: [] as string[],
          skipped: [] as string[],
          pending: [] as string[]
        };

        for (let si = 0; si < uniqueSuggestions.length; si++) {
          const sug = uniqueSuggestions[si];
          console.log(`\n${BOLD}▸ Applying change for ${sug.requirementId} (${sug.title})...${RESET}`);

          if (sug.classification === 'REVIEW') {
            // REVIEW changes have no code changes — apply directly by updating the cache
            try {
              updateCacheForApprovedSuggestion(projectRoot, sug, requirements, prevRequirements);
              summary.applied.push(sug.requirementId);
              changed = true;
            } catch (err) {
              console.error(`❌ Failed to update cache for ${sug.requirementId}:`, err);
              summary.failed.push(sug.requirementId);
            }
            continue;
          }

          try {
            const success = await applyAndVerifySuggestion(
              projectRoot,
              sug,
              specFiles,
              validRequirementIds,
              requirements,
              prevRequirements,
              askQuestion,
              true // Skip validation per suggestion — run once at the end
            );

            if (success) {
              summary.applied.push(sug.requirementId);
              changed = true;
            } else {
              summary.skipped.push(sug.requirementId);
            }
          } catch (err: any) {
            console.error(`\n${RED}❌ Error applying ${sug.requirementId} — skipping and continuing:${RESET}`, err?.message || err);
            summary.failed.push(sug.requirementId);
            // Continue with the next suggestion — never abort the loop
          }
        }

        // Run validation once at the end (only if at least one suggestion was applied)
        if (summary.applied.length > 0) {
          console.log(`\n${BOLD}🔍 Running final validation checks...${RESET}`);
          let validationSuccess = false;
          try {
            const { execSync } = require('child_process');
            console.log(`Checking TypeScript compilation...`);
            execSync('npx tsc --noEmit', { stdio: 'ignore', cwd: projectRoot });
            console.log(`Running Playwright tests...`);
            execSync('npx playwright test', { stdio: 'inherit', cwd: projectRoot });
            validationSuccess = true;
          } catch (e) {
            // Keep going to print the summary
          }
          if (validationSuccess) {
            console.log(`\n${GREEN}${BOLD}✅ Final verification successful: All tests passed!${RESET}\n`);
            // Update cache for all successfully applied changes
            for (const appliedReqId of summary.applied) {
              const sug = uniqueSuggestions.find(s => s.requirementId === appliedReqId);
              if (sug && sug.classification !== 'REVIEW') {
                try {
                  updateCacheForApprovedSuggestion(projectRoot, sug, requirements, prevRequirements);
                } catch (cacheError) {
                  console.error(`⚠️ Failed to update cache for suggestion ${sug.requirementId}:`, cacheError);
                }
              }
            }
          } else {
            console.log(`\n${RED}${BOLD}❌ Final verification failed: Compiling or tests failed. Cache not updated.${RESET}\n`);
          }
        }

        // Display Apply All Summary
        console.log(`\n${BOLD}${'═'.repeat(54)}${RESET}`);
        console.log(`${BOLD}              APPLICATION SUMMARY                     ${RESET}`);
        console.log(`${BOLD}${'═'.repeat(54)}${RESET}`);
        console.log(`  ✅ ${GREEN}${BOLD}Applied  (${summary.applied.length}):${RESET} ${summary.applied.join(', ') || 'None'}`);
        console.log(`  ❌ ${RED}${BOLD}Failed   (${summary.failed.length}):${RESET} ${summary.failed.join(', ') || 'None'}`);
        console.log(`  ⚠️  ${YELLOW}${BOLD}Skipped  (${summary.skipped.length}):${RESET} ${summary.skipped.join(', ') || 'None'}`);
        console.log(`  ⏭  ${DIM}${BOLD}Pending  (${summary.pending.length}):${RESET} ${summary.pending.join(', ') || 'None'}`);
        console.log(`${BOLD}${'═'.repeat(54)}${RESET}\n`);
      } else if (bulkChoice.toLowerCase() === 'r') {
        const orderedSuggestions = [
          ...projectSuggestions,
          ...newSuggestions,
          ...modifiedSuggestions,
          ...deletedSuggestions,
        ];
        let skipAll = false;

        const reviewSummary = {
          applied: [] as string[],
          failed: [] as string[],
          skipped: [] as string[],
          pending: [] as string[]
        };

        for (let ri = 0; ri < orderedSuggestions.length; ri++) {
          const sug = orderedSuggestions[ri];

          if (skipAll) {
            // Mark all remaining suggestions as pending
            reviewSummary.pending.push(sug.requirementId);
            continue;
          }

          const actionLabel = sug.action === 'ADD' ? 'NEW TEST'
            : sug.action === 'MODIFY' ? 'PATCH'
              : sug.action === 'REMOVE' ? 'REMOVE'
                : 'PROJECT UPDATE';

          console.log(`\n${BOLD}${'═'.repeat(60)}${RESET}`);
          console.log(`${BOLD}▸ ${sug.requirementId}  [${actionLabel}]  ${sug.title}${RESET}`);
          if (sug.filePath) {
            const relPath = path.relative(projectRoot, sug.filePath).replace(/\\/g, '/');
            console.log(`  ${BOLD}Target File:${RESET}  ${CYAN}${relPath}${RESET}`);
          }

          if (sug.differencesFound) {
            console.log(`  ${BOLD}Details:${RESET}`);
            console.log(`    ${sug.differencesFound.split('\n').join('\n    ')}`);
          }
          if (sug.locatorUpdates?.length) {
            console.log(`  ${BOLD}🔑 Locator updates:${RESET}`);
            sug.locatorUpdates.forEach(u => console.log(`     ${GREEN}+ ${u.key}: '${u.value}'${RESET}`));
          }

          if (sug.action === 'ADD' || sug.action === 'MODIFY') {
            const hasOpt2 = (sug.proposedCodeOpt2 && sug.proposedCodeOpt2.trim() !== '' && sug.proposedCodeOpt2.trim().toLowerCase() !== 'none') ||
                            (sug.patchDiffOpt2 && sug.patchDiffOpt2.trim() !== '' && sug.patchDiffOpt2.trim().toLowerCase() !== 'none');

            if (hasOpt2) {
              console.log(`\n  ${BOLD}[OPTION 1]:${RESET}`);
            } else {
              console.log(`\n  ${BOLD}[PROPOSED CHANGES]:${RESET}`);
            }

            if (sug.patchDiff) {
              printCleanCodeDiff(sug.patchDiff);
            } else if (sug.proposedCode) {
              printDiff(sug.originalCode, sug.proposedCode);
            }
            if (sug.whyNeeded) {
              console.log(`  ${BOLD}Reason/Why:${RESET} ${YELLOW}${sug.whyNeeded}${RESET}`);
            }

            if (hasOpt2) {
              console.log(`\n  ${BOLD}[OPTION 2]:${RESET}`);
              if (sug.patchDiffOpt2) {
                printCleanCodeDiff(sug.patchDiffOpt2);
              } else if (sug.proposedCodeOpt2) {
                printDiff(sug.originalCode, sug.proposedCodeOpt2);
              }
              if (sug.whyNeededOpt2) {
                console.log(`  ${BOLD}Why Option 2:${RESET} ${YELLOW}${sug.whyNeededOpt2}${RESET}`);
              }
            }

            console.log();
            const promptText = hasOpt2
              ? `Apply this change? (1 = Option 1, 2 = Option 2, N = skip, S = skip all remaining) → `
              : `Apply this change? (1 = Apply, N = skip, S = skip all remaining) → `;

            const optionChoice = await askQuestion(promptText);
            const choiceClean = optionChoice.toLowerCase().trim();
            let approvedSug: Suggestion | null = null;
            if (choiceClean === '1') {
              approvedSug = sug;
              console.log(`${GREEN}✓ Approved for application.${RESET}`);
            } else if (choiceClean === '2' && hasOpt2) {
              approvedSug = {
                ...sug,
                proposedCode: sug.proposedCodeOpt2,
                patchDiff: sug.patchDiffOpt2,
                whyNeeded: sug.whyNeededOpt2
              };
              console.log(`${GREEN}✓ Option 2 approved for application.${RESET}`);
            } else if (choiceClean === 's') {
              skipAll = true;
              reviewSummary.pending.push(sug.requirementId);
              console.log(`${YELLOW}Skipping all remaining changes.${RESET}`);
            } else {
              reviewSummary.skipped.push(sug.requirementId);
              console.log(`${DIM}Skipped.${RESET}`);
            }

            if (approvedSug) {
              try {
                const applySuccess = await applyAndVerifySuggestion(projectRoot, approvedSug, specFiles, validRequirementIds, requirements, prevRequirements, askQuestion);
                if (applySuccess) {
                  try {
                    updateCacheForApprovedSuggestion(projectRoot, approvedSug, requirements, prevRequirements);
                  } catch (cacheError) {
                    console.error(`⚠️ Failed to update cache for suggestion ${approvedSug.requirementId}:`, cacheError);
                  }
                  reviewSummary.applied.push(approvedSug.requirementId);
                  changed = true;
                } else {
                  reviewSummary.failed.push(approvedSug.requirementId);
                  console.log(`${YELLOW}⚠️  ${approvedSug.requirementId} was not applied — verification did not pass. Continuing with next suggestion.${RESET}`);
                }
              } catch (err: any) {
                console.error(`\n${RED}❌ Error applying ${approvedSug.requirementId} — continuing with next suggestion:${RESET}`, err?.message || err);
                reviewSummary.failed.push(approvedSug.requirementId);
              }
            }
          } else {
            // REMOVE or NONE with configuration/locator updates
            if (sug.action === 'REMOVE') {
              printDiff(sug.originalCode, undefined);
            }

            const perChoice = await askQuestion(
              `Apply this change? (Y = yes, N = skip, S = skip all remaining) → `
            );
            const choiceClean = perChoice.toLowerCase().trim();
            let approvedSug: Suggestion | null = null;
            if (choiceClean === 'y') {
              approvedSug = sug;
              console.log(`${GREEN}✓ Approved for application.${RESET}`);
            } else if (choiceClean === 's') {
              skipAll = true;
              reviewSummary.pending.push(sug.requirementId);
              console.log(`${YELLOW}Skipping all remaining changes.${RESET}`);
            } else {
              reviewSummary.skipped.push(sug.requirementId);
              console.log(`${DIM}Skipped.${RESET}`);
            }

            if (approvedSug) {
              try {
                const applySuccess = await applyAndVerifySuggestion(projectRoot, approvedSug, specFiles, validRequirementIds, requirements, prevRequirements, askQuestion);
                if (applySuccess) {
                  try {
                    updateCacheForApprovedSuggestion(projectRoot, approvedSug, requirements, prevRequirements);
                  } catch (cacheError) {
                    console.error(`⚠️ Failed to update cache for suggestion ${approvedSug.requirementId}:`, cacheError);
                  }
                  reviewSummary.applied.push(approvedSug.requirementId);
                  changed = true;
                } else {
                  reviewSummary.failed.push(approvedSug.requirementId);
                  console.log(`${YELLOW}⚠️  ${approvedSug.requirementId} was not applied — verification did not pass. Continuing with next suggestion.${RESET}`);
                }
              } catch (err: any) {
                console.error(`\n${RED}❌ Error applying ${approvedSug.requirementId} — continuing with next suggestion:${RESET}`, err?.message || err);
                reviewSummary.failed.push(approvedSug.requirementId);
              }
            }
          }
        }

        // Display Review Summary
        console.log(`\n${BOLD}${'═'.repeat(54)}${RESET}`);
        console.log(`${BOLD}              APPLICATION SUMMARY                     ${RESET}`);
        console.log(`${BOLD}${'═'.repeat(54)}${RESET}`);
        console.log(`  ✅ ${GREEN}${BOLD}Applied  (${reviewSummary.applied.length}):${RESET} ${reviewSummary.applied.join(', ') || 'None'}`);
        console.log(`  ❌ ${RED}${BOLD}Failed   (${reviewSummary.failed.length}):${RESET} ${reviewSummary.failed.join(', ') || 'None'}`);
        console.log(`  ⚠️  ${YELLOW}${BOLD}Skipped  (${reviewSummary.skipped.length}):${RESET} ${reviewSummary.skipped.join(', ') || 'None'}`);
        console.log(`  ⏭  ${DIM}${BOLD}Pending  (${reviewSummary.pending.length}):${RESET} ${reviewSummary.pending.join(', ') || 'None'}`);
        console.log(`${BOLD}${'═'.repeat(54)}${RESET}\n`);
      }
    }

    // Ensure that any requirement removed from the active document is also removed from the baseline cache
    if (fs.existsSync(cachePath)) {
      try {
        const raw = fs.readFileSync(cachePath, 'utf-8').trim();
        if (raw && raw !== '[]') {
          const cachedReqs: Requirement[] = JSON.parse(raw);
          const activeIds = new Set(requirements.map(r => normalizeRequirementId(r.id)));
          const filteredReqs = cachedReqs.filter(r => activeIds.has(normalizeRequirementId(r.id)));
          if (filteredReqs.length !== cachedReqs.length) {
            fs.writeFileSync(cachePath, JSON.stringify(filteredReqs, null, 2), 'utf-8');
            console.log(`${GREEN}  ✓ Cleaned up removed requirements from baseline cache.${RESET}`);
          }
        }
      } catch (e) { /* ignore */ }
    }

    if (!changed) {
      console.log(`\n${YELLOW}No changes were approved or applied.${RESET}\n`);
    }
    shouldRunAnalysis = false;
  }

  rl.close();
}

/**
 * Automator that spins up a live Playwright instance to discover missing CSS selectors on the site.
 * Fix Issue 1: accepts projectRoot and reads baseURL from config.ts instead of using a
 * hardcoded domain, making the function project-agnostic.
 */
async function discoverMissingLocatorsOnLiveSite(keys: string[], projectRoot: string): Promise<Map<string, string>> {
  const discovered = new Map<string, string>();
  let browser;
  try {
    // Resolve baseURL from the project's config.ts
    let baseUrl = 'http://localhost:3000';
    try {
      const configPath = findConfigFilePath(projectRoot);
      if (configPath && fs.existsSync(configPath)) {
        const configContent = fs.readFileSync(configPath, 'utf-8');
        const urlMatch = configContent.match(/baseURL:\s*(['"`])(.*?)\1/);
        if (urlMatch) baseUrl = urlMatch[2];
      }
    } catch (_) { /* keep default */ }

    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();

    // Navigate to the project's own base URL (not hardcoded to thesouledstore.com)
    await page.goto(baseUrl);

    for (const key of keys) {
      // PDP Product main image / thumbnails
      if (key.includes('mainProductImage') || key.includes('mainImage') || key.includes('galleryImages') ||
        key.includes('galleryThumbnails') || key.includes('thumbnails')) {
        try {
          await page.waitForSelector('input#search', { timeout: 6000 });
          const searchBox = page.locator('input#search');
          await searchBox.fill('T-shirt');
          await searchBox.press('Enter');

          await page.waitForSelector('a[href*="/product/"]', { timeout: 6000 });
          await page.locator('a[href*="/product/"]').first().click();
          await page.waitForTimeout(3000);

          if (key.includes('mainProductImage') || key.includes('mainImage') || key.includes('galleryImages')) {
            const mainImgSelector = await page.evaluate(() => {
              const generateRobustSelector = (el: any) => {
                if (el.id && !/\d{4,}/.test(el.id)) return `#${el.id}`;
                if (el.getAttribute('data-testid')) return `[data-testid="${el.getAttribute('data-testid')}"]`;
                if (el.getAttribute('alt')) return `img[alt="${el.getAttribute('alt')}"]`;
                const classes = el.className.split(' ').filter((c: any) => c && !c.startsWith('gm-') && !c.startsWith('customFade') && !/^[0-9_-]+$/.test(c) && !/[a-f0-9]{8,}/.test(c));
                if (classes.includes('minProd')) return 'img.minProd:visible';
                if (classes.length > 0) return `img.${classes.join('.')}:visible`;
                return 'LOW CONFIDENCE: img.minProd';
              };

              const imgs = Array.from((globalThis as any).document.querySelectorAll('img'));
              const validImgs = imgs.map((img: any) => {
                const rect = img.getBoundingClientRect();
                return { el: img, area: rect.width * rect.height, class: img.className };
              }).filter((info: any) => info.area > 50000 && info.class.includes('minProd'));

              if (validImgs.length > 0) {
                validImgs.sort((a: any, b: any) => b.area - a.area);
                return generateRobustSelector(validImgs[0].el);
              }
              return null;
            });
            if (mainImgSelector) discovered.set(key, mainImgSelector);
          }

          if (key.includes('galleryThumbnails') || key.includes('thumbnails')) {
            const thumbSelector = await page.evaluate(() => {
              const doc = (globalThis as any).document;
              const dots = doc.querySelector('.VueCarousel-dot');
              if (dots) return '.VueCarousel-dot:visible';
              const slickDots = doc.querySelector('.slick-dots li');
              if (slickDots) return '.slick-dots li:visible';
              const mdImg = doc.querySelector('img.minProd');
              if (mdImg) return 'img.minProd:visible';
              return 'LOW CONFIDENCE: .VueCarousel-dot';
            });
            if (thumbSelector) discovered.set(key, thumbSelector);
          }
        } catch (e) { /* ignore */ }
      }
    }
  } catch (err: any) {
    console.warn(`   ⚠️ Live DOM inspection failed: ${err.message || err}`);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
  return discovered;
}

main().catch((err) => {
  console.error('\n❌ An unexpected error occurred:', err);
  process.exit(1);
});