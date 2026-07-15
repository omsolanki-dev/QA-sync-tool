import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { GoogleGenerativeAI } from '@google/generative-ai';
import {
  Suggestion,
  Requirement,
  ParsedTest,
  applyChanges,
  resolveLocatorsPathForFile,
  findConfigFilePath,
  extractExecutableAtomics,
  parseTestFileStructure,
  cleanProposedCode,
  normalizeRequirementId
} from './testScanner';

// ANSI colors
const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const DIM = '\x1b[2m';

const PLAYWRIGHT_EXEC_PATTERN = /expect\s*\(|toBeVisible|toHaveURL|toHaveText|toHaveValue|toBeEnabled|toBeChecked|toHaveCount|toContainText|toHaveAttribute|toHaveTitle|\.click\s*\(|\.fill\s*\(|\.hover\s*\(|\.select|\.check\s*\(|\.uncheck\s*\(|\.dragTo\s*\(|locator\s*\(/i;

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

function detectDuplicates(content: string): string[] {
  const errors: string[] = [];

  // Check duplicate imports
  const importLines = content.split('\n').filter(l => l.trim().startsWith('import '));
  const importedSymbols = new Set<string>();
  for (const line of importLines) {
    const symbolMatch = line.match(/import\s+(?:type\s+)?\{([^}]+)\}/);
    if (symbolMatch) {
      const symbols = symbolMatch[1].split(',').map(s => s.trim().split(/\s+as\s+/)[0].trim());
      for (const sym of symbols) {
        if (sym && importedSymbols.has(sym)) {
          errors.push(`Duplicate import symbol: '${sym}' is imported multiple times.`);
        }
        importedSymbols.add(sym);
      }
    }
  }

  // Check duplicate helper function names
  const funcRegex = /function\s+(\w+)/g;
  let match;
  const declaredFunctions = new Set<string>();
  while ((match = funcRegex.exec(content)) !== null) {
    const name = match[1];
    if (declaredFunctions.has(name)) {
      errors.push(`Duplicate helper function declaration: function '${name}' is declared multiple times.`);
    }
    declaredFunctions.add(name);
  }

  // Check duplicate test titles
  const testTitleRegex = /test(?:\.describe|\.only|\.skip|\.fixme|\.fail)?\s*\(\s*(['"`])([\s\S]*?)\1/g;
  const testTitles = new Set<string>();
  while ((match = testTitleRegex.exec(content)) !== null) {
    const title = match[2];
    if (testTitles.has(title)) {
      errors.push(`Duplicate test case / block title: '${title}' is declared multiple times.`);
    }
    testTitles.add(title);
  }

  return errors;
}

function backupAffectedFiles(sug: Suggestion, projectRoot: string): Map<string, string | null> {
  const backups = new Map<string, string | null>();

  // 1. Spec file path
  if (sug.filePath) {
    const specPath = path.resolve(projectRoot, sug.filePath);
    backups.set(specPath, fs.existsSync(specPath) ? fs.readFileSync(specPath, 'utf-8') : null);

    // Backup resolved locators file path
    try {
      const resolvedLocators = resolveLocatorsPathForFile(specPath, projectRoot);
      if (resolvedLocators && resolvedLocators.path) {
        const locPath = path.resolve(projectRoot, resolvedLocators.path);
        backups.set(locPath, fs.existsSync(locPath) ? fs.readFileSync(locPath, 'utf-8') : null);
      }
    } catch (e) { /* ignore */ }
  }

  // 2. Config updates
  if (sug.configUpdates && sug.configUpdates.length > 0) {
    const configPath = findConfigFilePath(projectRoot) || path.join(projectRoot, 'config.ts');
    backups.set(configPath, fs.existsSync(configPath) ? fs.readFileSync(configPath, 'utf-8') : null);
  }

  // 3. Env updates
  if (sug.envUpdates && sug.envUpdates.length > 0) {
    const envPath = path.resolve(projectRoot, '.env');
    backups.set(envPath, fs.existsSync(envPath) ? fs.readFileSync(envPath, 'utf-8') : null);
  }

  return backups;
}

function restoreAffectedFiles(backups: Map<string, string | null>) {
  for (const [filePath, content] of backups.entries()) {
    if (content === null) {
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    } else {
      fs.writeFileSync(filePath, content, 'utf-8');
    }
  }
}

function extractNewCompileErrors(baseline: string, current: string): string[] {
  const baselineLines = new Set(baseline.split('\n').map(l => l.trim()));
  const currentLines = current.split('\n').map(l => l.trim());
  const newErrors: string[] = [];

  for (const line of currentLines) {
    if (line.includes('error TS') && !baselineLines.has(line)) {
      newErrors.push(line);
    }
  }
  return newErrors;
}

function validateProposedCode(
  sug: Suggestion,
  code: string,
  isOption2: boolean,
  projectRoot: string,
  specFiles: string[],
  validRequirementIds: string[],
  requirement: Requirement,
  baselineErrors: string
): { success: boolean; errors: string[] } {
  const origProposed = sug.proposedCode;
  const origProposedOpt2 = sug.proposedCodeOpt2;

  if (isOption2) {
    sug.proposedCodeOpt2 = code;
  } else {
    sug.proposedCode = code;
  }

  const backups = backupAffectedFiles(sug, projectRoot);
  const errors: string[] = [];

  try {
    // Temporarily apply modifications
    applyChanges(projectRoot, [sug], specFiles, validRequirementIds);

    const specPath = path.resolve(projectRoot, sug.filePath!);
    if (fs.existsSync(specPath)) {
      const content = fs.readFileSync(specPath, 'utf-8');

      // 1. Duplicate check
      const dupErrors = detectDuplicates(content);
      errors.push(...dupErrors);

      // 1b. Fallback expression and invented selectors check
      if (/theSouledStoreLocators\.[a-zA-Z0-9_.]+\s*\|\|/i.test(content) || /\.locator\([^)]*\|\|[^)]*\)/.test(content)) {
        errors.push(`Forbidden fallback expression (e.g. '||') used when referencing a locator. Never generate inline selector fallbacks. Add the missing locator to LOCATOR_UPDATES instead.`);
      }

      // 2. Atomic coverage check — scan ALL test cases in the file
      // Handles SCOPE A (testTitle may be undefined) and SCOPE B (inserted title may differ)
      const atomics = extractExecutableAtomics(requirement);
      const parsed = parseTestFileStructure(content, specPath);

      const execLines: string[] = [];
      for (const seg of parsed) {
        if (seg.type !== 'testCase') continue;
        for (const b of seg.structure.blocks) {
          const lines = b.code.split('\n').filter(l => {
            const trimmed = l.trim();
            if (trimmed.startsWith('//') || trimmed.startsWith('*') || trimmed.startsWith('/*')) return false;
            return PLAYWRIGHT_EXEC_PATTERN.test(trimmed);
          });
          execLines.push(...lines);
        }
      }

      for (let i = 0; i < atomics.length; i++) {
        const atomicText = atomics[i];
        const terms = extractAtomicKeyTerms(atomicText);
        if (terms.length < 2) continue; // Statement too short — assumed covered

        const covered = execLines.some(line => {
          const lineLower = line.toLowerCase();
          let hits = terms.filter(t => lineLower.includes(t)).length;
          const INTERACTION_KEYWORDS = ['select', 'click', 'tap', 'choose', 'press', 'submit', 'enter', 'fill', 'type', 'navigate', 'open', 'locate'];
          const PLAYWRIGHT_ACTIONS = ['.click', '.fill', '.press', '.goto', '.selectOption', '.hover', '.check', '.uncheck'];
          const OUTCOME_KEYWORDS = ['display', 'show', 'visible', 'contain', 'verify', 'assert', 'should', 'shall', 'must', 'reflect', 'confirm', 'exist', 'be'];
          const PLAYWRIGHT_ASSERTIONS = ['expect(', 'tobevisible', 'tohaveurl', 'tohavetext', 'tocontaintext', 'tobeenabled', 'tohavevalue', 'tobechecked', 'tohavecount'];

          if (terms.some(t => INTERACTION_KEYWORDS.includes(t)) && PLAYWRIGHT_ACTIONS.some(a => lineLower.includes(a))) hits++;
          if (terms.some(t => OUTCOME_KEYWORDS.includes(t)) && PLAYWRIGHT_ASSERTIONS.some(a => lineLower.includes(a))) hits++;
          return hits >= 2;
        });

        if (!covered) {
          errors.push(`Requirement statement [A${i + 1}] is missing verification/action coverage in the test: "${atomicText}"`);
        }
      }
    } else {
      errors.push(`Target spec file was not created: ${sug.filePath}`);
    }

    // 3. Compile validation
    let compilationError = '';
    try {
      execSync('npx tsc --noEmit --skipLibCheck', { cwd: projectRoot, stdio: 'pipe' });
    } catch (err: any) {
      compilationError = (err.stdout || err.stderr || '').toString();
    }

    if (compilationError) {
      const newErrors = extractNewCompileErrors(baselineErrors, compilationError);
      if (newErrors.length > 0) {
        errors.push(...newErrors.map(e => `TypeScript compilation error: ${e}`));
      }
    }
  } catch (applyErr: any) {
    errors.push(`Failed to apply proposed code suggestion: ${applyErr.message || applyErr}`);
  } finally {
    restoreAffectedFiles(backups);
    sug.proposedCode = origProposed;
    sug.proposedCodeOpt2 = origProposedOpt2;
  }

  return { success: errors.length === 0, errors };
}

async function repairCodeWithGemini(
  genAI: GoogleGenerativeAI,
  modelName: string,
  requirement: Requirement,
  sug: Suggestion,
  failedCode: string,
  validationErrors: string[],
  isOption2: boolean
): Promise<string> {
  const model = genAI.getGenerativeModel({
    model: modelName,
    generationConfig: { temperature: 0.0 }
  });

  // Determine insertion scope so the repair model generates the right code shape
  const repairScopeHint = sug.proposedCode?.includes('test.describe(')
    ? 'SCOPE A — Return a FULL .spec.ts file: imports + test.describe block + module-scope helpers + test() block + all assertions.'
    : /\btest\s*\(/.test(sug.proposedCode || '')
      ? 'SCOPE B — Return ONLY the new test() block (no imports, no describe wrapper). The block will be inserted into an existing describe.'
      : 'SCOPE C — Return ONLY the missing assertion/action lines. Do not re-emit the test() wrapper or surrounding unchanged code.';

  const prompt = `You are a senior Playwright test engineer. You previously generated proposed code for requirement ${requirement.id} ("${requirement.title}"), which failed validation checks.

Insertion Scope: ${repairScopeHint}

Failed Code:
\`\`\`typescript
${failedCode}
\`\`\`

Validation Errors Encountered:
${validationErrors.map(e => `- ${e}`).join('\n')}

Requirement Details:
ID: ${requirement.id}
Title: ${requirement.title}
Goal: ${requirement.goal || 'N/A'}
Description:
${requirement.description}
Expected Result: ${requirement.expectedResult || 'N/A'}

Rules for Regeneration:
1. Fix all validation errors listed above.
2. Do not duplicate imports, variables, helpers, or test cases that already exist.
3. Reuse existing locators, page objects, imports, and fixtures.
4. Ensure 100% atomic coverage of the requirement statements.
5. PROJECT-AWARE CODE GENERATION CONSTRAINTS:
   - Generate code ONLY using APIs, language features, and patterns that already exist in the project or are officially supported by the detected project configuration (tsconfig, package.json dependencies).
   - Never assume browser globals (e.g. window, document, alert), DOM APIs, or external dependencies exist.
   - If an equivalent helper function or utility exists in the project (such as dismissMembershipPopup in tests/helpers/dismissPopup.ts), reuse it instead of generating a new approach.
   - The generated code must adapt to the project instead of expecting the project to adapt to the generated code.
6. STRICT PLAYWRIGHT RULES:
   - Generate only valid Playwright TypeScript code using official Playwright APIs.
   - Never invent or assume Playwright methods, assertions, or matchers.
   - Examples of invalid APIs that must NEVER be generated:
     ✗ await expect(page).toHaveLoadState(...)
     ✗ await expect(page).toBeVisible()
     ✗ await expect(page).toBeEnabled()
     ✗ await expect(page).toContainText(...)
     ✗ await expect(page).toHaveText(...)
     ✗ page.toBeVisible()
     ✗ page.clickAndWait()
     ✗ locator.waitUntilVisible()
     ✗ page.waitForElement()
     ✗ page.waitForSelectorAndClick()
     ✗ page.getByCss()
     ✗ page.getByXPath()
     ✗ page.exists()
     ✗ page.isPresent()
   - Use only official Playwright APIs:
     ✅ await page.waitForLoadState('networkidle')
     ✅ await page.waitForURL(...)
     ✅ await page.goto(...)
     ✅ await expect(page).toHaveURL(...)
     ✅ await expect(locator).toBeVisible()
     ✅ await expect(locator).toBeHidden()
     ✅ await expect(locator).toBeEnabled()
     ✅ await expect(locator).toBeDisabled()
     ✅ await expect(locator).toHaveText(...)
     ✅ await expect(locator).toContainText(...)
     ✅ await expect(locator).toHaveValue(...)
     ✅ await expect(locator).toHaveCount(...)
     ✅ await expect(locator).toBeChecked()
    - expect(page) should only be used with valid page matchers such as toHaveURL() or toHaveTitle().
    - Visibility, text, state, attributes, values, and similar assertions must always use a Locator, never a Page.
    - Page loading must always use page.waitForLoadState(), never expect(page).toHaveLoadState().
    - If you are unsure whether a Playwright API exists, do not generate it.
    - NEVER generate fallback expressions (e.g. \`locator || '.selector'\`, \`theSouledStoreLocators.register.errorMsg || '.error-message'\`) in the test script. Access the locator variable directly (e.g., \`theSouledStoreLocators.register.errorMsg\`).
    - Verify that every property referenced on page objects, configs, or locator databases actually exists in the provided context. Never reference non-existent properties.
    - Reuse only existing locator paths from the locators file/object. Check if any locator file exists in the context.
    - If a new locator is required, you must generate and insert it in the locators file (or declare it in LOCATOR_UPDATES) before using it.
    - Insert code only inside an existing async Playwright test or helper function. Never generate code outside a valid scope.
    - Verify that every referenced property exists in the locators file, config, or page objects before presenting/suggesting the code.
6. Return ONLY the corrected code inside the PROPOSED_CODE block below. Do not wrap the entire response in markdown blocks. Output exactly:

PROPOSED_CODE:
\`\`\`typescript
// corrected code here
\`\`\`
`;

  try {
    const result = await model.generateContent(prompt);
    const responseText = result.response.text();

    // Parse corrected code
    const codeRegex = /PROPOSED_CODE:\s*```(?:typescript|ts)?\n([\s\S]*?)```/i;
    const match = responseText.match(codeRegex);
    if (match) {
      return cleanProposedCode(match[1].trim());
    }

    const fallbackRegex = /```(?:typescript|ts)?\n([\s\S]*?)```/i;
    const fallbackMatch = responseText.match(fallbackRegex);
    if (fallbackMatch) {
      return cleanProposedCode(fallbackMatch[1].trim());
    }

    return cleanProposedCode(responseText);
  } catch (err: any) {
    console.error(`  [Repair API Error]: ${err.message || err}`);
    return failedCode;
  }
}

export class CodeGenerationEngine {
  private genAI: GoogleGenerativeAI;
  private modelName: string;

  constructor(apiKey: string, modelName: string) {
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.modelName = modelName;
  }

  public async validateAndSelfHeal(
    suggestions: Suggestion[],
    requirements: Requirement[],
    projectRoot: string,
    specFiles: string[],
    validRequirementIds: string[]
  ): Promise<Suggestion[]> {
    const logFilePath = path.join(projectRoot, 'qa-sync-code-gen.log');
    fs.writeFileSync(logFilePath, '', 'utf-8');

    const writeLog = (msg: string) => {
      const cleanMsg = msg.replace(/\x1B\[\d+m/g, '');
      fs.appendFileSync(logFilePath, cleanMsg + '\n', 'utf-8');
    };

    const originalLog = console.log;
    const originalWarn = console.warn;
    const originalError = console.error;

    // Redirect logs to background file
    console.log = (...args) => writeLog(args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' '));
    console.warn = (...args) => writeLog('[WARN] ' + args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' '));
    console.error = (...args) => writeLog('[ERROR] ' + args.map(a => typeof a === 'string' ? a : JSON.stringify(a)).join(' '));

    try {
      console.log(`\n${BOLD}🔍 Code Generation Engine: Starting validation ${RESET}`);

      // Get baseline compiler output to compare against
      let baselineErrors = '';
      try {
        execSync('npx tsc --noEmit --skipLibCheck', { cwd: projectRoot, stdio: 'pipe' });
      } catch (err: any) {
        baselineErrors = (err.stdout || err.stderr || '').toString();
      }

      const processedSuggestions: Suggestion[] = [];

      for (const sug of suggestions) {
        if (sug.action === 'NONE' || sug.action === 'REMOVE') {
          processedSuggestions.push(sug);
          continue;
        }

        const requirement = requirements.find(r => normalizeRequirementId(r.id) === normalizeRequirementId(sug.requirementId));
        if (!requirement) {
          processedSuggestions.push(sug);
          continue;
        }

        console.log(`\n  Checking proposed code for ${BOLD}${sug.requirementId}${RESET}:`);

        // ── Option 1: Single-pass validation ──
        if (sug.proposedCode) {
          console.log(`    Validating Option 1...`);
          const res = validateProposedCode(
            sug,
            sug.proposedCode,
            false,
            projectRoot,
            specFiles,
            validRequirementIds,
            requirement,
            baselineErrors
          );

          if (res.success) {
            console.log(`    ${GREEN}✓ Option 1 passed validation!${RESET}`);
          } else {
            console.log(`    ${RED}✗ Option 1 failed validation. Showing suggestion with errors noted.${RESET}`);
            console.log(res.errors.map(e => `      - ${e}`).join('\n'));
          }
        }

        // ── Option 2: Single-pass validation ──
        if (sug.proposedCodeOpt2) {
          console.log(`    Validating Option 2...`);
          const res = validateProposedCode(
            sug,
            sug.proposedCodeOpt2,
            true,
            projectRoot,
            specFiles,
            validRequirementIds,
            requirement,
            baselineErrors
          );

          if (res.success) {
            console.log(`    ${GREEN}✓ Option 2 passed validation!${RESET}`);
          } else {
            console.log(`    ${RED}✗ Option 2 failed validation. Showing suggestion with errors noted.${RESET}`);
            console.log(res.errors.map(e => `      - ${e}`).join('\n'));
          }
        }

        processedSuggestions.push(sug);
      }

      return processedSuggestions;
    } finally {
      // Restore console outputs
      console.log = originalLog;
      console.warn = originalWarn;
      console.error = originalError;
    }
  }
}
