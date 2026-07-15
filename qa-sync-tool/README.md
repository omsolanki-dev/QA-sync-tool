# Playwright QA Sync Tool

The **Playwright QA Sync Tool** is an autonomous synchronization engine that maps, verifies, and keeps your Playwright test suites in perfect sync with business requirement documents. It detects gaps, proposes additions/modifications, updates locators, and ensures your test suite successfully compiles before making changes.

---

## 🚀 Key Features

* **Universal Requirement Parsing**: Full support for tables (including merged cells), bullet lists, numbered lists, nested lists, Roman numerals, alphabetic lists, and plain text.
* **Stable Content Matching**: Uses content fingerprinting and semantic similarity to preserve mapping links even if requirement IDs or document layouts change.
* **Describe-Block Inheritance**: Nesting support—enclosing `test.describe` comments automatically propagate mapping properties down to inner tests.
* **Self-Healing Loop**: Runs test code validation in the background using `tsc --noEmit` and heals compiler errors automatically up to 3 times before updating files.
* **Merged ES6 Imports**: Groups and merges new import modules, preventing duplicate lines and keeping libraries cleanly sorted.

---

## 🛠 Standalone Integration Manual

Follow these exact steps to copy and execute this tool inside any target Playwright TypeScript project.

### 📋 Prerequisites
Make sure your system has the following installed:
* **Node.js**: `v16.x` or higher
* **TypeScript**: Installed locally or globally in the project

---

### 📥 Step-by-Step Installation

#### Step 1: Copy the Tool Files
Copy the `qa-sync-tool/` folder directly into the root directory of your target Playwright project:
```text
your-playwright-project/
├── qa-sync-tool/
│   ├── index.ts
│   ├── testScanner.ts
│   ├── aiAnalyzer.ts
│   ├── codeGenerator.ts
│   └── README.md
├── package.json
└── ...
```

#### Step 2: Initialize Project Configuration Files (If missing)
If your target folder does not have a `package.json` or `tsconfig.json`, run these commands first in the terminal root of your project:
```bash
# 1. Initialize npm package configuration (creates package.json)
npm init -y

# 2. Initialize TypeScript configuration (creates tsconfig.json)
npx tsc --init
```

#### Step 3: Install Dependencies (Copy-paste this single command)
Run this single command in your project terminal root to install all required libraries:
```bash
npm install @playwright/test @google/generative-ai mammoth dotenv tsx typescript @types/node --save-dev
```

#### Step 4: Install Playwright Browsers (If missing)
To download and configure Playwright browsers in your workspace:
```bash
npx playwright install
```

#### Step 5: Configure Environment Variables (`.env`)
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Create or edit the `.env` file in the root of your project and paste your keys:
```env
# Mandatory Google Gemini API Key for coverage analysis and code draft generation
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

#### Step 6: Setup Supporting Config & Locator Modules
The sync engine resolves config and selector references automatically. Ensure the following files are present:
1. **Requirements Document**: Add your requirements specifications as a `.docx` file (e.g. `Requirement-txt.docx`) in the project root.
2. **Locators File**: Create a centralized locators file (e.g., `config/Locators.ts`) exporting selector properties:
   ```typescript
   export const theSouledStoreLocators = {
     nav: {
       logo: '.brand-logo',
       registerLink: 'a[href="/register"]'
     },
     register: {
       emailInput: 'input[name="email"]',
       submitBtn: 'button[type="submit"]',
       errorMsg: '.error-message',
       successMsg: '.success-message'
     }
   };
   ```
3. **Config File**: Create a base configuration file (e.g., `config/config.ts`) exporting execution values:
   ```typescript
   export const config = {
     baseURL: 'https://www.thesouledstore.com'
   };
   ```

---

## 🏃 Running the Synchronizer

You can run the script directly with `npx` or create a package script shortcut:

### Option A: Direct Execution (Recommended for Node v24+)
Run the runner index from your terminal:
```bash
npx tsx qa-sync-tool/index.ts
```

### Option B: Package Script Shortcut
1. Open your project's `package.json` file.
2. Add a start script script inside the `"scripts"` object:
   ```json
   "scripts": {
     "qa-sync": "tsx qa-sync-tool/index.ts"
   }
   ```
3. Execute the script by running:
   ```bash
   npm run qa-sync
   ```

---

## 🔄 Core Workflow

1. **Scan**: Searches for a `.docx` file in the root directory, reads requirements line-by-line, and lists tests from all `.spec.ts` files inside your tests folder.
2. **Analysis**: Correlates test assertions with specific requirement goals and expected results.
3. **Draft Selection**: Prompts you to review suggestions one-by-one or apply them in bulk.
4. **Compile & Heal**: Saves the code draft to a temporary buffer and validates it using `npx tsc --noEmit`. If compile warnings occur, the self-healing engine rewrites the code using generative feedback.
5. **Deduplicate Imports**: Standardizes import structures, removes duplicate lines, and saves changes back to the target `.spec.ts` file.
