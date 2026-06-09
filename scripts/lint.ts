import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const forbidden = [
  "sweepThreshold(test",
  'test_data["labels"]',
  "train_one(model, device, train_data, test_data",
  "train_one(...test_data",
];

const files = await collectFiles(process.cwd());
const findings: string[] = [];
for (const file of files) {
  if (file.endsWith("scripts/lint.ts") || file.endsWith("src/audit.ts") || file.includes("tests/")) continue;
  const text = await readFile(file, "utf8");
  for (const pattern of forbidden) {
    if (text.includes(pattern)) findings.push(`${file}: forbidden strict-blind pattern: ${pattern}`);
  }
}

if (findings.length > 0) {
  console.error(findings.join("\n"));
  process.exit(1);
}
console.log("lint ok");

async function collectFiles(dir: string): Promise<string[]> {
  const out: string[] = [];
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    if ([".git", "node_modules", "outputs"].includes(entry.name)) continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await collectFiles(path));
    else if (/\.(ts|py)$/u.test(entry.name)) out.push(path);
  }
  return out;
}

