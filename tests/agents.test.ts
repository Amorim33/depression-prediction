import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";

describe("AGENTS.md", () => {
  test("captures strict-blind project memory", async () => {
    const text = await readFile("AGENTS.md", "utf8");
    expect(text).toContain("PostgreSQL");
    expect(text).toContain("Do not call embedding APIs");
    expect(text).toContain("71.51%");
    expect(text).toContain("Never use test labels or test prevalence");
    expect(text).toContain("Only SetembroBR is in scope");
    expect(text).toContain("make db-check-setembrobr");
    expect(text).toContain("conventional commits");
  });
});

