import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";

interface Split {
  users: number;
  diagnosed: number;
  control: number;
  tweets: number;
}

interface Target {
  fullCorpusReported: {
    diagnosedUsers: number;
    controlUsers: number;
    allUsers: number;
    diagnosedPrevalence: number;
    controlToDiagnosedRatio: number;
  };
  exactV6Split: {
    train: Split;
    test: Split;
    total: Split;
    folds: Array<{ users: number; diagnosed: number; control: number }>;
    splitManifestSha256: string;
  };
  sourceFiles: Record<string, string>;
  rawEmbeddingManifestSha256: string;
  relevanceDataQuality?: {
    trainNonDigitValues: number;
    testNonDigitValues: number;
  };
}

describe("SetembroBR paper documentation", () => {
  test("keeps exact depression and anxiety split arithmetic auditable", async () => {
    const provenance = JSON.parse(
      await readFile("docs/setembrobr-dataset-provenance.json", "utf8"),
    ) as { targets: Record<"depression" | "anxiety", Target> };

    for (const target of Object.values(provenance.targets)) {
      const { train, test, total, folds } = target.exactV6Split;
      expect(train.users + test.users).toBe(total.users);
      expect(train.diagnosed + test.diagnosed).toBe(total.diagnosed);
      expect(train.control + test.control).toBe(total.control);
      expect(train.tweets + test.tweets).toBe(total.tweets);
      expect(total.diagnosed + total.control).toBe(total.users);
      expect(folds.reduce((sum, fold) => sum + fold.users, 0)).toBe(train.users);
      expect(folds.reduce((sum, fold) => sum + fold.diagnosed, 0)).toBe(train.diagnosed);
      expect(folds.reduce((sum, fold) => sum + fold.control, 0)).toBe(train.control);
      expect(target.fullCorpusReported.allUsers).toBe(total.users);
      expect(target.fullCorpusReported.diagnosedUsers).toBe(total.diagnosed);
      expect(target.fullCorpusReported.controlUsers).toBe(total.control);
      expect(target.fullCorpusReported.diagnosedPrevalence).toBe(0.125);
      expect(target.fullCorpusReported.controlToDiagnosedRatio).toBe(7);
      expect(target.exactV6Split.splitManifestSha256).toMatch(/^[a-f0-9]{64}$/);
      expect(target.rawEmbeddingManifestSha256).toMatch(/^[a-f0-9]{64}$/);
      expect(Object.values(target.sourceFiles).every((hash) => /^[a-f0-9]{64}$/.test(hash))).toBe(true);
    }

    expect(provenance.targets.depression.exactV6Split.total.tweets).toBe(19_419_992);
    expect(provenance.targets.anxiety.exactV6Split.total.tweets).toBe(27_408_040);
    expect(provenance.targets.depression.relevanceDataQuality?.trainNonDigitValues).toBe(23_382);
    expect(provenance.targets.depression.relevanceDataQuality?.testNonDigitValues).toBe(6_377);
  });

  test("includes paper wording, limitations, and restricted-distribution guidance", async () => {
    const markdown = await readFile("docs/setembrobr-dataset-and-splits.md", "utf8");
    const html = await readFile("docs/lock-results-summary.html", "utf8");

    expect(markdown).toContain("Paper-ready description");
    expect(markdown).toContain("Exact v6 depression split");
    expect(markdown).toContain("Exact v6 anxiety split");
    expect(markdown).toContain("Strict-blind split protocol");
    expect(markdown).toContain("The comparison group is pseudo-random rather than clinically confirmed negative");
    expect(markdown).toContain("restricted preservation/authorized-transfer archive");
    expect(markdown).toContain("SanDisk Portable SSD");
    expect(markdown).toContain("all 1,036 hashes passed");
    expect(markdown).toContain("Only the external SanDisk device was erased and formatted");
    expect(markdown).toContain("https://docs.x.com/developer-terms/policy");
    expect(html).toContain("docs/setembrobr-dataset-and-splits.md");
    expect(html).toContain("docs/setembrobr-dataset-provenance.json");
    expect(html).toContain("docs/fedora-setembrobr-archive.json");
  });
});
