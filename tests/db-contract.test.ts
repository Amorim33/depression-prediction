import { describe, expect, test } from "bun:test";
import { countVectorDimensions } from "../src/db-contract.ts";

describe("db contract helpers", () => {
  test("counts pgvector dimensions from text", () => {
    expect(countVectorDimensions("[1,2,3]")).toBe(3);
    expect(countVectorDimensions("[]")).toBe(0);
  });
});

