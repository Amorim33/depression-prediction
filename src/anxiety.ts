export interface NestedCrossFitRecord {
  outerFold: number;
  innerValidationFold: number;
  fitFolds: number[];
}

export function validateNestedCrossFitRecords(records: readonly NestedCrossFitRecord[]): boolean {
  return records.length > 0 && records.every(
    (record) =>
      !record.fitFolds.includes(record.outerFold) &&
      !record.fitFolds.includes(record.innerValidationFold),
  );
}
