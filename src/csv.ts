export type CsvRecord = Record<string, string>;

export function parseCsv(text: string): CsvRecord[] {
  const rows = parseRows(text);
  if (rows.length === 0) return [];
  const header = rows[0]!;
  return rows.slice(1).filter((row) => row.some((cell) => cell.length > 0)).map((row) => {
    const record: CsvRecord = {};
    for (let index = 0; index < header.length; index += 1) {
      record[header[index]!] = row[index] ?? "";
    }
    return record;
  });
}

export function writeCsv(headers: readonly string[], rows: readonly Record<string, string | number | null>[]): string {
  const lines = [
    headers.map(escapeCell).join(","),
    ...rows.map((row) => headers.map((header) => escapeCell(row[header] ?? "")).join(",")),
  ];
  return `${lines.join("\n")}\n`;
}

function parseRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]!;
    const next = text[index + 1];

    if (quoted) {
      if (char === '"' && next === '"') {
        cell += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
      continue;
    }

    if (char === '"') quoted = true;
    else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (char !== "\r") {
      cell += char;
    }
  }

  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function escapeCell(value: string | number | null): string {
  const text = value === null ? "" : String(value);
  if (/[",\n\r]/u.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

