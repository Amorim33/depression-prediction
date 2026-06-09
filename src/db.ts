import postgres from "postgres";

export interface DatabaseClient {
  sql: postgres.Sql;
  end(): Promise<void>;
}

export function createDatabaseClient(databaseUrl: string): DatabaseClient {
  const sql = postgres(databaseUrl, {
    max: 1,
    prepare: false,
    onnotice: () => undefined,
  });
  return {
    sql,
    async end() {
      await sql.end({ timeout: 1 });
    },
  };
}

export function assertSafeIdentifier(identifier: string): string {
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/u.test(identifier)) {
    throw new Error(`Unsafe SQL identifier: ${identifier}`);
  }
  return identifier;
}

