export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function stringField(record: Record<string, unknown>, ...names: string[]): string | undefined {
  for (const name of names) {
    const value = record[name];
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'bigint') return String(value);
  }
  return undefined;
}

export function booleanField(record: Record<string, unknown>, ...names: string[]): boolean | undefined {
  for (const name of names) if (typeof record[name] === 'boolean') return record[name];
  return undefined;
}

export function numberField(record: Record<string, unknown>, ...names: string[]): number | undefined {
  for (const name of names) {
    const value = record[name];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value === 'bigint') return Number(value);
  }
  return undefined;
}

export function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}
