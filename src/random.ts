export interface RandomSource {
  next(): number;
  nextInt(maxExclusive: number): number;
}

export function deriveSeed(seed: number, salt: number): number {
  let value = (seed ^ 0x9e3779b9 ^ salt) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x85ebca6b) >>> 0;
  value = Math.imul(value ^ (value >>> 13), 0xc2b2ae35) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

export function createSeededRandom(seed: number): RandomSource {
  let state = seed >>> 0;
  return {
    next() {
      state = (state + 0x6d2b79f5) >>> 0;
      let value = state;
      value = Math.imul(value ^ (value >>> 15), value | 1);
      value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    },
    nextInt(maxExclusive: number) {
      return Math.floor(this.next() * maxExclusive);
    },
  };
}

export function shuffleInPlace<T>(values: T[], random: RandomSource): void {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const swapIndex = random.nextInt(index + 1);
    [values[index], values[swapIndex]] = [values[swapIndex]!, values[index]!];
  }
}

