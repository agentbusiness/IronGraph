// Independent Bolt 5 wire client for complete-result measurements; no database library calls.
import assert from 'node:assert/strict'
import { connect } from 'node:net'
import { once } from 'node:events'

const structureTag = Symbol('PackStream structure')
const structure = (signature, ...fields) => ({signature, fields, [structureTag]: true})
function encode(value) {
  const parts = []
  const byte = n => parts.push(Buffer.from([n]))
  const sized = (length, tiny, short, medium, long) => {
    if (tiny !== null && length < 16) byte(tiny + length)
    else if (length < 256) { byte(short); byte(length) }
    else if (length < 65536) { byte(medium); const n = Buffer.allocUnsafe(2); n.writeUInt16BE(length); parts.push(n) }
    else { byte(long); const n = Buffer.allocUnsafe(4); n.writeUInt32BE(length); parts.push(n) }
  }
  const write = value => {
    if (value === null) byte(0xc0)
    else if (typeof value === 'boolean') byte(value ? 0xc3 : 0xc2)
    else if (typeof value === 'bigint' || Number.isInteger(value)) {
      const n = BigInt(value)
      if (n >= -16n && n <= 127n) byte(Number(n & 255n))
      else { byte(0xcb); const data = Buffer.allocUnsafe(8); data.writeBigInt64BE(n); parts.push(data) }
    } else if (typeof value === 'number') {
      byte(0xc1); const data = Buffer.allocUnsafe(8); data.writeDoubleBE(value); parts.push(data)
    } else if (typeof value === 'string') {
      const data = Buffer.from(value); sized(data.length, 0x80, 0xd0, 0xd1, 0xd2); parts.push(data)
    } else if (Array.isArray(value)) {
      sized(value.length, 0x90, 0xd4, 0xd5, 0xd6); value.forEach(write)
    } else if (value[structureTag]) {
      assert(value.fields.length < 16); byte(0xb0 + value.fields.length); byte(value.signature); value.fields.forEach(write)
    } else {
      const entries = Object.entries(value); sized(entries.length, 0xa0, 0xd8, 0xd9, 0xda)
      for (const [key, item] of entries) {write(key); write(item)}
    }
  }
  write(value)
  return Buffer.concat(parts)
}
function decode(input) {
  let offset = 0
  const take = n => { assert(offset + n <= input.length, 'Truncated Bolt value'); const data = input.subarray(offset, offset + n); offset += n; return data }
  const byte = () => take(1)[0]
  const sized = width => width === 1 ? byte() : width === 2 ? take(2).readUInt16BE() : take(4).readUInt32BE()
  const list = n => { const values = []; for (let i = 0; i < n; i++) values.push(read()); return values }
  const map = n => {
    const result = Object.create(null)
    for (let i = 0; i < n; i++) { const key = read(); assert(typeof key === 'string'); assert(!Object.hasOwn(result, key)); result[key] = read() }
    return result
  }
  const read = () => {
    const marker = byte()
    if (marker <= 0x7f) return BigInt(marker)
    if (marker >= 0xf0) return BigInt(marker - 256)
    if (marker >= 0x80 && marker <= 0x8f) return take(marker & 15).toString('utf8')
    if (marker >= 0x90 && marker <= 0x9f) return list(marker & 15)
    if (marker >= 0xa0 && marker <= 0xaf) return map(marker & 15)
    if (marker >= 0xb0 && marker <= 0xbf) {const signature = byte(); return structure(signature, ...list(marker & 15))}
    switch (marker) {
      case 0xc0: return null
      case 0xc1: return take(8).readDoubleBE()
      case 0xc2: return false
      case 0xc3: return true
      case 0xc8: return BigInt(take(1).readInt8())
      case 0xc9: return BigInt(take(2).readInt16BE())
      case 0xca: return BigInt(take(4).readInt32BE())
      case 0xcb: return take(8).readBigInt64BE()
      case 0xcc: case 0xcd: case 0xce: return Buffer.from(take(sized(marker - 0xcb === 3 ? 4 : marker - 0xcb)))
      case 0xd0: case 0xd1: case 0xd2: return take(sized(marker - 0xcf === 3 ? 4 : marker - 0xcf)).toString('utf8')
      case 0xd4: case 0xd5: case 0xd6: return list(sized(marker - 0xd3 === 3 ? 4 : marker - 0xd3))
      case 0xd8: case 0xd9: case 0xda: return map(sized(marker - 0xd7 === 3 ? 4 : marker - 0xd7))
      default: throw Error(`Unknown PackStream marker ${marker}`)
    }
  }
  const value = read(); assert(offset === input.length, 'Trailing Bolt bytes'); return value
}
const typedMap = map => Object.fromEntries(Object.entries(map).map(([name, value]) => [name, typed(value)]))
function typed(value) {
  if (value === null) return {type: 'null'}
  if (typeof value === 'bigint') return {type: 'integer', value: String(value)}
  if (typeof value === 'number') return {type: 'float', value}
  if (typeof value === 'string') return {type: 'string', value}
  if (typeof value === 'boolean') return {type: 'boolean', value}
  if (Buffer.isBuffer(value)) return {type: 'bytes', value: [...value]}
  if (Array.isArray(value)) return {type: 'list', value: value.map(typed)}
  if (!value[structureTag]) return {type: 'map', value: typedMap(value)}
  const f = value.fields
  if (value.signature === 0x4e) return {type: 'node', value: {id: f[3] ?? String(f[0]), labels: f[1], properties: typedMap(f[2])}}
  if (value.signature === 0x52) return {type: 'relationship', value: {id: f[5] ?? String(f[0]), source: f[6] ?? String(f[1]), target: f[7] ?? String(f[2]), relationship_type: f[3], properties: typedMap(f[4])}}
  // Retain all fields of other native structures, including paths and temporal values.
  return {type: 'bolt_structure', value: {signature: value.signature, fields: f.map(typed)}}
}
function frame(value) {
  const payload = encode(value), chunks = []
  for (let offset = 0; offset < payload.length; offset += 65535) {
    const chunk = payload.subarray(offset, offset + 65535), size = Buffer.allocUnsafe(2)
    size.writeUInt16BE(chunk.length); chunks.push(size, chunk)
  }
  chunks.push(Buffer.alloc(2)); return Buffer.concat(chunks)
}

export async function rawBoltClient(url) {
  const address = new URL(url)
  const socket = connect({host: address.hostname, port: Number(address.port)})
  socket.setNoDelay(true)
  await once(socket, 'connect')
  const incoming = socket[Symbol.asyncIterator]()
  let pending = Buffer.alloc(0), offset = 0
  const read = async n => {
    while (pending.length - offset < n) {
      const next = await incoming.next(); assert(!next.done, 'Bolt connection closed')
      pending = Buffer.concat([pending.subarray(offset), next.value]); offset = 0
    }
    const value = pending.subarray(offset, offset + n); offset += n; return value
  }
  const message = async () => {
    const chunks = []
    while (true) {
      const length = (await read(2)).readUInt16BE()
      if (length === 0) {if (chunks.length) return decode(Buffer.concat(chunks)); continue}
      chunks.push(await read(length))
    }
  }
  const success = value => {
    if (value.signature === 0x7f) throw Error(JSON.stringify(value.fields[0], (_, item) => typeof item === 'bigint' ? String(item) : item))
    assert.equal(value.signature, 0x70, 'Expected Bolt SUCCESS'); return value.fields[0]
  }
  const handshake = Buffer.alloc(20); handshake.writeUInt32BE(0x6060b017); handshake.writeUInt32BE(5, 4)
  socket.write(handshake); assert.equal((await read(4)).readUInt32BE(), 5)
  socket.write(frame(structure(0x01, {user_agent: 'IronGraph public protocol benchmark', scheme: 'none'})))
  success(await message())
  // One Bolt connection serializes its protocol messages. Independent concurrent queries get
  // independent sockets through the public benchmark's connection pool below.
  return {
    close: () => socket.destroy(),
    async query(cypher, project, parameters = {}) {
      socket.write(Buffer.concat([frame(structure(0x10, cypher, parameters, {db: project})), frame(structure(0x3f, {n: -1}))]))
      const header = success(await message())
      const result = {columns: header.fields.map(name => ({name, value_type: 'ANY', nullable: true})), rows: [], summary: {}}
      while (true) {
        const value = await message()
        if (value.signature === 0x71) {
          assert.equal(value.fields[0].length, result.columns.length)
          result.rows.push(value.fields[0].map(typed))
        } else {
          const summary = success(value); assert(!summary.has_more)
          result.summary = {truncated: Boolean(summary.truncated), statistics: {rows: result.rows.length}, bookmark: summary.bookmark}
          return result
        }
      }
    },
  }
}

export function rawBoltPool(url) {
  const idle = [], all = new Set()
  return {
    close() { for (const client of all) client.close(); idle.length = 0; all.clear() },
    async query(...args) {
      const client = idle.pop() || await rawBoltClient(url); all.add(client)
      try { const result = await client.query(...args); idle.push(client); return result }
      catch (error) {client.close(); all.delete(client); throw error}
    },
  }
}
