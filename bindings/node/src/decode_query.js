(async function decodeQuery(result, next) {
  let offset = 0
  for (;;) {
    const batch = next()
    for (const row of batch.rows) result.rows[offset++] = row
    if (batch.done) return result
    await new Promise(setImmediate)
  }
})
