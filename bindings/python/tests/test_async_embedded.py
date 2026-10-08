import asyncio
import concurrent.futures
import tempfile
import threading
import unittest

from irongraph import EmbeddedDatabase


class ActualNativeAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_large_result_conversion_keeps_event_loop_running(self):
        with tempfile.TemporaryDirectory() as directory:
            database = await EmbeddedDatabase.open_async(directory, device="cpu", load_embeddings=False)
            await database.query_async("CREATE PROJECT decoderpython")
            body = "Unicode 🦀\0\\\"" * 1024
            await database.query_async(
                "USE decoderpython UNWIND range(0,32767) AS value CREATE (:Record {body:$body,value:value})",
                parameters={"body": body},
            )
            loop = asyncio.get_running_loop()
            ticks = []
            finished = asyncio.Event()
            async def heartbeat():
                while not finished.is_set():
                    ticks.append(loop.time())
                    await asyncio.sleep(0.001)
            pulse = asyncio.create_task(heartbeat())
            try:
                await asyncio.sleep(0.01)
                result = await database.query_async("USE decoderpython MATCH (n:Record) RETURN n")
                await asyncio.sleep(0.01)
            finally:
                finished.set()
                await pulse
                await database.close_async()
            self.assertEqual(len(result["rows"]), 32768)
            self.assertFalse(result["summary"]["truncated"])
            for row in result["rows"]:
                self.assertEqual(row[0]["value"]["properties"]["body"]["value"], body)
            maximum_gap = max(b - a for a, b in zip(ticks, ticks[1:]))
            print(f"PYTHON_ASYNC_DECODE_MAX_GAP_MS={maximum_gap * 1000:.3f}", flush=True)
            self.assertLess(maximum_gap, 0.1, f"Native result conversion blocked asyncio for {maximum_gap:.3f} seconds")

    async def test_lifecycle_parallel_reads_cancellation_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            database = await EmbeddedDatabase.open_async(directory, device="cpu", load_embeddings=False)
            await database.query_async("CREATE PROJECT asyncpython")
            result = await database.query_async("USE asyncpython RETURN 1")
            project = result["catalog"]["project_id"]
            await database.query_async("USE asyncpython CREATE TOPIC events PARTITIONS 1")
            acknowledgement = await database.stream_append_async({"project_id": project, "topic": "events", "partition": 0, "records": [{"key": None, "headers": {}, "value": [1, 2, 3], "create_time_ms": None}]})
            self.assertEqual(acknowledgement["record_count"], 1)
            fetch = {"project_id": project, "topic": "events", "partition": 0, "offset": 0, "max_records": 1, "max_bytes": 4096}
            page = await database.stream_fetch_async(fetch)
            self.assertEqual(page["records"][0][1]["payload"], [1, 2, 3])
            await database.query_async("USE asyncpython UNWIND range(1,256) AS id CREATE (:Work {value:id})")
            active = asyncio.create_task(database.query_async("USE asyncpython MATCH (a:Work), (b:Work), (c:Work) WHERE a.value+b.value+c.value > 0 RETURN sum(a.value)", operation_options={"operation_id": "active-python-cancel", "timeout_ms": 10000}))
            deadline = asyncio.get_running_loop().time() + 3
            while (await database.status_async())["active_operations"] == 0:
                self.assertFalse(active.done(), "native query completed before cancellation")
                self.assertLess(asyncio.get_running_loop().time(), deadline)
                await asyncio.sleep(0.001)
            parallel = await database.query_async("USE asyncpython MATCH (n:Work) RETURN count(n)")
            self.assertEqual(int(parallel["rows"][0][0]["value"]), 256)
            active.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await active
            async def drained():
                while (await database.status_async())["active_operations"]:
                    await asyncio.sleep(0.002)
            await asyncio.wait_for(drained(), 3)
            await database.flush_async()
            await database.snapshot_async()
            await database.close_async()
            reopened = await EmbeddedDatabase.open_async(directory, device="cpu", load_embeddings=False)
            async with reopened:
                self.assertEqual((await reopened.stream_fetch_async(fetch))["high_watermark"], 1)
                self.assertEqual(int((await reopened.query_async("USE asyncpython MATCH (n:Work) RETURN count(n)"))["rows"][0][0]["value"]), 256)

    async def test_pending_capacity_cancellation_keeps_event_loop_running(self):
        with tempfile.TemporaryDirectory() as directory:
            database = await EmbeddedDatabase.open_async(directory, device="cpu", load_embeddings=False)
            await database.query_async("CREATE PROJECT pendingpython")
            loop = asyncio.get_running_loop()
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            loop.set_default_executor(pool)
            started = threading.Event()
            release = threading.Event()
            def hold_worker():
                started.set()
                release.wait(5)
            blocker = loop.run_in_executor(None, hold_worker)
            while not started.is_set():
                await asyncio.sleep(0.001)
            pending = [asyncio.create_task(database.query_async("USE pendingpython CREATE (:Cancelled)")) for _ in range(64)]
            queued = asyncio.create_task(database.query_async("USE pendingpython RETURN 1"))
            try:
                await asyncio.sleep(0.02)
                self.assertFalse(queued.done(), "saturated workers rejected or ran the queued query")
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            finally:
                release.set()
                await blocker
            # Native workers must discard cancelled queued calls before invoking the engine.
            await asyncio.wait_for(queued, 3)
            result = await asyncio.wait_for(database.query_async("USE pendingpython MATCH (n:Cancelled) RETURN count(n)"), 3)
            self.assertEqual(int(result["rows"][0][0]["value"]), 0)
            await database.close_async()


if __name__ == "__main__":
    unittest.main()
