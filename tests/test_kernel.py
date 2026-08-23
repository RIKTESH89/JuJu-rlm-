import asyncio
import contextlib

from kernel.manager import KernelManager


@contextlib.asynccontextmanager
async def kernel():
    manager = KernelManager()
    try:
        yield manager
    finally:
        await manager.shutdown()


def test_startup_is_lazy():
    """No kernel process exists until the first execute()."""

    async def body():
        async with kernel() as km:
            assert km.started is False
            await km.execute("1")
            assert km.started is True

    asyncio.run(body())


def test_namespace_persists_between_calls():
    """A variable set in one execute() is readable in the next."""

    async def body():
        async with kernel() as km:
            first = await km.execute("kept = 6 * 7")
            assert first.error is None

            second = await km.execute("print(kept)")
            assert second.error is None
            assert second.stdout.strip() == "42"

            third = await km.execute("kept")
            assert third.result.strip() == "42"

    asyncio.run(body())


def test_exception_is_captured_not_raised():
    """An error comes back as a traceback string; the host keeps running."""

    async def body():
        async with kernel() as km:
            result = await km.execute("1 / 0")

            assert result.error is not None
            assert "ZeroDivisionError" in result.error
            assert "division by zero" in result.error
            assert "Traceback" in result.error or "----" in result.error

            # the host process is unharmed and the kernel still works
            after = await km.execute("print('still here')")
            assert after.error is None
            assert after.stdout.strip() == "still here"

    asyncio.run(body())


def test_bash_magic_writes_to_stdout():
    """%%bash echo hi returns "hi" on stdout."""

    async def body():
        async with kernel() as km:
            result = await km.execute("%%bash\necho hi")

            assert result.error is None
            assert result.stdout.strip() == "hi"

    asyncio.run(body())


def test_interrupt_stops_runaway_cell_and_namespace_survives():
    """Ctrl-C kills the cell, not the session."""

    async def body():
        async with kernel() as km:
            # warm the kernel up so the sleep below is not spent on startup
            await km.execute("survivor = 'alive'")

            runaway = asyncio.create_task(km.execute("while True: pass"))
            await asyncio.sleep(2)
            await km.interrupt()

            result = await asyncio.wait_for(runaway, timeout=30)
            assert result.error is not None
            assert "KeyboardInterrupt" in result.error

            after = await km.execute("print(survivor)")
            assert after.error is None
            assert after.stdout.strip() == "alive"

    asyncio.run(body())
