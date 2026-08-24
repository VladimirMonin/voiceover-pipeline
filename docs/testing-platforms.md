# Platform test taxonomy

The offline test suite distinguishes host-dependent checks from platform-contract checks:

- `native_linux` requires a real Linux/POSIX capability and must skip on other operating systems.
- `native_windows` requires a real Windows capability and must skip on other operating systems.
- `platform_simulated` injects platform state and fake native APIs, so it runs on every host. It must not import or call an unavailable native module through the simulated path.

Run the complete offline suite with `uv run pytest -q`. Select a class of checks with, for example, `uv run pytest -m native_windows` or `uv run pytest -m platform_simulated`. A skip for a native marker is expected when its operating system is unavailable; broad skips must not be used to hide cross-platform contract failures.