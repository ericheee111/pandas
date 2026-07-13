import platform

IS_ARM = platform.machine() in ("aarch64", "arm64", "armv8l", "armv7l")
