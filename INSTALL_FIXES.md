# MCP Server Installation Fixes

## Issues Fixed (2025-11-11)

### 5. Filesystem Server Access Denied (Latest Fix)

**Problem:** When using the filesystem MCP server, all operations failed with:
```
Error: Access denied - path outside allowed directories: /path/to/dir not in
```

This happened because the filesystem server requires "allowed directories" to be specified as command-line arguments for security, but Code-God was starting it without any directories.

**Solution:** Modified `start_server()` in `mcp_discovery.py:550-603`:
- Added `allowed_dirs` parameter to `start_server()` function
- Special handling for filesystem server to append allowed directories to run command
- Defaults to current working directory and user home directory if not specified
- Command becomes: `node dist/index.js "/home/user" "/path/to/project"`

**Configuration:** The filesystem server now automatically allows access to:
- Current working directory (`os.getcwd()`)
- User home directory (`Path.home()`)
- Any additional directories can be passed via `allowed_dirs` parameter

## Earlier Issues Fixed (2025-11-11)

### 1. TypeScript Monorepo Configuration Issues

**Problem:** MCP servers from monorepo projects (like modelcontextprotocol/servers) had `tsconfig.json` files that extended parent configurations:

```json
{
  "extends": "../../tsconfig.json",
  ...
}
```

When copying individual servers to `~/.codegod/mcp_servers/`, the parent tsconfig.json didn't exist, causing build failures:
- `Cannot read file '/home/sudoje/.codegod/tsconfig.json'`
- Multiple TypeScript compiler errors about module resolution

**Solution:** Added `_fix_monorepo_tsconfig()` function in `mcp_discovery.py:211-299` that:
1. Detects when tsconfig.json extends a parent config
2. Resolves the parent config path from the original repo structure
3. Merges parent compiler options into the local tsconfig
4. Removes the `extends` key, making the config self-contained
5. Falls back to sensible defaults if parent config not found

### 2. Go Project Detection

**Problem:** Go-based MCP servers (like github's official server) were incorrectly marked as TypeScript, causing npm installation to fail with:
- `npm error enoent Could not read package.json`

**Solution:**
- Added Go project detection in `mcp_discovery.py:222-241`
- Updated catalog metadata for `github` server (go language, correct build commands)
- Added prerequisite checks for Go compiler

### 3. Incorrect Catalog Metadata

**Problem:** Multiple servers had wrong language/repo information:
- `github`: Marked as TypeScript, actually Go
- `fetch`: Marked as TypeScript, actually Python
- `postgres`: Pointed to non-existent path

**Solution:** Updated both `mcp_servers_catalog.json` and hardcoded defaults in `mcp_discovery.py`:
- Fixed github server metadata (Go, correct repo)
- Fixed fetch server metadata (Python)
- Fixed postgres server metadata (Python, correct repo)

### 4. Improved Error Diagnostics

**Problem:** When npm installations failed, error messages weren't helpful.

**Solution:** Enhanced error handling in `mcp_discovery.py:402-431`:
- Detects project type mismatches
- Lists directory contents for diagnosis
- Provides specific installation suggestions
- Identifies actual project type when catalog is wrong

## Testing

All fixes verified:

```bash
# TypeScript/npm servers work
✓ filesystem: npm install && build successful
✓ memory: tsconfig fixed, build successful

# Python/pip servers work
✓ fetch: correctly detected as Python
✓ git: pip install successful

# Go detection works
✓ github: correctly identified as Go

# Catalog metadata accurate
✓ All server definitions corrected
```

## Files Modified

1. **mcp_discovery.py**
   - Added `_fix_monorepo_tsconfig()` (lines 211-299)
   - Added Go project detection (lines 222-241)
   - Enhanced npm error handling (lines 407-428)
   - Fixed installation flow (line 480)

2. **mcp_servers_catalog.json**
   - Fixed github server metadata
   - Fixed fetch server metadata
   - Fixed postgres server metadata

## Usage

The fixes are automatic - no user action required. When installing MCP servers:

```bash
./codegod
> /install memory     # Now works correctly
> /install github     # Detects Go, shows helpful message
> /install fetch      # Auto-detects Python
```

## Known Limitations

1. **Go servers require Go installed**: If Go is not installed, installation fails with helpful message directing to https://go.dev/
2. **Monorepo detection**: Assumes standard structure (src/<name>/ with ../../tsconfig.json)
3. **Build commands**: Uses standard commands (tsc, go build, pip install)
