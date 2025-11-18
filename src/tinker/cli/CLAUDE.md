# Tinker CLI Design Documentation

## Overview

The Tinker CLI is a command-line interface for the Tinker SDK, designed with a focus on fast startup times, modular architecture, and user-friendly output formats. The CLI uses Click framework with custom lazy loading to maintain performance.

## Key Design Decisions

### 1. Lazy Import Strategy with Click

**Decision**: Use Click framework with a custom `LazyGroup` class for lazy loading. Only Click is imported at the module level.

**Rationale**: This ensures that `tinker --help` is lightning fast (<50ms startup time). Users shouldn't have to wait for heavy imports when they just want to see available commands.

**Implementation**:
- Main `__init__.py` only imports `click` and `lazy_group`
- Command modules are loaded only when invoked via `LazyGroup`
- Output formatting imports `rich` only when table output is needed
- JSON module imported only when JSON output is requested
- Version information loaded from `_version.py` only when `tinker version` is used

### 2. Click Framework with LazyGroup

**Decision**: Migrated from argparse to Click, implementing a custom `LazyGroup` class that extends Click's Group to support lazy loading.

**Rationale**:
- Click provides cleaner command structure with decorators
- Better subcommand isolation - each command file is self-contained
- Automatic help generation with better formatting
- Built-in type conversion and validation
- LazyGroup enables fast startup by deferring imports

**LazyGroup Implementation**:
```python
class LazyGroup(click.Group):
    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        # Map of command name to "module.path:command_name"
        self.lazy_subcommands = lazy_subcommands or {}

    def get_command(self, ctx, cmd_name):
        if cmd_name in self.lazy_subcommands:
            # Import only when command is actually invoked
            import_path = self.lazy_subcommands[cmd_name]
            module_name, attr_name = import_path.rsplit(":", 1)
            mod = importlib.import_module(module_name)
            return getattr(mod, attr_name)
```

### 3. Hierarchical Command Structure

**Decision**: Commands are organized hierarchically with main commands and subcommands (e.g., `tinker run list`, `tinker checkpoint info`), plus standalone commands like `tinker version`.

**Rationale**:
- Provides a consistent, predictable interface
- Groups related functionality together
- Makes the CLI extensible for future commands
- Follows common CLI patterns (like `git`, `docker`, etc.)

**Examples**:
- `tinker version` - Show CLI and SDK version
- `tinker run list` - List all training runs
- `tinker run info <run-id>` - Show details of a specific run
- `tinker checkpoint list` - List all checkpoints
- `tinker checkpoint info <checkpoint-id>` - Show checkpoint details

### 4. Output System with Inheritance

**Decision**: Use an abstract base class (`OutputBase`) that all command outputs inherit from. Each command defines its own output class.

**Rationale**:
- Enforces consistent interface across all commands
- Encapsulates output logic with the command that generates it
- Makes it easy to support multiple output formats (table, JSON)
- Keeps related code together in the same module

**Implementation**:
- `OutputBase` in `output.py` defines the contract
- Each command module contains its own output classes (e.g., `RunListOutput`, `RunInfoOutput`)
- Base class handles format selection and rendering

### 5. Self-Contained Command Modules

**Decision**: Each command is a self-contained Click command/group in its own file with a `cli` entry point.

**Rationale**:
- Modular architecture - commands can be developed independently
- Clear separation of concerns
- Easy to add new commands without modifying core files
- Consistent pattern across all commands

**Command Structure**:
```python
# Each command file follows this pattern:
@click.group()  # or @click.command() for simple commands
def cli():
    """Command description."""
    pass

@cli.command()  # For subcommands
def list():
    """Subcommand implementation."""
    pass
```

### 6. Centralized Client Management

**Decision**: All SDK client creation and error handling is centralized in `client.py`.

**Rationale**:
- Single place to handle authentication and connection errors
- Consistent error messages across all commands
- Reusable error handling decorator
- Clean separation of concerns

### 7. Rich Tables for Human-Readable Output

**Decision**: Use the `rich` library for table formatting, kept as an optional dependency.

**Rationale**:
- Provides beautiful, formatted tables with colors and borders
- Handles column width adjustment automatically
- Supports both dark and light terminal themes
- Optional dependency keeps the core package lightweight

### 8. Unix-Style Default Output

**Decision**: Default output is human-readable tables, with `--format json` flag for machine-readable output.

**Rationale**:
- Follows Unix philosophy
- Tables are better for human consumption
- JSON is better for scripting and automation
- Single flag switches between formats consistently

## Performance Optimizations

1. **LazyGroup for deferred imports** - Commands only loaded when invoked
2. **No heavy imports at module level** - Only Click imported initially
3. **Lazy loading** of all SDK dependencies
4. **Progress indicators** that clear themselves
5. **Efficient data fetching** - fetch all data by default instead of pagination

## Error Handling Strategy

1. **User-friendly messages** - Technical errors are translated to helpful messages
2. **Proper exit codes** - Uses TinkerCliError for consistent exit codes
3. **Graceful degradation** - Continue operation when possible
4. **Detailed error info** - Show details when available, traceback only in TTY

### TinkerCliError Exception Pattern

All CLI errors should raise `TinkerCliError` instead of calling `sys.exit()`:

```python
from ..exceptions import TinkerCliError

# Instead of:
print(f"Error: Something went wrong", file=sys.stderr)
sys.exit(1)

# Use:
raise TinkerCliError(
    "Something went wrong",
    "Optional details or help text",
    exit_code=1  # Optional, defaults to 1
)
```

**Benefits:**
- Better testability (can catch exceptions in tests)
- Centralized error formatting in `__main__.py`
- Consistent exit codes across the CLI
- Stack traces preserved for debugging

**Important Notes:**
- The `handle_api_errors` decorator automatically re-raises `TinkerCliError` without modification
- Always catch and convert specific exceptions to `TinkerCliError` with helpful messages
- The main error handler in `__main__.py` handles printing to stderr and exiting

## Future Extensibility

The architecture supports easy addition of:

### New Commands
- Create new module in `commands/` directory
- Define output classes in the same module if needed
- Add command to lazy_subcommands in `__init__.py`

### New Subcommands
- Add new Click command decorator to existing command module
- Define corresponding output class if needed
- Subcommands automatically discovered by Click

### New Output Formats
- Override `print()` method in `OutputBase`
- Or add new format handling to base class

## Testing Guidelines

1. **Startup time**: `time tinker --help` should be <50ms
2. **Import verification**: Check that modules aren't imported unnecessarily
3. **Output formats**: Test both table and JSON output
4. **Error cases**: Test with missing auth, invalid IDs, network errors
5. **Empty results**: Ensure graceful handling of no data

## Module Structure

```
cli/
├── __init__.py           # Main entry with LazyGroup configuration
├── __main__.py           # Module execution support
├── lazy_group.py         # LazyGroup implementation for lazy loading
├── output.py             # OutputBase class and formatting utilities
├── client.py             # SDK client creation and error handling
├── commands/
│   ├── __init__.py       # Command module marker
│   ├── version.py        # Version command
│   ├── run.py            # Run commands and output classes
│   └── checkpoint.py     # Checkpoint commands and output classes
└── CLAUDE.md             # This documentation
```

## Command Examples

```bash
# Show version
tinker version

# List all training runs
tinker run list

# Show run details
tinker run info run-abc123

# List all checkpoints
tinker checkpoint list

# List checkpoints for specific run
tinker checkpoint list run-abc123

# Show checkpoint details
tinker checkpoint info ckpt-xyz789

# JSON output
tinker --format json run list
tinker --format json checkpoint list
```

## Dependencies

### Required
- Python 3.11+
- tinker SDK (main package)
- click>=8.0.0 (CLI framework)

### Optional
- `rich` - For table formatting (installed with `pip install tinker[cli]`)

## Migration from Argparse to Click

### Key Changes:
1. **Command Definition**: Decorators instead of `parser.add_argument()`
2. **Lazy Loading**: Custom `LazyGroup` instead of manual dispatch
3. **Context Passing**: Click's context system for sharing format option
4. **Error Handling**: Click handles exits and error formatting
5. **Help Generation**: Automatic from docstrings and decorators

### Benefits:
- Cleaner, more Pythonic code
- Better command organization
- Built-in testing utilities
- Easier to extend with plugins
- More consistent behavior across commands

## Maintenance Notes

1. **Keep imports lazy** - Use LazyGroup for all commands
2. **Test startup time** - Regularly verify fast startup is maintained
3. **Follow Click patterns** - Use decorators and context properly
4. **Document changes** - Update this file when making architectural changes
5. **Maintain consistency** - All commands should follow the same structure
