#!/usr/bin/env python3
# /// script
# dependencies = [
#   "pydoc-markdown>=4.8.0",
#   "pyyaml>=6.0",
# ]
# ///


import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


def cd_to_project_root():
    """Change to the project root (parent of the scripts directory)."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)
    print(f"Changed to project root: {project_root}")


class ModuleAnalyzer:
    """Analyze Python modules to extract public API information."""

    def __init__(self, src_path: Path):
        self.src_path = src_path

    def get_module_exports(self, module_path: Path) -> List[str]:
        """Extract __all__ exports from a module."""
        try:
            content = module_path.read_text()
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            if isinstance(node.value, ast.List):
                                return [
                                    elt.s for elt in node.value.elts
                                    if isinstance(elt, ast.Str)
                                ] or [
                                    elt.value for elt in node.value.elts
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                                ]
        except Exception as e:
            print(f"Warning: Could not parse {module_path}: {e}")
        return []

    def find_all_modules(self) -> Dict[str, Path]:
        """Find all Python modules in the package."""
        modules = {}
        tinker_path = self.src_path / "tinker"

        for py_file in tinker_path.rglob("*.py"):
            # Skip test files and private modules
            if any(part.startswith('test') or part.startswith('_test') for part in py_file.parts):
                continue
            if '__pycache__' in py_file.parts:
                continue

            # Calculate module name
            relative_path = py_file.relative_to(self.src_path)
            module_parts = list(relative_path.parts[:-1])  # Remove .py file
            module_parts.append(relative_path.stem)

            # Skip __init__ files in module name
            if module_parts[-1] == '__init__':
                module_parts = module_parts[:-1]

            module_name = '.'.join(module_parts)
            if module_name:  # Skip empty module names
                modules[module_name] = py_file

        return modules


class DocumentationGenerator:
    """Generate documentation using pydoc-markdown."""

    def __init__(self, config_path: Path, output_dir: Path):
        self.config_path = config_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = ModuleAnalyzer(Path('src'))

    def run_pydoc_markdown(self, modules: List[str], output_file: Path) -> bool:
        """Run pydoc-markdown for specific modules."""
        try:
            # Build the command
            cmd = ['pydoc-markdown', 'pydoc-markdown.yml', '-I', 'src']

            # Add modules
            for module in modules:
                cmd.extend(['-m', module])

            # Run pydoc-markdown
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                # Write output to file
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(result.stdout)
                print(f"Generated: {output_file}")
                return True
            else:
                print(f"Error generating {output_file}: {result.stderr}")
                return False

        except Exception as e:
            print(f"Exception generating {output_file}: {e}")
            return False

    def generate_public_interfaces(self):
        """Generate documentation for public interface classes."""
        print("\n=== Generating Public Interfaces Documentation ===")

        # Generate individual pages for each client
        client_modules = [
            ('ServiceClient', 'tinker.lib.public_interfaces.service_client'),
            ('TrainingClient', 'tinker.lib.public_interfaces.training_client'),
            ('SamplingClient', 'tinker.lib.public_interfaces.sampling_client'),
            ('RestClient', 'tinker.lib.public_interfaces.rest_client'),
            ('APIFuture', 'tinker.lib.public_interfaces.api_future'),
        ]

        for class_name, module in client_modules:
            output_file = self.output_dir / f'{class_name.lower().replace("_", "-")}.md'
            self.run_pydoc_markdown([module], output_file)

    def generate_all_types(self):
        """Generate complete types reference."""
        print("\n=== Generating Complete Types Reference ===")

        # Get all type modules
        all_modules = self.analyzer.find_all_modules()
        type_modules = [m for m in all_modules.keys() if m.startswith('tinker.types')]

        if type_modules:
            output_file = self.output_dir / 'types.md'
            self.run_pydoc_markdown(type_modules, output_file)

    def generate_exceptions(self):
        """Generate exception hierarchy documentation."""
        print("\n=== Generating Exception Documentation ===")

        output_file = self.output_dir / 'exceptions.md'
        self.run_pydoc_markdown(['tinker._exceptions'], output_file)

    def generate_nextra_meta(self):
        """Generate _meta.json for Nextra navigation."""
        print("\n=== Generating Nextra Navigation Metadata ===")

        meta = {
            "serviceclient": "ServiceClient",
            "trainingclient": "TrainingClient",
            "samplingclient": "SamplingClient",
            "restclient": "RestClient",
            "apifuture": "APIFuture",
            "types": "Parameters",
            "exceptions": "Exceptions"
        }

        meta_file = self.output_dir / '_meta.json'
        meta_file.write_text(json.dumps(meta, indent=2))
        print(f"Generated: {meta_file}")

    def generate_all(self):
        """Generate all documentation."""
        print("Starting documentation generation...")
        print(f"Output directory: {self.output_dir}")

        # Generate documentation for each category
        self.generate_public_interfaces()
        self.generate_all_types()
        self.generate_exceptions()

        # Generate Nextra metadata
        self.generate_nextra_meta()

        print("\n=== Documentation Generation Complete ===")
        print(f"Markdown files generated in: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.rglob("*.md")):
            print(f"  - {file.relative_to(self.output_dir)}")


def main():
    """Main entry point."""
    # Change to project root first
    cd_to_project_root()

    # Paths
    project_root = Path.cwd()
    config_path = project_root / 'pydoc-markdown.yml'
    output_dir = project_root / 'docs' / 'api'

    # Check if config exists
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        print("Please run this script from the project root directory")
        sys.exit(1)

    # Create generator and run
    generator = DocumentationGenerator(config_path, output_dir)
    generator.generate_all()

    # Print usage instructions
    print("\n" + "=" * 50)
    print("To use these docs in your Nextra project:")
    print("1. Copy the docs/api directory to your Nextra project")
    print("2. The markdown files are ready to use with Nextra")
    print("3. Navigation structure is defined in _meta.json")
    print("\nTo regenerate docs after code changes:")
    print("  uv run scripts/generate_docs.py")


if __name__ == "__main__":
    main()
