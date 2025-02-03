"""Test runner for executing code examples in docstrings."""

import ast
import textwrap
import pytest
import re
import logging
import importlib
from pathlib import Path
from typing import List, Dict, Any


pytestmark = pytest.mark.anyio

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_code_blocks(docstring: str) -> List[str]:
    """Extract Python code blocks (```python ... ```) from a docstring."""
    if not docstring:
        return []
    pattern = r"```(?:python[ ]*)?(.*?)\s*```"

    matches = re.finditer(pattern, docstring, re.DOTALL)
    blocks = [textwrap.dedent(m.group(1)).strip() for m in matches]
    logger.debug(f"Found {len(blocks)} code blocks in docstring")
    for i, block in enumerate(blocks):
        logger.debug(f"Code block {i}:\n{block}")
    return blocks


class DocstringVisitor(ast.NodeVisitor):
    """AST visitor that finds function definitions and extracts docstring code blocks."""

    def __init__(self):
        super().__init__()
        self.functions = {}
        self.current_class = None
        self.current_module = None

    def visit_FunctionDef(self, node):
        name = node.name
        if self.current_class:
            name = f"{self.current_class}.{name}"
        if self.current_module:
            name = f"{self.current_module}.{name}"

        docstring = ast.get_docstring(node)
        if docstring:
            code_blocks = extract_code_blocks(docstring)
            if code_blocks:
                self.functions[name] = {
                    "name": name,
                    "examples": code_blocks,
                    "module": self.current_module,
                }
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class


def get_module_functions(module_path: str) -> Dict[str, Any]:
    """Parse a Python file and return docstring code blocks from its functions."""
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        # Construct a module name from the path, so importlib can locate it
        src_dir = Path(__file__).parent.parent / "src"
        rel_path = Path(module_path).relative_to(src_dir)
        module_name = str(rel_path.with_suffix("")).replace("/", ".")

        visitor = DocstringVisitor()
        visitor.current_module = module_name
        visitor.visit(tree)
        return visitor.functions
    except Exception as e:
        logger.error(f"Error processing module {module_path}: {e}")
        return {}


def extract_readme_examples():
    """Extract Python code blocks from README.md."""
    readme_path = Path(__file__).parent.parent / "README.md"
    if not readme_path.exists():
        logger.warning(f"README.md not found at {readme_path}")
        return []

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    code_blocks = extract_code_blocks(content)
    test_cases = []
    for i, block in enumerate(code_blocks):
        test_id = f"README.md::example_{i}"
        logger.info(f"Adding README test case: {test_id}")
        test_cases.append(
            pytest.param(
                None,  # No module
                "README.md",  # Function name placeholder
                [block],  # Single example
                id=test_id,
            )
        )
    return test_cases


def collect_docstring_tests():
    """Collect all docstring Python code blocks from the 'src/' tree and README.md."""
    src_dir = Path(__file__).parent.parent / "src"
    logger.info(f"Scanning for Python files in {src_dir}")
    py_files = list(src_dir.rglob("*.py"))
    logger.info(f"Found {len(py_files)} Python files")

    test_cases = []

    test_cases.extend(extract_readme_examples())

    # Add docstring examples
    for py_file in py_files:
        logger.debug(f"Processing file: {py_file}")
        funcs = get_module_functions(str(py_file))
        logger.debug(f"Found {len(funcs)} functions with examples in {py_file}")

        for func_name, details in funcs.items():
            test_id = f"{py_file.relative_to(src_dir)}::{func_name}"
            logger.info(f"Adding test case: {test_id}")
            test_cases.append(
                pytest.param(
                    details["module"],
                    func_name,
                    details["examples"],  # Pass all examples together
                    id=test_id,
                )
            )
    logger.info(f"Collected {len(test_cases)} test cases")
    return test_cases


@pytest.mark.parametrize("module_name,func_name,code_blocks", collect_docstring_tests())
@pytest.mark.asyncio_cooperative
@pytest.mark.langsmith
async def test_docstring_example(
    module_name: str, func_name: str, code_blocks: List[str]
):
    """Execute all docstring code blocks from a function in sequence, maintaining state."""
    # For README examples, we don't need to import anything
    if module_name is None:
        obj = None
        module = None
    else:
        # Dynamically import the module
        module = importlib.import_module(module_name)

        # Find the function object inside the module
        obj = module
        func_name_ = (
            func_name[len(obj.__name__) :].lstrip(".")
            if func_name.startswith(obj.__name__)
            else func_name
        )
        for part in func_name_.split("."):
            obj = getattr(obj, part)

    # Prepare a fresh namespace that will be shared across all code blocks
    namespace = {
        "__name__": f"docstring_example_{func_name.replace('.', '_')}",
        "__file__": getattr(module, "__file__", None),
        module_name.split(".")[-1]: module,
        func_name.split(".")[-1]: obj,
    }

    # Execute each code block in sequence, maintaining the namespace
    for i, code_block in enumerate(code_blocks):
        try:
            if "await " in code_block:
                # For async blocks, we need to capture the locals after execution
                wrapped_code = f"""
async def _test_docstring():
    global_ns = globals()
{textwrap.indent(code_block, '    ')}
    # Update namespace with all locals
    global_ns.update(locals())
"""
                exec(wrapped_code, namespace, namespace)
                await namespace["_test_docstring"]()
            else:
                exec(code_block, namespace, namespace)

            # Log what was added to namespace
        except Exception as e:
            logger.error(f"Error executing code block {i} for {func_name}: {e}")
            logger.error(f"Code block contents:\n{code_block}")
            raise
