"""
Trick Search Agent (TSA)

An always-running agent that catalogues algorithmic efficiencies from across
the sciences that may be applicable for model training. Documents them as
markdown notes in the tricks/ folder.

The TSA searches for:
1. Matrix decompositions and representations (WY, Householder, DPLR, etc.)
2. Parallelization tricks (chunking, scan operations, prefix sums)
3. Approximation methods (linear attention, random features, etc.)
4. Numerical stability techniques (normalization, gradient clipping strategies)
5. Kernel optimization patterns (FlashAttention-style, tiling, fusion)
6. Algebraic structures that enable efficient computation

Each trick is documented with:
- Name and source (paper, field, etc.)
- What it does (the mathematical operation or transformation)
- Why it's efficient (complexity reduction, hardware utilization, etc.)
- Applicability to model architectures (what layer types benefit)
- Known limitations or tradeoffs
- Classification: expressivity / efficiency / flexibility gain

Usage:
    # One-shot search
    python -m agents.trick_search "matrix decomposition tricks for SSMs"

    # Interactive mode
    python -m agents.trick_search --interactive

    # Programmatic usage
    from agents.trick_search import run_trick_search
    async for msg in run_trick_search("find parallelization techniques"):
        print(msg)
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, AsyncIterator

from claude_agent_sdk import query, ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition


TRICKS_DIR = Path(__file__).parent.parent / "tricks"
PAPERS_DIR = Path(__file__).parent.parent / "papers"
PROJECT_ROOT = Path(__file__).parent.parent


# =============================================================================
# System Prompts
# =============================================================================

TRICK_SEARCH_SYSTEM_PROMPT = """You are the Trick Search Agent (TSA), an expert in finding and documenting
algorithmic efficiencies that can be applied to neural network architectures.

Your goal is to search for, analyze, and document computational "tricks" - techniques from
numerical linear algebra, signal processing, parallel computing, and other fields that enable
more efficient neural network training and inference.

## Human Feedback

**IMPORTANT**: Always check `human_feedback.md` at the project root for ongoing instructions and preferences from the human researcher. This feedback takes priority and should guide your search focus and documentation approach.

## What to Search For

1. **Matrix decompositions**: WY representation, Householder reflections, DPLR, low-rank factorizations
2. **Parallelization tricks**: Chunkwise scans, prefix sums, parallel associative operations
3. **Approximation methods**: Linear attention, random features, kernel approximations
4. **Numerical stability**: Normalization techniques, gradient clipping, mixed precision strategies
5. **Kernel optimizations**: FlashAttention-style tiling, operator fusion, memory-efficient patterns
6. **Algebraic structures**: Semirings, monoids, and other structures enabling efficient computation

## How to Document

For each trick found:

### 1. Download the Source Paper

Download PDFs to the papers/ folder using curl:
```bash
curl -L -o papers/[paper-slug].pdf "[pdf-url]"
```

For arXiv papers, use the PDF URL format: https://arxiv.org/pdf/XXXX.XXXXX.pdf

Name the PDF file using a slug matching the trick name (e.g., `woodbury-identity.pdf`).

### 2. Read the PDF

Use the Read tool to read the downloaded PDF and extract the key mathematical details:
```
Read papers/[paper-slug].pdf
```

The Read tool can directly read PDF files. For long papers, you can specify page ranges.
Focus on sections containing:
- The main algorithm or mathematical formulation
- Complexity analysis
- Implementation details

### 3. Create Documentation

**IMPORTANT: Numbering**
- Each trick must be numbered sequentially (e.g., 001-trick-name.md, 002-trick-name.md)
- Check existing tricks to find the next available number
- Use 3-digit zero-padded numbers (001, 002, ..., 999)

Create a markdown file in the tricks/ folder with this exact format:

**Filename**: `XXX-trick-name-slug.md` (where XXX is the next available number)

```markdown
# XXX: [Trick Name]

**Category**: [decomposition|parallelization|approximation|stability|kernel|algebraic]
**Gain type**: [expressivity|efficiency|flexibility]
**Source**: [Original paper or field]
**Paper**: [papers/paper-slug.pdf] (local path to downloaded PDF)
**Documented**: [YYYY-MM-DD]

## Description

[What it does and why it matters]

## Mathematical Form

Use LaTeX notation for all mathematical expressions. Use `$...$` for inline math and `$$...$$` for display equations.

**Core Operation:**

$$
[Main equation, e.g., Y = (I + UV^T)^{{-1}}X = X - U(I + V^TU)^{{-1}}V^TX]
$$

**Key Definitions:**

- $A \in \mathbb{{R}}^{{n \times n}}$ — [description]
- $U, V \in \mathbb{{R}}^{{n \times k}}$ — [description]

**Derivation (if applicable):**

[Step-by-step derivation using LaTeX]

## Complexity

| Operation | Naive | With Trick |
|-----------|-------|------------|
| [Op 1] | $O(n^3)$ | $O(n^2 k)$ |
| [Op 2] | $O(n^2)$ | $O(n \log n)$ |

**Memory:** $O(nk)$ vs $O(n^2)$

## Applicability

[Which architectures or layer types benefit]

## Limitations

[Known tradeoffs or constraints]

## Implementation Notes

```python
# Pseudocode or key implementation insight
```

## References

- [Paper 1]
- [Paper 2]
```

## LaTeX Guidelines

- Use `$...$` for inline math: "the matrix $A \in \mathbb{{R}}^{{n \times n}}$"
- Use `$$...$$` for display equations on their own line
- Common symbols:
  - Matrices: $A, B, X, Y$ (uppercase)
  - Vectors: $x, y, v$ (lowercase)
  - Real numbers: $\mathbb{{R}}$
  - Transpose: $A^T$ or $A^\top$
  - Inverse: $A^{{-1}}$
  - Big-O: $O(n^2)$, $O(n \log n)$
  - Summations: $\sum_{{i=1}}^{{n}}$
  - Products: $\prod_{{i=1}}^{{n}}$
  - Norms: $\|x\|_2$, $\|A\|_F$

## Working Directories

- Tricks folder: {tricks_dir}
- Papers folder: {papers_dir}

Always read existing tricks first to avoid duplicates and match the documentation style.
Ensure the papers/ directory exists before downloading (create it if needed).
"""

PAPER_READER_PROMPT = """You are a specialized paper reader agent. Your job is to:
1. Fetch and read academic papers or technical blog posts
2. Extract the key algorithmic insights
3. Identify computational tricks that reduce complexity or improve efficiency
4. Return a structured summary with mathematical details in LaTeX notation

Focus on:
- The core technique or transformation (express as LaTeX equations)
- Complexity analysis (before/after) using Big-O notation: $O(n^2)$, $O(n \log n)$
- Key matrix/tensor operations with proper notation: $A \in \mathbb{{R}}^{{n \times d}}$
- Implementation considerations
- Applicability to neural networks

Always use LaTeX for mathematical expressions:
- Inline: $x \in \mathbb{{R}}^n$
- Display: $$Y = XW + b$$
"""


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_papers_dir() -> Path:
    """Ensure the papers directory exists."""
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    return PAPERS_DIR


def list_existing_papers() -> list[str]:
    """List all downloaded paper files."""
    if not PAPERS_DIR.exists():
        return []
    return sorted([f.name for f in PAPERS_DIR.glob("*.pdf")])


def get_next_trick_number() -> int:
    """Get the next available trick number."""
    if not TRICKS_DIR.exists():
        return 1

    existing = []
    for f in TRICKS_DIR.glob("*.md"):
        # Extract number from filename like "001-trick-name.md"
        import re
        match = re.match(r'(\d+)-', f.stem)
        if match:
            existing.append(int(match.group(1)))

    return max(existing, default=0) + 1


def list_existing_tricks() -> list[str]:
    """List all existing trick files."""
    if not TRICKS_DIR.exists():
        return []
    return sorted([f.stem for f in TRICKS_DIR.glob("*.md")])


def trick_exists(trick_name: str) -> bool:
    """
    Check if a trick with this name is already documented.
    Handles both numbered (001-name) and non-numbered (name) formats.
    """
    slug = trick_name.lower().replace(" ", "-").replace("/", "-")

    # Check exact match first
    if (TRICKS_DIR / f"{slug}.md").exists():
        return True

    # Check if any numbered trick has this slug
    import re
    for f in TRICKS_DIR.glob("*.md"):
        # Strip number prefix if present
        stem = f.stem
        match = re.match(r'\d+-(.+)', stem)
        if match:
            base_name = match.group(1)
            if base_name == slug:
                return True

    return False


def get_trick_summary() -> str:
    """Generate a summary table of all documented tricks."""
    tricks = list_existing_tricks()
    if not tricks:
        return "No tricks documented yet.\n"

    lines = [
        "| Trick | Category | Gain Type |",
        "|-------|----------|-----------|",
    ]

    for trick_name in tricks:
        filepath = TRICKS_DIR / f"{trick_name}.md"
        content = filepath.read_text()

        # Parse category and gain type from frontmatter
        category = "?"
        gain_type = "?"
        for line in content.split("\n"):
            if line.startswith("**Category**:"):
                category = line.split(":", 1)[1].strip()
            elif line.startswith("**Gain type**:"):
                gain_type = line.split(":", 1)[1].strip()

        display_name = trick_name.replace("-", " ").title()
        lines.append(f"| [{display_name}](tricks/{trick_name}.md) | {category} | {gain_type} |")

    return "\n".join(lines) + "\n"


# =============================================================================
# One-Shot Search (Simple API)
# =============================================================================

async def run_trick_search(
    search_query: str,
    max_tricks: int = 5,
) -> AsyncIterator:
    """
    Run the Trick Search Agent to find and document algorithmic tricks.

    Args:
        search_query: What to search for (e.g., "matrix decomposition tricks for SSMs")
        max_tricks: Maximum number of new tricks to document per run

    Yields:
        Agent messages as they stream in
    """
    existing_tricks = list_existing_tricks()
    existing_tricks_str = "\n".join(f"- {t}" for t in existing_tricks) if existing_tricks else "None yet"
    next_trick_number = get_next_trick_number()

    prompt = f"""Search for algorithmic tricks related to: {search_query}

## Existing Tricks (do not duplicate)
{existing_tricks_str}

## Next Available Trick Number
**{next_trick_number:03d}** - Use this as the starting number for new tricks

## Instructions
1. Ensure the papers/ directory exists (create it if needed)
2. Search the web for relevant papers, blog posts, and implementations
3. For each promising trick found (up to {max_tricks} new ones):
   - Download the source paper PDF to papers/
   - Use the Read tool to read the PDF and extract mathematical details
   - Document it in tricks/ using numbered format: {next_trick_number:03d}-trick-name.md
   - Increment the number for each new trick ({next_trick_number:03d}, {next_trick_number+1:03d}, {next_trick_number+2:03d}, etc.)
   - Include the number in the title: # {next_trick_number:03d}: Trick Name
4. Prioritize tricks that:
   - Have clear complexity improvements
   - Are applicable to sequence models (transformers, SSMs, linear attention)
   - Have been validated in practice (cited implementations)
"""

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            model="opus",
            system_prompt=TRICK_SEARCH_SYSTEM_PROMPT.format(tricks_dir=TRICKS_DIR, papers_dir=PAPERS_DIR),
            allowed_tools=["Read", "Write", "Glob", "Grep", "WebSearch", "WebFetch", "Bash"],
            permission_mode="acceptEdits",
            cwd=str(PROJECT_ROOT),
        )
    ):
        yield message


# =============================================================================
# Interactive Mode (Client-Based API)
# =============================================================================

async def run_interactive():
    """
    Run the Trick Search Agent in interactive mode.

    Uses ClaudeSDKClient for persistent sessions with subagent support.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY not found.")
        print("Set it in your environment or .env file.")
        print("Get your key at: https://console.anthropic.com/settings/keys\n")
        return

    # Define specialized subagent for reading papers
    agents = {
        "paper-reader": AgentDefinition(
            description=(
                "Use this agent to read and analyze academic papers (PDFs) or technical blog posts. "
                "The paper-reader can read local PDF files or fetch web content, extract algorithmic insights, and "
                "return structured summaries with mathematical details. Useful for deep-diving "
                "into specific papers found during web search or downloaded PDFs in papers/."
            ),
            tools=["WebFetch", "Read"],
            prompt=PAPER_READER_PROMPT,
            model="haiku",
        ),
    }

    options = ClaudeAgentOptions(
        system_prompt=TRICK_SEARCH_SYSTEM_PROMPT.format(tricks_dir=TRICKS_DIR, papers_dir=PAPERS_DIR),
        allowed_tools=["Read", "Write", "Glob", "Grep", "WebSearch", "WebFetch", "Bash", "Task"],
        permission_mode="acceptEdits",
        cwd=str(PROJECT_ROOT),
        agents=agents,
        model="opus",
    )

    print("\n" + "=" * 60)
    print(" Trick Search Agent (TSA)")
    print("=" * 60)
    print("\nSearch for algorithmic tricks from across the sciences.")
    print("Findings are documented in tricks/ folder.")
    print("Source papers are downloaded to papers/ folder.")
    print("\nExamples:")
    print("  - 'Find matrix decomposition tricks for SSMs'")
    print("  - 'Search for parallelization techniques in linear attention'")
    print("  - 'Look for numerical stability tricks from FlashAttention'")
    print("\nType 'summary' to see documented tricks, 'papers' to list downloads, 'exit' to quit.\n")

    async with ClaudeSDKClient(options=options) as client:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                break

            if user_input.lower() == "summary":
                print("\n" + get_trick_summary())
                continue

            if user_input.lower() == "papers":
                papers = list_existing_papers()
                if papers:
                    print("\nDownloaded papers:")
                    for p in papers:
                        print(f"  - {p}")
                else:
                    print("\nNo papers downloaded yet.")
                continue

            # Enhance the query with context
            existing_tricks = list_existing_tricks()
            existing_str = ", ".join(existing_tricks) if existing_tricks else "none"
            existing_papers = list_existing_papers()
            papers_str = ", ".join(existing_papers) if existing_papers else "none"

            enhanced_prompt = f"""{user_input}

(Already documented: {existing_str})
(Downloaded papers: {papers_str})"""

            await client.query(prompt=enhanced_prompt)

            print("\nAgent: ", end="", flush=True)
            async for msg in client.receive_response():
                # Print text content as it streams
                if hasattr(msg, 'content'):
                    for block in getattr(msg, 'content', []):
                        if hasattr(block, 'text'):
                            print(block.text, end="", flush=True)
                elif hasattr(msg, 'result'):
                    print(f"\n\nResult: {msg.result}")

            print()

    print("\nGoodbye!")


# =============================================================================
# CLI Entry Point
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Trick Search Agent - Find and document algorithmic efficiencies"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (e.g., 'matrix decomposition tricks for SSMs')"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-tricks", "-n",
        type=int,
        default=5,
        help="Maximum number of tricks to document per search (default: 5)"
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show summary of documented tricks and exit"
    )

    args = parser.parse_args()

    if args.summary:
        print("\n" + get_trick_summary())
        return

    if args.interactive:
        await run_interactive()
    elif args.query:
        print(f"\nSearching for: {args.query}\n")
        async for msg in run_trick_search(args.query, max_tricks=args.max_tricks):
            # Print progress
            if hasattr(msg, 'content'):
                for block in getattr(msg, 'content', []):
                    if hasattr(block, 'text'):
                        print(block.text, end="", flush=True)
            elif hasattr(msg, 'result'):
                print(f"\n\nDone! {msg.result}")
        print()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
