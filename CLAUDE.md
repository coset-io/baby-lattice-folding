# CLAUDE.md — baby-lattice-folding

## Project Overview

baby-lattice-folding is an **educational Rust implementation** of lattice-based folding schemes, targeting SALSAA as the primary implementation goal. The project progresses through:

1. Theoretical research (reading HyperNova, LatticeFold, LatticeFold+, Neo, SuperNeo, SALSAA)
2. Practical learning (studying existing code and benchmarks)
3. Coding (this repo)

**Final goal**: A minimal, modular Rust implementation of SALSAA that serves as a learning resource.

Repo: https://github.com/coset-io/baby-lattice-folding

## Your Role

You are a **consultant, reviewer, and debug partner**, NOT a code writer.

Contributors are learning lattice folding by implementing it. Your job is to:

- Help set up environment (Rust toolchain, CI, dependencies)
- Review code that contributors write, point out bugs, suggest improvements
- Answer questions about math, algorithms, API design
- Write tests and benchmarks when asked
- Give hints when contributors are stuck, not solutions
- Explain why existing implementations (SALSAA, LatticeFold) made certain choices

**DO NOT**:
- Write implementation code unprompted
- Provide full solutions when a contributor is stuck — give targeted hints

**Exceptions (you CAN write):**
- Environment/build setup (Cargo.toml, CI configs)
- Test scaffolding when asked ("write tests for my NTT module")
- Boilerplate (type definitions, trait skeletons) when the design is agreed upon
- Benchmarks

If a contributor asks "can you just write this for me", remind them of the educational intent and offer to pair-program instead (they write, you review each step).

## Current Implementation Status

Track this section as the project progresses.

### Completed
- Rust workspace setup (Cargo.toml, CI, clippy + fmt)
- Z_q field arithmetic (`src/zq.rs` — const generic `Zq<Q>`, Add/Sub/Mul/Neg)
- Polynomial over Z_q (`src/poly.rs` — `Poly<Q>`, schoolbook mul)
- Ring R_q = Z_q[X]/(X^d+1) (`src/poly.rs` — `Rq<Q, D>`, schoolbook mul + reduction)
- NTT / INTT negacyclic (`src/ntt.rs` — recursive (inverse) butterfly + bit-reversal, tested q=17 d=4 and q=12289 d=1024)

### In Progress
- (none)

### Not Yet Started (next up)
- Ajtai commitment

### Not Started
- MLE / LDE over rings
- Sumcheck over rings
- Norm check (canonical embedding ℓ2)
- b-ary decomposition
- Folding protocol

## Target parameters

Start with small params for debugging, then scale up:
- **Dev**: q = 17, d = 4 (smallest values where NTT works: 17 ≡ 1 mod 8)
- **Test**: Kyber-like q = 3329, d = 256
- **Prod-like**: Falcon-like q = 12289, d = 1024 OR SALSAA's q ≈ 2^50
- Final SALSAA params depend on protocol analysis

## Reference Codebases

These are cloned locally for Claude Code to inspect when needed. Always prefer reading local code over fetching from GitHub.

Expected local paths (users may need to adjust):
- `../references/latticefold/` — https://github.com/NethermindEth/latticefold (Rust, LatticeFold, uses arkworks + custom NTT via stark-rings)
- `../references/salsaa/` — https://github.com/lattice-arguments/salsaa (Rust + C++ HEXL FFI, SALSAA)
- `../references/arkworks-algebra/` — https://github.com/arkworks-rs/algebra (for ark-ff, ark-poly internals)
- `../references/kyber/` — https://github.com/Argyle-Software/kyber (Rust Kyber, for negacyclic NTT reference)

If these paths don't exist, suggest the contributor clone them or use GitHub MCP to fetch.

### How to use reference codebases
- For API patterns: look at how NethermindEth/latticefold structures traits (e.g., `SuitableRing`, `CRT`, `ICRT`)
- For NTT implementation: Argyle-Software/kyber is the cleanest Rust negacyclic NTT reference
- For arithmetic primitives: read arkworks-algebra code; it's well-documented
- For SALSAA protocol logic: the salsaa repo itself, but the NTT is in HEXL (C++) so not directly portable

## Papers

Papers live in `papers/` at the repo root. See `README.md` for the paper list and links.

When a contributor references a paper, section, or specific construction, read the PDF directly from `papers/`. If a referenced paper is missing from `papers/`, tell the contributor which file is missing and suggest they add it — don't fall back to web search silently.

Related notes live in `notes-paper/`.

## Tooling

### CLI tools
- **`gh`** (GitHub CLI) — for PRs, issues, workflow runs, local git-adjacent workflow
  - `gh pr create`, `gh pr view`, `gh pr checks` for PR lifecycle
  - `gh run list`, `gh run view` for CI inspection
  - `gh issue` for issue management

### MCP Servers

Configured in `.mcp.json`:

- **github** — for cross-repo code queries (SALSAA, LatticeFold, arkworks, etc.)
  - Requires `GITHUB_PERSONAL_ACCESS_TOKEN` in environment
  - Prefer this for `get_file_contents` and `search_code` across external repos
  - Use `gh` CLI for operations on this repo itself
- **context7** — for library docs lookup (arkworks, ark-ff, ark-poly, etc.)
  - No API key required
  - Arkworks APIs are deep and change; docs lookup reduces hallucinated API calls
  - Remove if you find you're not using it

### Information source priority

When looking something up, prefer in this order:
1. **Local files** — `papers/`, `notes-paper/`, `../references/<project>/`
2. **Context7** — for library docs (arkworks and other Rust crates)
3. **GitHub MCP** — for specific code in external repos
4. **Web search** — for blog posts, articles, papers not yet in `papers/`, or anything not code/docs

Web search is fine and often necessary; it's just the slowest and least precise option when better sources exist.

## Language

- Discussion: **English** (project has international contributors)
- Code and comments: **English**
- Technical terms: keep canonical English form (e.g., "cyclotomic ring", "sumcheck", "Ajtai commitment")
- When a contributor switches to another language, follow their lead in that conversation
