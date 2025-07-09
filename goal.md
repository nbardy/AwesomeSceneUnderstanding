# Project Goals: Unified Paper Organization System

## Objective
Unify two disjoint paper dumps (`initial_dump/` and `new_paper_batch/`), remove duplicates, and create a standardized organizational system for scene understanding and related research papers.

## Current State
- **Initial Dump**: Contains 1 `paper_index.md` + 5 deep review files focusing on Gaussian Splatting papers
- **New Paper Batch**: Contains 5 `Paper_index_{1-5}.md` files covering video understanding, self-supervised learning, and broader scene understanding topics
- **Format Inconsistency**: Different naming conventions, link structures, and organizational patterns

## Target Outputs

### 1. Table_Of_Contents.md
- **Purpose**: Master index organized by top-level categories
- **Structure**: 
  - Paper title as link to project page (or paper if no project page)
  - `[Deep Review]` link to relevant section in deep review files
  - `[Brief Summary]` link to relevant section in paper index files
- **Categories**: Auto-generated based on paper tags/topics

### 2. Final_batch/paper_index_{n}.md Files
- **Purpose**: Standardized paper summaries with consistent format
- **Content**: Brief summaries, links, tags, key innovations
- **Sharding**: Multiple files to manage large paper collections efficiently

### 3. Final_batch/deep_review_{n}.md Files
- **Purpose**: In-depth technical analysis and reviews
- **Content**: Detailed methodology, results, comparisons, implications
- **Alignment**: Same sharding strategy as paper indices

## Success Criteria
- Zero duplicate papers across all files
- Consistent naming and linking conventions
- Navigable cross-reference system between all three output types
- Scalable sharding strategy for future paper additions
- Preserved existing deep review content from initial dump