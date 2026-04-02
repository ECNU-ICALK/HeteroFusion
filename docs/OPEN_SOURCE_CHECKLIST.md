# HeteroFusion Open-Source Checklist

This checklist is designed for the public GitHub release of the HeteroFusion codebase.

## 1. Repository Boundary

- Keep the **code release** focused on the `HeteroFusion/` tree.
- Keep the **paper source** (`H2/`) as a separate paper repository or a separate archival folder if needed.
- Do not mix LaTeX build artifacts, experiment outputs, and release code in the public default branch.

## 2. Identity Fields to Finalize

- Replace placeholder citation metadata in `README.md`.
- Add the public paper link once the arXiv record is finalized.
- Add author names, affiliations, and contact information if desired.
- Decide whether the public repository name is `HeteroFusion` or something matching the final paper title.

## 3. Licensing and Redistribution

- Choose and add a repository license.
- Verify that the vendored `llamafactory/` code can be redistributed in the intended form.
- Verify redistribution rights for any included sample data.
- Verify that released adapters or checkpoints are compatible with the licenses of the underlying base models.
- If some assets cannot be redistributed, keep only scripts and document external download steps.

## 4. Clean the Repository

- Remove `__pycache__/`
- Remove `outputs/`
- Remove generated `logs/`
- Remove machine-specific temp files such as `.DS_Store`
- Remove any private paths, usernames, tokens, or internal notes from scripts and configs

## 5. Reproducibility

- Confirm that every paper table or figure maps to a released config or script.
- Confirm that the environment setup in `README.md` and `docs/REPRODUCE.md` works on a fresh machine.
- Document the expected folder structure for `MODEL_ROOT` and `ADAPTER_ROOT`.
- Decide whether to release evaluation scripts or only training/fusion scripts.
- Decide whether to release fused adapters, source experts, or only the fusion framework.

## 6. Data Policy

- List which datasets are included directly in the repository.
- List which datasets require external download.
- Document preprocessing assumptions for the JSON files under `data/sample/`.
- Add dataset citations and license notes if you keep sample subsets in the repository.

## 7. GitHub Presentation

- Keep the framework figure at the top of the README.
- Keep one short setup section and one short quick-start section.
- Link detailed reproduction notes from the README instead of overloading the front page.
- Add issue templates and a contributing guide if you expect external participation.

## 8. Recommended Release Order

1. Finalize the public README, citation, and license.
2. Remove non-release artifacts and verify `.gitignore`.
3. Test one config from a clean environment.
4. Create the GitHub repository and push the cleaned codebase.
5. Add the arXiv link, release tag, and optional pretrained artifacts.

## 9. Nice-to-Have Additions

- `environment.yml` or locked requirements file
- `LICENSE`
- `CONTRIBUTING.md`
- `assets/` for more paper figures
- model-card style documentation if fused adapters are released

