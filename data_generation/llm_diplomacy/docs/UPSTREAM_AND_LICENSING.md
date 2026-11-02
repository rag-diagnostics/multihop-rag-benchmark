# Third-party source, attribution, licensing, and dataset release

## Upstream project

- Project: [GoodStartLabs/AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy)
- Authorship credited by its README: Alex Duffy and Tyler Marques (Good Start Labs), with other contributors in the GitHub history.
- License: [Good Start Labs Non-Commercial License](https://github.com/GoodStartLabs/AI_Diplomacy/blob/main/LICENSE) (also present in the supplied source archive).
- Archived `.git/HEAD` resolves to `287d845d4ccc8ae67d47ab5131199dfbd5aee532`. The exact archived working-tree diff against that commit has not been verified.

## Why the code is referenced, not vendored

The license permits some noncommercial uses and permits distribution under conditions, **but its Additional Restrictions section also prohibits use of the software for the purpose of gathering datasets, annotations or telemetry for AI/ML systems**, including simulation data. It also places restrictions on some AI/ML uses in commercial applications.

**Your benchmark source-data generation appears close enough to the restricted language to warrant written clarification or permission.** We cannot determine your legal rights, the timing/applicability of any licence changes, or whether the original research was separately authorised from these ZIPs. This is a rights question, not a claim that infringement occurred.

Therefore this package:

- DOES include independently written, portable configuration and run-wrapper code, installation/usage directions, references, and prompt hashes.
- DOES NOT copy upstream `lm_game.py`, `ai_diplomacy/` source code or third-party prompt bodies.
- DOES NOT contain generated game histories, model outputs, benchmark datasets, actual API secrets, or Hugging Face tokens.
- DOES NOT claim that rerunning the upstream engine to gather a dataset is authorised. Resolve this before using the included runner for that purpose.

If permission is granted, preserve the source's copyright notices and license terms, credit the original authors, and document any modifications. Consider pinning an authorised upstream revision or separately distributing a permission-compliant patch, rather than copying a vast `.venv` and unrelated experiments.

## Dataset release is a separate decision

The ECIR Resource Track asks reviewers to have access to the **resource** at review time and discusses open availability/licensing. You do not necessarily need to commit large raw simulation outputs to GitHub, but a GitHub repository containing only generation scripts may not satisfy that expectation. Consider hosting a documented, appropriately licensed benchmark artifact in an archival repository and linking to it, or arranging approved reviewer access, **after confirming both upstream and source-data rights**.

The later paper stages (narrative conversion, decontamination, multi-hop composition, QA generation) have separate provenance and distribution considerations.
