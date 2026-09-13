# Public-release checklist

Keep the repository private until every item below is complete.

- [ ] Choose and add a source-code license approved by all contributors.
- [ ] Confirm that redistribution of both derived score archives is permitted,
      and add the required dataset attribution and license notices.
- [ ] Add the final paper citation and archival identifier.
- [ ] Run `make reproduce` in a clean clone with the pinned Python packages.
- [ ] Run `make figure-check` and manually inspect every generated PDF before
      recording any intentional visual difference.
- [ ] Run `make privacy` and manually inspect Git-tracked text and PDF metadata.
- [ ] Confirm that no raw prompts, responses, credentials, private hostnames,
      local paths, unpublished notes, or collaborator-only artifacts are
      tracked.
- [ ] Review the complete Git history. This repository begins with a sanitized
      snapshot so source-project history is not inherited.
- [ ] Protect the default branch and enable dependency/security alerts.
- [ ] Only then change the GitHub repository visibility from private to public.

The score archives intentionally omit raw prompt and response text. That
reduces privacy risk but does not replace a redistribution-license review.
