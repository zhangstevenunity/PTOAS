---
name: ptoas-publish-pr
description: Publish PTOAS changes as a GitCode pull request by default, with GitHub supported as a compatibility path. Use when turning intended local PTOAS edits into a branch, commit, push, and PR.
---

# PTOAS Publish PR

## Overview

Use this skill to safely publish PTOAS work from the local checkout. Unless the user explicitly requests otherwise, publish the source branch from the user's personal fork and open the PR against the canonical repository. Confirm the intended scope, keep unrelated files out of the commit, and never rewrite a protected canonical-repository branch.

GitCode is the default PTOAS hosting platform: use a personal fork as the push remote and `gitcode` (`cann/pto-as`) as the canonical repository. Do not push feature branches directly to the canonical repository by default; canonical branches may reject force-pushes and complicate squash/review workflows. Use GitHub only when the user explicitly requests it or the task targets the GitHub mirror.

## Preconditions

For GitCode, inspect remotes and available repositories first:

```bash
git remote -v
gc repo list
```

Create a personal GitCode fork when one is missing:

```bash
gc repo fork cann/pto-as
```

Add its SSH URL as `gitcode-fork` and use `gitcode` for the canonical repository.

For GitHub compatibility, check authentication only when the user explicitly targets GitHub:

```bash
gh auth status
```

In this repository, expect:

- `origin` to be the user's GitHub fork
- `upstream` to be `hw-native-sys/PTOAS`

## Workflow

1. Confirm branch and worktree state:

```bash
git branch --show-current
git status -sb
```

If the worktree contains unrelated tracked or untracked files, do not include them by default. Stage only the files that belong in the PR.

2. If currently on `main`, create a feature branch:

```bash
git switch -c feature/<short-description>
```

If already on a non-default branch for the task, stay on that branch.

3. Stage only the intended files. Prefer explicit paths:

```bash
git add -- <path1> <path2> ...
git status --short
git diff --cached --stat
```

Use `git add -A` only when the entire worktree is intentionally part of the PR.

4. Run the most relevant validation before commit. For `.agents/skills` work, validate each affected skill with `quick_validate.py`.

5. Commit with a short message:

```bash
git commit -m "<terse-summary>"
```

6. Push to the personal fork (unless the user explicitly requested a canonical-repository branch):

```bash
git push -u origin "$(git branch --show-current)"
```

For GitCode:

```bash
git push -u gitcode-fork "$(git branch --show-current)"
```

7. Open or update the PR:

- Prefer a draft PR unless the user explicitly wants ready-for-review.
- If the current branch is already attached to an open PR, pushing new commits updates that PR automatically.
- When creating a new PR, target `upstream/main` if the branch lives on the fork and the canonical repo is `hw-native-sys/PTOAS`.
- For GitCode, create a cross-repository PR with `gc pr create -R cann/pto-as --fork <user>/pto-as --head <branch> --base master`.
- Use the GitHub app connector when available. Use `gh pr create` as a fallback after `gh auth status` confirms login.

Example fallback:

```bash
gh pr create --repo hw-native-sys/PTOAS --base main --head "$(git branch --show-current)" --draft --title "<summary>" --body-file <path-to-body>
```

## Safety Checks

- Never stage unrelated user changes silently.
- Never clean or reset the worktree just to make publishing easier.
- Keep untracked scratch directories out of the PR unless the user explicitly wants them included.
- Re-check `git status --short` after commit; only unrelated leftover files should remain.
- If `git push` fails because of auth, confirm `gh auth status` and ask the user to run `gh auth login` when needed.

## Publish Summary

Before finishing, report:

- branch name
- commit SHA
- whether the push succeeded
- PR URL if one exists
- what validation was run
