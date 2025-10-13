---
description: Commit and push all changes in the project
---

You are asked to commit and push all changes in the project.

Follow these steps:

1. Run `git status` to show all untracked and modified files
2. Run `git diff` to show the changes (both staged and unstaged)
3. Run `git log -5 --oneline` to show recent commit history for context

4. Analyze the changes and draft a clear, descriptive commit message that:
   - Summarizes the nature and purpose of the changes
   - Follows the commit message style used in this repository
   - Is concise but informative (1-3 sentences)

5. Stage all changes using `git add .`

6. Create the commit with the message, including the Claude Code footer:
   ```
   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>
   ```

7. Push the changes to the remote repository using `git push`

8. Confirm the push was successful by running `git status`

IMPORTANT:
- Do NOT skip hooks or use --no-verify
- Do NOT use --force or --force-with-lease
- If there are merge conflicts or push fails, report the error and stop
- Use a HEREDOC for the commit message to ensure proper formatting
