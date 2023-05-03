## PR Naming Convention

When creating a pull request, please follow this naming convention:

```
<job>(module): description
```

### Job

Use one of the following keywords to describe the nature of the changes made in the PR:

- `feat`: add new features to the code base
- `fix`: fix bugs
- `refactor`: refactor code to make it more user-friendly
- `perf`: optimize code for faster performance
- `build`: modify dependencies (requirements.txt, environments, etc.)

If there is more than one job, separate them with a comma. For example: `feat(evaluation,model): add GPT-J pipeline`.

### Module

Specify which module(s) the changes were made in. Use one or more of the following keywords:

- `core`
- `model`
- `evaluation`
- `data`

If there is more than one module, separate them with a comma. For example: `feat,build(model): add flash attention`.