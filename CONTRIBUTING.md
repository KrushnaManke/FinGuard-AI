# Contributing

Thanks for helping improve FinGuard AI. Keep contributions focused, reproducible, and safe for financial-risk learning.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[app,api,test]"
python -m pytest -p no:cacheprovider -q
```

## Pull requests

1. Create a focused branch.
2. Add or update tests for behavior changes.
3. Run formatting or lint checks available in the repository.
4. Run the complete test suite.
5. Explain the user impact and include screenshots for dashboard changes.
6. Never commit credentials, private customer data, model secrets, or generated `.venv` files.

## Responsible data and model changes

- Preserve the educational and decision-support scope.
- Report benchmark-specific metrics honestly.
- Keep model outputs separate from final human decisions.
- Include a clear provenance note for data or policy sources.
- Do not add production claims, automated blocking, or false precision.
