# Security policy

## Supported use

FinGuard AI is an educational prototype, not a production banking system. Use only sanitized or approved benchmark data.

## Reporting a vulnerability

Please do not publish exploitable details in a public issue. Open a private GitHub security advisory for this repository and include:

- affected component and version or commit;
- reproduction steps or proof-of-concept;
- impact assessment;
- suggested remediation when available.

Do not include credentials, private transaction data, or real customer information in reports.

## Secret handling

The application loads environment variables and does not require credentials for the core demo. API bearer keys and optional OpenAI/AWS credentials must remain in local `.env` files or deployment secret stores, never in source control.
