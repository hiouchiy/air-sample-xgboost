#!/usr/bin/env bash
# One-time Unity Catalog setup for the XGBoost AI Runtime demo.
#
# Creates the schema and the "predictions" volume this demo writes to. Idempotent.
#
# Usage:
#   CATALOG=main PROFILE=air ./setup.sh
#
#   CATALOG  Unity Catalog you can write to (required)             e.g. main
#   PROFILE  Databricks CLI profile from `databricks auth login`   (default: DEFAULT)
#   SCHEMA   Schema name (default: air_samples)
set -euo pipefail

: "${CATALOG:?Set CATALOG to a Unity Catalog you can write to, e.g. CATALOG=main PROFILE=air ./setup.sh}"
PROFILE="${PROFILE:-DEFAULT}"
SCHEMA="${SCHEMA:-air_samples}"

echo "Using catalog=$CATALOG schema=$SCHEMA profile=$PROFILE"
databricks current-user me --profile "$PROFILE" >/dev/null

echo "Creating schema $CATALOG.$SCHEMA ..."
databricks schemas create "$SCHEMA" "$CATALOG" --profile "$PROFILE" >/dev/null 2>&1 \
  && echo "  created." || echo "  already exists (ok)."

echo "Creating volume $CATALOG.$SCHEMA.predictions ..."
databricks volumes create "$CATALOG" "$SCHEMA" predictions MANAGED --profile "$PROFILE" >/dev/null 2>&1 \
  && echo "  created." || echo "  already exists (ok)."

cat <<EOF

Done. Now point the demo at this catalog by either:
  - editing the UC_CATALOG / UC_SCHEMA default lines near the top of each 02_cli/*.py, or
  - prefixing the command in each 02_cli/*.yaml, e.g.:
      command: UC_CATALOG=$CATALOG python \$CODE_SOURCE_PATH/02_cli/01_train_singlegpu.py
Then run:  COPYFILE_DISABLE=1 air run --file 02_cli/train_singlegpu.yaml --watch --profile $PROFILE
EOF
