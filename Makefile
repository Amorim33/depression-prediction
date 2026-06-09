.PHONY: lint typecheck test db-check-setembrobr manifest-setembrobr export-sequences-setembrobr train-tabular-oof-setembrobr train-seq-oof-setembrobr audit-oof-setembrobr select-ensemble-setembrobr evaluate-test-setembrobr reproduce-setembrobr

lint:
	bun run lint

typecheck:
	bun run typecheck

test:
	bun run test

db-check-setembrobr:
	bun run db-check-setembrobr

manifest-setembrobr:
	bun run manifest-setembrobr

export-sequences-setembrobr:
	python3 scripts/export_sequences_setembrobr.py --config configs/setembrobr.seed42.strict-blind.json

train-tabular-oof-setembrobr:
	python3 scripts/train_tabular_oof_setembrobr.py --config configs/setembrobr.seed42.strict-blind.json

train-seq-oof-setembrobr:
	python3 scripts/train_seq_oof_setembrobr.py --config configs/setembrobr.seed42.strict-blind.json

audit-oof-setembrobr:
	bun run audit-oof-setembrobr

select-ensemble-setembrobr:
	bun run select-ensemble-setembrobr

evaluate-test-setembrobr:
	bun run evaluate-test-setembrobr

reproduce-setembrobr: lint typecheck test db-check-setembrobr manifest-setembrobr export-sequences-setembrobr train-tabular-oof-setembrobr train-seq-oof-setembrobr audit-oof-setembrobr select-ensemble-setembrobr evaluate-test-setembrobr

