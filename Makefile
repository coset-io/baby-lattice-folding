.PHONY: check fix fmt clippy test

check: fmt clippy test

fix: fmt
	cargo clippy --fix --allow-dirty --allow-staged

fmt:
	cargo fmt

clippy:
	cargo clippy -- -D warnings

test:
	cargo test
