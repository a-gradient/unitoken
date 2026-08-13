# Releasing FFBPE

The repository tag version is shared by the Rust crate, Python package,
WebAssembly crate, and core npm package. Companion packages follow their own
versions; for the `v0.1.9` release, `ffbpe-pat` and both npm companions start at
`0.1.0`.

## Prepare

1. Update `Cargo.toml`, `pyproject.toml`, `crates/ffbpe-wasm/Cargo.toml`, and
   `packages/ffbpe/package.json` to the same version.
2. Update the companion packages' `@tokn-ai/ffbpe` peer dependency range.
3. Refresh the local `Cargo.lock` and `uv.lock` files; both are intentionally
   ignored by this library repository.
4. Add the release notes to `CHANGELOG.md`.
5. Run `.github/scripts/verify-release-versions.py` and the complete CI suite.

## First crates.io dependency release

`ffbpe 0.1.9` depends on the new `ffbpe-pat 0.1.0` crate. Trusted publishing
cannot be configured until a crate exists. Before tagging the release, a
maintainer must manually publish a non-release bootstrap version such as
`ffbpe-pat 0.0.0-bootstrap.0` from a temporary copy of the crate. Do not commit
the bootstrap version to the repository.

After the package identity exists, add a GitHub Actions trusted publisher for
`ffbpe-pat` on crates.io with:

- organization: `tokn-ai`
- repository: `ffbpe`
- workflow filename: `release.yml`
- environment: `cargo`

The existing `ffbpe` trusted publisher should use the same values. Revoke the
manual bootstrap token after configuring trusted publishing. Do not store a
crates.io publishing token in GitHub.

## First npm release

The three scoped packages must be created before npm trusted publishing can be
configured in their package settings. Before tagging the release, a maintainer
must manually publish each package from a temporary copy with the non-release
version `0.0.0-bootstrap.0`. Do not commit bootstrap versions to the
repository.

After the package identities exist, create the `npm` GitHub environment and
configure each package's npm trusted publisher with:

- organization: `tokn-ai`
- repository: `ffbpe`
- workflow filename: `release.yml`
- environment: `npm`
- allowed action: `npm publish`

Revoke any temporary manual bootstrap token after configuring trusted
publishing. Do not store an npm publishing token in GitHub.

The release workflow verifies both crates.io identities and all three npm
package identities before requesting any OIDC credential. If every identity
exists, it publishes all release versions in dependency order. Existing
release versions are treated as errors rather than silently skipped.

## Tag and publish

Only tag a commit on `master` after its CI checks pass, all first versions are
available in their registries, and every trusted publisher is configured:

```sh
git tag -a v0.1.9 -m "FFBPE 0.1.9"
git push origin v0.1.9
```

The tag starts three publication paths:

- `.github/workflows/release.yml`: crates.io and npm
- `.github/workflows/wheels.yml`: PyPI wheels and source distribution
- GitHub Pages: deployment of the tagged code still follows normal `master`
  deployment, not the release tag

After all registry jobs pass, create the GitHub release from the matching
`CHANGELOG.md` section and verify the published versions on crates.io, PyPI, and
npm.
