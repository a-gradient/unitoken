# Releasing FFBPE

The repository tag version is shared by the Rust crate, Python package,
WebAssembly crate, and core npm package. `ffbpe-pat` and the npm companion
packages have independent versions, but every normal release publishes a new
version of each package named by the release workflow.

## Prepare

1. Update `Cargo.toml`, `pyproject.toml`, `crates/ffbpe-wasm/Cargo.toml`, and
   `packages/ffbpe/package.json` to the new tag version.
2. Bump `crates/ffbpe-pat/Cargo.toml` to a new published version and update the
   root `ffbpe-pat` dependency requirement to that exact version.
3. Bump every npm manifest that `release.yml` publishes: the core
   `packages/ffbpe/package.json` at the tag version, plus new versions for
   `packages/ffbpe-inspect/package.json` and
   `packages/ffbpe-presets/package.json`. Update both companion
   `@tokn-ai/ffbpe` peer ranges, for example, to `>=0.1.10 <0.2.0`.
   The normal npm publish job does not skip already-published versions.
4. Refresh the local `Cargo.lock` and `uv.lock` files; both are intentionally
   ignored by this library repository.
5. Add the release notes to `CHANGELOG.md`.
6. Run `.github/scripts/verify-release-versions.py` and the complete CI suite.

Before tagging, ensure the GitHub `cargo`, `npm`, and `pypi` environments have
their trusted-publishing configuration. PyPI publishing uses the `pypi`
environment in `.github/workflows/wheels.yml`.

## Historical v0.1.9 bootstrap

The `v0.1.9` release established package identities required by trusted
publishing. This was a one-release bootstrap, not normal release preparation.

`ffbpe 0.1.9` depended on the new `ffbpe-pat 0.1.0` crate. Because trusted
publishing cannot be configured until a crate exists, a maintainer manually
published `ffbpe-pat 0.1.0` to create its package identity.

The `ffbpe-pat` GitHub Actions trusted publisher on crates.io uses:

- organization: `tokn-ai`
- repository: `ffbpe`
- workflow filename: `release.yml`
- environment: `cargo`

The existing `ffbpe` trusted publisher uses the same values. Revoke any manual
bootstrap token after configuring trusted publishing. Do not store a crates.io
publishing token in GitHub.

The three scoped npm packages also had to exist before trusted publishing could
be configured. A maintainer manually published `@tokn-ai/ffbpe 0.1.9`,
`@tokn-ai/ffbpe-inspect 0.1.0`, and `@tokn-ai/ffbpe-presets 0.1.0` to create
their package identities.

After the package identities existed, the `npm` GitHub environment and each
package's npm trusted publisher were configured with:

- organization: `tokn-ai`
- repository: `ffbpe`
- workflow filename: `release.yml`
- environment: `npm`
- allowed action: `npm publish`

Revoke any temporary manual bootstrap token after configuring trusted
publishing. Do not store an npm publishing token in GitHub.

The `v0.1.9` release workflow verifies these four exact manual publications,
skips their publish steps, and publishes only `ffbpe 0.1.9` to crates.io and
PyPI through their existing trusted publishers. It is a one-release exception.
Later releases publish `ffbpe-pat` and `ffbpe` to crates.io in dependency
order, then all three npm packages. Existing release versions are errors rather
than silently skipped.

## Tag and publish

Only tag the committed release preparation on `master` after the version
verifier and full CI pass, every package version is new in its registry, and
every trusted publisher is configured:

```sh
git tag -a v0.1.10 -m "FFBPE 0.1.10"
git push origin v0.1.10
```

The tag starts three publication paths:

- `.github/workflows/release.yml`: crates.io and npm
- `.github/workflows/wheels.yml`: PyPI wheels and source distribution
- GitHub Pages: deployment of the tagged code still follows normal `master`
  deployment, not the release tag

After all registry jobs pass, create the GitHub release from the matching
`CHANGELOG.md` section and verify the published versions on crates.io, PyPI, and
npm.
