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
cannot be configured until a crate exists, so add a crates.io API token as the
`CRATES_IO_TOKEN` Actions secret in the `cargo` environment for this release.
The release workflow publishes `ffbpe-pat` first, waits for it to reach the
registry index, and then publishes `ffbpe` through its existing trusted
publisher. Afterward, configure trusted publishing for `ffbpe-pat` and remove
the token.

## First npm release

The three scoped packages must be created before npm trusted publishing can be
configured in their package settings. For the first release only:

1. Add a granular npm access token as the `NPM_TOKEN` Actions secret. It must
   be allowed to publish public packages under the `@tokn-ai` scope.
2. Ensure the `npm` GitHub environment allows the release job to run.
3. Push `v0.1.9`. The release workflow publishes these packages in dependency
   order:
   - `@tokn-ai/ffbpe@0.1.9`
   - `@tokn-ai/ffbpe-inspect@0.1.0`
   - `@tokn-ai/ffbpe-presets@0.1.0`

After the first publish, configure each package's npm trusted publisher with:

- organization: `tokn-ai`
- repository: `ffbpe`
- workflow filename: `release.yml`
- environment: `npm`
- allowed action: `npm publish`

Then remove `NPM_TOKEN`. The workflow uses GitHub OIDC for later releases and
requests provenance for every npm publication.

## Tag and publish

Only tag a commit on `master` after its CI checks pass:

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
