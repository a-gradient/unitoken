fn main() {
  println!("cargo:rerun-if-changed=build.rs");
  println!(
    "cargo:warning=unitoken has moved to ffbpe and is no longer maintained; \
     migrate with `cargo add ffbpe`"
  );
}
