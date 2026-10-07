use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_vector_math)");
    println!("cargo:rerun-if-changed=rust/vector_math.c");
    println!("cargo:rerun-if-env-changed=CC");
    let enabled = env::var_os("CARGO_FEATURE_VECTOR_MATH").is_some()
        && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
        && env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("gnu")
        && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64")
        && env::var("HOST") == env::var("TARGET");
    if !enabled {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo OUT_DIR"));
    let object = out.join("vector_math.o");
    let status = Command::new(env::var_os("CC").unwrap_or_else(|| "cc".into()))
        .args(["-O3", "-fPIC", "-std=c11", "-c", "rust/vector_math.c", "-o"])
        .arg(&object)
        .status()
        .expect("C compiler for vector-math feature");
    assert!(status.success(), "vector-math C compilation failed");
    let status = Command::new("ar")
        .arg("crs")
        .arg(out.join("libprusie_vector_math.a"))
        .arg(object)
        .status()
        .expect("archiver for vector-math feature");
    assert!(status.success(), "vector-math archive failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=prusie_vector_math");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
    println!("cargo:rustc-cfg=has_vector_math");
}
