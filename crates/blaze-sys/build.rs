//! Builds kernels/ (CMake: CUDA C++ kernels and the C++ MoE engine) and links its static libraries.
//! BLAZE_CUDA_ARCH selects the GPU architecture (CMake syntax: 89, 120a, ...; default: the local GPU).
use std::env;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../kernels");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-env-changed=BLAZE_CUDA_ARCH");
    let arch = env::var("BLAZE_CUDA_ARCH").unwrap_or_else(|_| "native".into());
    let dst = cmake::Config::new(&root)
        .define("CMAKE_CUDA_ARCHITECTURES", &arch)
        .define("BLAZE_BUILD_TESTS", "OFF")
        .build_target("blaze_cpu")
        .build();
    let build = dst.join("build");
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-lib=static=blaze_cpu");
    println!("cargo:rustc-link-lib=static=blaze_kernels");
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/opt/cuda".into());
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-search=native={cuda}/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}
