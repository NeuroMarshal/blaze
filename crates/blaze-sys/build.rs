//! Builds kernels/ (CMake: CUDA C++ kernels and the C++ MoE engine) and links its static libraries.
//! BLAZE_CUDA_ARCH selects the GPU architecture (CMake syntax: 89, 120a, ...; default: the local GPU).
use std::env;
use std::path::PathBuf;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../../kernels");
    println!("cargo:rerun-if-changed={}", root.display());
    println!("cargo:rerun-if-env-changed=BLAZE_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    let arch = env::var("BLAZE_CUDA_ARCH").unwrap_or_else(|_| local_archs().unwrap_or_else(|| "native".into()));
    // CUDA_HOME, else the first toolkit found at the usual places (Arch: /opt/cuda; NVIDIA packages: /usr/local/cuda).
    let cuda = env::var("CUDA_HOME")
        .ok()
        .or_else(|| ["/opt/cuda", "/usr/local/cuda"].iter().find(|p| PathBuf::from(p).join("bin/nvcc").exists()).map(|p| p.to_string()))
        .expect("CUDA toolkit not found: set CUDA_HOME");
    let dst = cmake::Config::new(&root)
        .define("CMAKE_CUDA_COMPILER", format!("{cuda}/bin/nvcc"))
        .define("CMAKE_CUDA_ARCHITECTURES", &arch)
        .define("BLAZE_BUILD_TESTS", "OFF")
        .build_target("blaze_cpu")
        .build();
    let build = dst.join("build");
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-lib=static=blaze_cpu");
    println!("cargo:rustc-link-lib=static=blaze_kernels");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-search=native={cuda}/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

/// The local GPUs' architectures for CMake, with the architecture-specific variant (120a, 100a, 90a) where the GPU has
/// one: the FP4 / block-scaled MMA instructions exist only there ("native" builds the plain one). None without
/// nvidia-smi.
fn local_archs() -> Option<String> {
    let out = std::process::Command::new("nvidia-smi").args(["--query-gpu=compute_cap", "--format=csv,noheader"]).output().ok()?;
    let mut archs: Vec<String> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let (major, minor) = l.trim().split_once('.')?;
            let a = format!("{major}{minor}");
            Some(if matches!(a.as_str(), "90" | "100" | "101" | "120" | "121") { format!("{a}a") } else { a })
        })
        .collect();
    archs.sort();
    archs.dedup();
    (!archs.is_empty()).then(|| archs.join(";"))
}
