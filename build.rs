use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=csrc/rvv_matrix.c");
    
    // 编译 C 代码
    let mut build = cc::Build::new();
    
    build
        .file("csrc/rvv_matrix.c")
        .flag_if_supported("-march=rv64gcv")  // 启用 RISC-V Vector 扩展
        .flag_if_supported("-mabi=lp64d");

    build.flag("-march=rv64gcv");
    build.compile("rvv_matrix");
    
    // 链接数学库
    println!("cargo:rustc-link-lib=m");
}