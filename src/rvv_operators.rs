use std::os::raw::{c_float, c_int};

// 声明外部 C 函数
unsafe extern "C" {
    fn matrix_transpose_multiply_basic(
        A: *const c_float, 
        B: *const c_float, 
        C: *mut c_float, 
        beta: c_float, 
        alpha: c_float,
        M: c_int, 
        N: c_int, 
        K: c_int
    );
}

// 安全的 Rust 包装器
pub struct RvvMath;

impl RvvMath {
    pub fn matrix_transpose_multiply_basic(
        A: &[f32], 
        B: &[f32], 
        C: &mut [f32], 
        beta: f32, 
        alpha: f32,
        M: i32, 
        N: i32, 
        K: i32,
        offset_a: usize,
        offset_b: usize,
        offset_c: usize
    ) -> Result<(), &'static str> {
        // 参数验证
        if A.len() != (M * K) as usize {
            return Err("矩阵 A 的大小不匹配");
        }
        if B.len() != (N * K) as usize {
            return Err("矩阵 B 的大小不匹配");
        }
        if C.len() != (M * N) as usize {
            return Err("输出矩阵 C 的大小不匹配");
        }
        
        unsafe {
            matrix_transpose_multiply_basic(
                A.as_ptr().add(offset_a), 
                B.as_ptr().add(offset_b), 
                C.as_mut_ptr().add(offset_c), 
                beta, 
                alpha, 
                M, 
                N, 
                K
            );
        }
        
        Ok(())
    }
}