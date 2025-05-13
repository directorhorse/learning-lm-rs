use std::collections::btree_map::Range;

use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let dims = x.shape();
    let dims_len = dims.len();
    let vec_len = dims[dims_len-1];
    let vec_num = dims.iter().fold(1, |acc,x| acc*x)/vec_len;
    let y_mut = unsafe {
        y.data_mut()
    };
    let x_data = x.data();
    for i in 0..vec_num{
        // let mut temp = 0.0;
        // for j in 0..vec_len{
        //     temp += x_data[i*vec_len+j]*x_data[i*vec_len+j];
        // }
        let temp = &x_data[i*vec_len..(i*vec_len+vec_len)].iter().fold(0.0, |acc, xij| acc + xij.powf(2.0));
        let sqrt_data = (temp/(vec_len as f32) + epsilon).sqrt();
        for j in 0..vec_len{
            y_mut[i*vec_len+j] = x_data[i*vec_len+j]*w.data()[j]/sqrt_data;
        }
    }
    //todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {

    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    let sigmoid_x = _x.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect::<Vec<_>>();
    for i in 0..len {
        _y[i] = sigmoid_x[i] * _x[i] * _y[i];
    }
    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // todo!("实现 matmul_transb，计算前做一些必要的检查会帮助你后续调试");
    let a_shape = a.shape(); // (..., m, k)
    let b_shape = b.shape(); // (..., n, k)
    let c_shape = c.shape(); // (..., m, n)
    let m = c_shape[c_shape.len() - 2];
    let n = c_shape[c_shape.len() - 1];
    let k = a_shape[a_shape.len() - 1];
    assert!(a_shape[a_shape.len() - 2] == m);
    assert!(b_shape[b_shape.len() - 2] == n);
    assert!(b_shape[b_shape.len() - 1] == k);

    // no broadcast now
    let batch = c.size() / m / n;
    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    for b in 0..batch {
        let offset_c = b * m * n;
        let offset_a = b * m * k;
        let offset_b = b * n * k;
        for i in 0..m {
            for j in 0..n {
                let idx_c = offset_c + i * n + j;
                _c[idx_c] = beta * _c[idx_c];
                let mut tmp: f32 = 0.0;
                for e in 0..k {
                    tmp += _a[offset_a + i * k + e] * _b[offset_b + j * k + e];
                }
                _c[idx_c] = _c[idx_c] + alpha * tmp;
            }
        }
    }
}

pub fn transb(c: &mut Tensor<f32>,a: &Tensor<f32>){
    let _a = a.data(); 
    let _c = unsafe {
        c.data_mut()
    };
    for i in 0..a.shape()[1]{
        for j in 0..a.shape()[0]{
            _c[i*a.shape()[0]+j] = _a[j*a.shape()[1]+i];
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    for data in y.data(){
        println!("{}",data)
    }
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
    // let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.,5.,6.], &vec![2, 3]);
    // let x = Tensor::<f32>::new(vec![1., 2., 3., 4.,5.,6.], &vec![2, 3]);
    // let w = Tensor::<f32>::new(vec![1., 2.,3.], &vec![3]);
    // rms_norm(&mut y, &x, &w, 1e-6);
    // for data in y.data(){
    //     println!("{}",data)
    // }
}

#[test]
fn test_matmul_transb() {
    // let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    // let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    // // let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    // let b = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1.], &vec![2, 3]);
    // matmul_transb(&mut c, 1., &a, &b, 1.);
    // assert!(c.close_to(
    //     // &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
    //     &Tensor::<f32>::new(vec![7., 8., 18., 19.], &vec![2, 2]),
    //     1e-3
    // ));
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.,5.,6.,7.,8.], &vec![2, 4]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.,7.,8.,9.,10.,11.,12.], &vec![4,3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    for data in c.data(){
        println!("{}",data)
    }
}
